"""
Brine Shrimp Detector and Counter - CNN Deep Learning Model
Full Training and Inference Pipeline Combined
Dr. Erik Duboue's Lab
"""

""" LOAD LIBRARIES """
import os
import csv
import json
import random
import math
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pillow_heif import register_heif_opener
from tqdm import tqdm
from PIL import Image, ImageDraw, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

""" CONFIGURATION """
# Root folder where this code resides
ROOT = Path(__file__).resolve().parent
# Path to the Label Studio annotations JSON file
PATH_JSON = ROOT / "brine_annotations.json"
# Folder containing manually labeled training images
PATH_IMAGES_ROOT = ROOT / "prelabeled_training_images"
# Folder containing unlabeled images for inference
PATH_UNLABELED   = ROOT / "unlabeled_images"
# Root folder where all run outputs are stored
SAVE_ROOT = ROOT / "shrimp_runs"
# Ensure run output folder exists
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# Create a per run directory named with the current date
TODAY_STR = datetime.now().strftime("%Y-%m-%d")
RUN_DIR = SAVE_ROOT / TODAY_STR
# Ensure the per run directory exists
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Best model file for general users
BEST_MODEL_PATH = SAVE_ROOT / "brineshrimp_best_model.pth"

# Training hyperparameters 
NUM_EPOCHS = 30
BATCH_SIZE = 2
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
VAL_SPLIT = 0.25
CONF_THRESH = 0.85

# Enable HEIC support if available
try:
    register_heif_opener()
    HEIF_ENABLED = True
except Exception:
    HEIF_ENABLED = False
    print("[heif] pillow-heif not found or failed to load.")
    print("HEIC/HEIF images will be skipped unless converted.")

# Set of allowed image suffixes
IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}
if HEIF_ENABLED:
    IMG_SUFFIXES.update({".heic", ".heif"})
IMG_SUFFIXES |= {s.upper() for s in IMG_SUFFIXES}

""" UTILITIES """
def is_image_ok(p: Path) -> bool:
    """Return True if PIL can open and verify image."""
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False

def _resolve_image_path(img_rel: str, root: Path) -> Optional[Path]:
    """Try to locate corresponding image file for given JSON path."""
    # First try direct basename under the root
    base = os.path.basename(img_rel)
    p = root / base
    if p.is_file():
        return p

    # Then try only the tail part after a dash
    tail = base.split("-")[-1] if "-" in base else base
    p2 = root / tail
    if p2.is_file():
        return p2

    # Build candidate filenames
    tail_lower = tail.lower()
    swaps = {tail_lower}
    if tail_lower.endswith(".jpeg"):
        swaps.add(tail_lower.replace(".jpeg", ".jpg"))
    if tail_lower.endswith(".jpg"):
        swaps.add(tail_lower.replace(".jpg", ".jpeg"))

    # Search directory for file matching one of the candidate tails
    for f in root.iterdir():
        if f.is_file():
            if f.name.lower() == tail_lower or f.name.lower() in swaps:
                return f
    return None

def load_labelstudio_rects(json_path: Path) -> Dict[str, List[List[float]]]:
    """Parse Label Studio JSON and return mapping of image paths to bounding boxes."""
    # Load JSON file contents
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_to_boxes = {}
    missing = []

    # Iterate through each annotation record
    for item in data:
        img_rel = item.get("image", "")
        resolved = _resolve_image_path(img_rel, PATH_IMAGES_ROOT)
        # Track images that could not be found
        if resolved is None:
            missing.append(os.path.basename(img_rel))
            continue

        labels = item.get("label") or []
        boxes_abs = []

        # Convert each labeled rectangle from percentage coordinates to pixel coordinates
        for ann in labels:
            ow, oh = ann["original_width"], ann["original_height"]
            x = (ann["x"] / 100.0) * ow
            y = (ann["y"] / 100.0) * oh
            w = (ann["width"] / 100.0) * ow
            h = (ann["height"] / 100.0) * oh

            # Compute bounding box as xmin, ymin, xmax, ymax
            xmin = max(0, min(x, ow - 1))
            ymin = max(0, min(y, oh - 1))
            xmax = max(1, min(x + w, ow))
            ymax = max(1, min(y + h, oh))

            # Store only valid boxes
            if xmax > xmin and ymax > ymin:
                boxes_abs.append([xmin, ymin, xmax, ymax])

        img_to_boxes[str(resolved)] = boxes_abs

    # Summarize any missing images
    if missing:
        print(f"[load_labelstudio_rects] Missing on disk ({len(missing)}). First few: {missing[:5]}")
    print(f"[load_labelstudio_rects] Resolved {len(img_to_boxes)} labeled images.")
    return img_to_boxes

def list_unlabeled_images(folder: Path) -> List[Path]:
    """Return a list of verified unlabeled images."""
    # If folder does not exist or is not a directory, return empty list
    if not folder.is_dir():
        return []

    # Collect all image files with supported extensions
    all_imgs = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_SUFFIXES]
    good, bad = [], []

    # Validate each image via PIL
    for p in all_imgs:
        if is_image_ok(p):
            good.append(p)
        else:
            bad.append(p.name)

    # Warn about unreadable images
    if bad:
        print(f"[unlabeled validator] Skipped {len(bad)} unreadable unlabeled images. First few: {bad[:5]}")
    return good

""" AUGMENTATION HELPERS """
def hflip_boxes(boxes_xyxy: torch.Tensor, img_w: int) -> torch.Tensor:
    """Flip bounding boxes horizontally."""
    # If no boxes exist, return original tensor
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy
    
    # Compute new x coordinates after horizontal flip
    xmin = img_w - boxes_xyxy[:, 2]
    xmax = img_w - boxes_xyxy[:, 0]
    return torch.stack([xmin, boxes_xyxy[:, 1], xmax, boxes_xyxy[:, 3]], dim=1)

def vflip_boxes(boxes_xyxy: torch.Tensor, img_h: int) -> torch.Tensor:
    """Flip bounding boxes vertically."""
    # If no boxes exist, return original tensor
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy
    
    # Compute new y coordinates after vertical flip
    ymin = img_h - boxes_xyxy[:, 3]
    ymax = img_h - boxes_xyxy[:, 1]
    return torch.stack([boxes_xyxy[:, 0], ymin, boxes_xyxy[:, 2], ymax], dim=1)

""" EVALUATION METRICS """
def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Intersection over Union (IoU) matrix between all pairs of boxes."""
    # If either set of boxes is empty, return a zero matrix
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    # Compute intersection coordinates
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    # Compute intersection width and height
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    # Compute area of each box set
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    # Compute union and IoU
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)

def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """Compute PASCAL Visual Object Classes (VOC)-style Average Precision."""
    # Extend recall and precision at both ends for interpolation
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Enforce a non increasing precision envelope from right to left
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Compute AP by summing area under the precision recall curve
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

""" DATASET """
class BrineShrimpDataset(Dataset):
    """Custom dataset for shrimp detection."""
    def __init__(self, img_to_boxes: Dict[str, List[List[float]]], augment: bool = True):
        readable, unreadable = [], []

        # Verify each labeled image is present and readable
        for p in img_to_boxes.keys():
            P = Path(p)
            if P.is_file() and is_image_ok(P):
                readable.append((p, img_to_boxes[p]))
            else:
                unreadable.append(P.name)

        # Store summary info about skipped labeled images
        self.unreadable_count = len(unreadable)
        self.unreadable_files = unreadable

        # Print summary of unreadable labeled images
        if unreadable:
            print(f"[labeled validator] Skipped {len(unreadable)} unreadable labeled images. First few: {unreadable[:5]}")

        # Store final list of valid samples and augmentation flag
        self.samples = readable
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Fetch a single item from the dataset and return image tensor, target dict, and image path."""
        img_path, boxes_list = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Convert bounding boxes to a tensor, or create an empty tensor if no boxes exist
        boxes = torch.tensor(boxes_list, dtype=torch.float32) if boxes_list else torch.zeros((0, 4))
        labels = torch.ones((boxes.size(0),), dtype=torch.int64)
        iscrowd = torch.zeros((boxes.size(0),), dtype=torch.int64)
        # Compute box areas with minimum clamp to avoid negative sizes
        area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)

        # Apply random horizontal flip augmentation
        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                boxes = hflip_boxes(boxes, img_w=w)
            # Apply random vertical flip augmentation
            if random.random() < 0.2:
                img = TF.vflip(img)
                boxes = vflip_boxes(boxes, img_h=h)

        # Convert image to torch tensor
        img_t = TF.to_tensor(img)
        
        # Construct target dictionary in torchvision detection format
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }
        return img_t, target, img_path

def collate_fn(batch):
    """Collate function to batch images and annotations."""
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)

""" MODEL """
def get_detector(num_classes: int = 2):
    """Load pretrained CNN."""
    # Load a pretrained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Extract the number of input features for the classifier head
    infl = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the predictor head with the desired number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(infl, num_classes)
    return model

""" TRAINING """
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Perform one training epoch and return mean loss."""
    model.train()
    running_loss = 0.0

    # Use tqdm for progress status
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Train {epoch:02d}")
    for i, (images, targets, _) in pbar:
        # Move images and targets to the appropriate device
        images = [img.to(device) for img in images]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Compute detection losses for batch
        loss_dict = model(images, tgts)
        losses = sum(loss for loss in loss_dict.values())

        # Standard backward pass and optimizer step
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Update running loss for progress display
        running_loss += losses.item()
        pbar.set_postfix(loss=f"{running_loss/(i+1):.4f}")

    # Return mean loss over the epoch
    return running_loss / max(1, len(data_loader))

""" VALIDATION """
@torch.no_grad()
def validate(model, data_loader, device, iou_thresh=0.5, score_thresh=0.05):
    """Evaluate model performance on validation set."""
    # Set model to evaluation mode
    model.eval()

    all_scores = []
    all_tp = []
    all_fp = []
    n_gt_total = 0

    # Iterate over validation dataset
    for images, targets, _ in tqdm(data_loader, desc="Validate"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        # Assume batch size of one for validation and extract outputs
        out = outputs[0]
        scores = out["scores"].cpu().numpy()
        boxes_p = out["boxes"].cpu().numpy()

        # Filter predictions below the score threshold
        keep = scores >= score_thresh
        scores = scores[keep]
        boxes_p = boxes_p[keep]

        # Ground truth boxes for the current image
        boxes_g = targets[0]["boxes"].cpu().numpy().astype(np.float32)
        n_gt_total += boxes_g.shape[0]

        # Compute IoUs between predicted and ground truth boxes
        ious = _iou_matrix(boxes_p.astype(np.float32), boxes_g)

        matched_g = set()
        # Sort predictions by score in descending order
        order = np.argsort(-scores)
        scores_sorted = scores[order]
        boxes_sorted = boxes_p[order]

        # Determine true positive and false positive assignments
        for j, box in enumerate(boxes_sorted):
            idx_match = np.argmax(ious[order[j]])
            iou = ious[order[j], idx_match]

            if iou >= iou_thresh and idx_match not in matched_g:
                all_tp.append(1)
                all_fp.append(0)
                matched_g.add(idx_match)
            else:
                all_tp.append(0)
                all_fp.append(1)

            all_scores.append(scores_sorted[j])

    # Handle edge case where no predictions were produced
    if len(all_scores) == 0:
        return {"precision": 0.0, "recall": 0.0, "AP50": 0.0}

    # Sort assignments by descending score and compute cumulative sums
    idx = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[idx]
    fp = np.array(all_fp)[idx]

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    # Compute precision and recall vectors
    prec = tp_cum / np.maximum(1, tp_cum + fp_cum)
    rec = tp_cum / np.maximum(1, n_gt_total)
    # Compute VOC style AP at IoU 0.5
    ap = _voc_ap(rec, prec)

    return {"precision": float(prec[-1]), "recall": float(rec[-1]), "AP50": float(ap)}

""" INFERENCE """
@torch.no_grad()
def infer_count_on_folder(model, folder: Path, device, conf_thresh=CONF_THRESH, save_viz=True, training_summary=None):
    """Run inference on unlabeled images and save results."""
    # Output CSV file for per image shrimp counts
    out_csv = RUN_DIR / f"{TODAY_STR}_brine_shrimp_counts.csv"
    conf_thresh = conf_thresh

    # Ensure model is in evaluation mode
    model.eval()
    # Collect and validate unlabeled images
    paths = list_unlabeled_images(folder)
    results = []
    skipped = []

    img_index = 0

    # Iterate through unlabeled images in a sorted order
    for p in tqdm(sorted(paths, key=lambda x: x.name.lower()), desc="Inference", unit="image"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            skipped.append(p.name)
            continue

        # Convert image to tensor and move to device
        inp = TF.to_tensor(img).unsqueeze(0).to(device)
        out = model(inp)[0]

        scores = out["scores"].cpu().numpy()
        keep = scores >= conf_thresh

        boxes = out["boxes"].cpu().numpy()[keep]
        scores = scores[keep]

        # Total number of detections above threshold
        count = int(keep.sum())
        rel_path = str(p.relative_to(ROOT))

        results.append((rel_path, count))

        # Draw bounding boxes and save visualization image
        if save_viz and count > 0:
            img_index += 1
            draw = ImageDraw.Draw(img)
            for (xmin, ymin, xmax, ymax), s in zip(boxes, scores):
                draw.rectangle([xmin, ymin, xmax, ymax], outline=(0,255,0), width=3)
                draw.text((xmin, ymin - 12), f"{s:.2f}", fill=(0,255,0))

            # Create output filename based on original filename
            original_stem = p.stem 
            out_name = f"{TODAY_STR}_segmented_{original_stem}.png"

            img.save(RUN_DIR / out_name)

    # Warn about unreadable unlabeled images
    if skipped:
        print(f"[infer] Skipped {len(skipped)} unreadable unlabeled images. First few: {skipped[:5]}")

    # Write per image counts to CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Path", "Count"])
        writer.writerows(results)

    # Compute total shrimp count across all unlabeled images
    total = sum([c for _, c in results])

    # Print detailed count summary to terminal
    print("\n==================== BRINE SHRIMP COUNTS ====================")
    for path, count in results:
        print(f"{path:<50} {count:>5}")
    print("------------------------------------------------------------")
    print(f"{'TOTAL SHRIMP DETECTED:':<50} {total:>5}")
    print("==============================================================\n")

    # Append aggregated information and training summary to the CSV file
    try:
        with open(out_csv, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write(f"Total Shrimp Detected,{total}\n")

            if training_summary:
                f.write(training_summary + "\n")

            # If a skip file exists for labeled images, append its content as well
            skip_file = RUN_DIR / f"{TODAY_STR}_skipped_labeled_info.txt"
            if skip_file.exists():
                with open(skip_file) as sf:
                    f.write("\n" + sf.read() + "\n")

    except Exception as e:
        print("[infer] Failed to append extra info:", e)

    return results

""" MAIN PIPELINE """
def main():
    # Decide whether to use GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run directory: {RUN_DIR}")

    # Load labeled data and split into training and validation sets
    """ LOAD TRAINING DATA """
    img_to_boxes = load_labelstudio_rects(PATH_JSON)
    
    # Filter dictionary to only include images that still exist and are readable
    readable = {p: b for p, b in img_to_boxes.items() if Path(p).is_file() and is_image_ok(Path(p))}
    assert len(readable) > 0, "No readable labeled images found."

    # Shuffle labeled image paths before splitting
    labeled_paths = list(readable.keys())
    random.shuffle(labeled_paths)

    # Compute number of training and validation images
    n_total = len(labeled_paths)
    n_val = max(1, int(math.ceil(n_total * VAL_SPLIT)))

    # Use the first subset for validation and the rest for training
    val_paths = set(labeled_paths[:n_val])
    train_paths = set(labeled_paths[n_val:])

    # Build dictionaries for training and validation sets
    train_dict = {p: readable[p] for p in train_paths}
    val_dict = {p: readable[p] for p in val_paths}

    # Create dataset objects for training and validation
    ds_train = BrineShrimpDataset(train_dict, augment=True)
    ds_val = BrineShrimpDataset(val_dict, augment=False)

    # Gather unreadable labeled image information from both subsets
    total_unreadable = ds_train.unreadable_count + ds_val.unreadable_count
    skip_file = RUN_DIR / f"{TODAY_STR}_skipped_labeled_info.txt"

    # Write a summary text file documenting unreadable labeled images
    with open(skip_file, "w") as f:
        f.write(f"Unreadable labeled images: {total_unreadable}\nFiles:\n")
        for name in ds_train.unreadable_files + ds_val.unreadable_files:
            f.write(name + "\n")

    print(f"[train] Total unreadable labeled images: {total_unreadable}")

    # Create training DataLoader with shuffling and specified batch size
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    # Create validation DataLoader with batch size of one and deterministic ordering
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # Initialize model and optimizer for training
    """ MODEL SETUP """
    model = get_detector(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    """ TRAIN LOOP """
    best_ap = -1.0
    training_summary = None

    # Iterate over the configured number of epochs
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train for one epoch and get average loss
        loss = train_one_epoch(model, optimizer, dl_train, device, epoch)
        # Validate and compute detection metrics
        metrics = validate(model, dl_val, device)

        # Compute error rate from AP50 for reporting
        error_rate = 1 - metrics["AP50"]

        # Construct a summary line for this epoch
        summary_line = (
            f"[epoch {epoch:02d}] Loss={loss:.4f}, "
            f"Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, "
            f"AP50={metrics['AP50']:.2f}, ErrorRate={error_rate:.2f}"
        )

        print(summary_line)
        training_summary = summary_line

        # Save checkpoint if current epoch achieves a new best AP50
        if metrics["AP50"] > best_ap:
            best_ap = metrics["AP50"]

            # Save a dated checkpoint inside the run directory
            dated_ckpt = RUN_DIR / f"{TODAY_STR}_bestmodel.pth"
            torch.save({"model": model.state_dict(), "epoch": epoch, "ap50": best_ap}, dated_ckpt)
            print(f"[checkpoint] Saved dated checkpoint to {dated_ckpt}")

            # Update global best model path as symlink or copy
            if BEST_MODEL_PATH.exists():
                BEST_MODEL_PATH.unlink()

            try:
                os.symlink(dated_ckpt, BEST_MODEL_PATH)
                print(f"[checkpoint] Updated symlink: {BEST_MODEL_PATH} -> {dated_ckpt}")
            except OSError:
                shutil.copy2(dated_ckpt, BEST_MODEL_PATH)
                print(f"[checkpoint] Symlink unsupported. Copied model to: {BEST_MODEL_PATH}")

    # Load the best model and run inference on unlabeled images
    """ LOAD BEST MODEL AND RUN INFERENCE """
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[load] Loaded best model from {BEST_MODEL_PATH} (AP50={ckpt['ap50']:.3f})")

    # If unlabeled images directory exists, run inference and counting
    if PATH_UNLABELED.exists():
        infer_count_on_folder(model, PATH_UNLABELED, device,
                                conf_thresh=CONF_THRESH,
                                save_viz=True,
                                training_summary=training_summary)
    else:
        print(f"[skip infer] No unlabeled_images directory found at {PATH_UNLABELED}")

if __name__ == "__main__":
    main()
