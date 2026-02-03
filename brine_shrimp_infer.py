"""
Brine Shrimp Detector and Counter - CNN Deep Learning Model
Inference-only Pipeline for General Users
Dr. Erik Duboue's Lab
"""

""" IMPORTS """
# Import libraries
import csv
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch
from torchvision.transforms import functional as TF

# Import functions from full pipeline code
from brine_shrimp_train_and_infer import (
    ROOT,
    SAVE_ROOT,
    PATH_UNLABELED,
    BEST_MODEL_PATH,
    list_unlabeled_images,
    get_detector,
    IMG_SUFFIXES,
)

""" CONFIGURATION """
# Create dated output directory
TODAY_STR = datetime.now().strftime("%Y-%m-%d")
RUN_DIR = SAVE_ROOT / f"infer_{TODAY_STR}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Detection confidence threshold
CONF_THRESH = 0.85 

""" LOAD MODEL """
def load_trained_model(device):
    """Load the trained Faster R-CNN checkpoint and return the model."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {BEST_MODEL_PATH}. "
            "Ensure the lab has trained the model before running inference."
        )

    # Build model architecture
    model = get_detector(num_classes=2).to(device)

    # Load saved checkpoint
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded trained model from {BEST_MODEL_PATH}")
    return model

""" RUN INFERENCE """
@torch.no_grad()
def run_inference(model, device):
    """Run inference on unlabeled images and save detection outputs and counts."""
    # Ensure unlabeled directory exists
    if not PATH_UNLABELED.exists():
        print(f"No unlabeled_images directory found at {PATH_UNLABELED}")
        return

    # Gather image paths
    images = list_unlabeled_images(PATH_UNLABELED)
    if not images:
        print("No readable images found in unlabeled_images/")
        return

    # Output CSV path
    out_csv = RUN_DIR / f"{TODAY_STR}_brine_shrimp_counts.csv"
    results = []
    img_index = 0

    print(f"Running inference on {len(images)} images...")

    # Loop over images
    for p in tqdm(sorted(images, key=lambda x: x.name.lower()), desc="Inference", unit="image"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            print(f"Skipping unreadable image: {p.name}")
            continue

        # Convert image to tensor
        inp = TF.to_tensor(img).unsqueeze(0).to(device)

        # Run model
        output = model(inp)[0]

        # Filter detections by score
        scores = output['scores'].cpu().numpy()
        keep = scores >= CONF_THRESH

        # Extract accepted detections
        boxes = output['boxes'].cpu().numpy()[keep]
        scores = scores[keep]
        count = int(keep.sum())

        # Store results
        results.append((p.name, count))

        # Draw boxes if detections exist
        if count > 0:
            draw = ImageDraw.Draw(img)
            img_index += 1

            for (xmin, ymin, xmax, ymax), s in zip(boxes, scores):
                draw.rectangle([xmin, ymin, xmax, ymax], outline=(0,255,0), width=3)
                draw.text((xmin, ymin - 12), f"{s:.2f}", fill=(0,255,0))

            # Save visualization
            original_stem = p.stem
            out_name = f"{TODAY_STR}_segmented_{original_stem}.png"

            img.save(RUN_DIR / out_name)

    # Write CSV summary
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "Count"])
        writer.writerows(results)

    # Compute total count
    total = sum(c for _, c in results)

    # Console summary
    print("\n==================== BRINE SHRIMP COUNTS ====================")
    for name, cnt in results:
        print(f"{name:<40} {cnt:>5}")
    print("------------------------------------------------------------")
    print(f"{'TOTAL SHRIMP DETECTED:':<50} {total:>5}")
    print("============================================================\n")

    # Append metadata (matching the training script style)
    try:
        with open(out_csv, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write(f"Total Shrimp Detected,{total}\n")

            # Look for skipped labeled info file
            # Use latest dated run automatically if multiple exist
            skip_files = list(SAVE_ROOT.rglob("*_skipped_labeled_info.txt"))
            if skip_files:
                latest_skip_file = max(skip_files, key=lambda p: p.stat().st_mtime)
                with open(latest_skip_file, "r", encoding="utf-8") as sf:
                    f.write("\n=== Skipped Labeled Images Info ===\n")
                    f.write(sf.read() + "\n")

    except Exception as e:
        print("[infer] Failed to append extra info:", e)

""" MAIN """
if __name__ == "__main__":
    # Select appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Saving inference outputs to: {RUN_DIR}")

    # Load and run inference
    model = load_trained_model(device)
    run_inference(model, device)
