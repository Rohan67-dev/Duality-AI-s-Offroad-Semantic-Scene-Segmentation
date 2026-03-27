import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATASET_ROOT = os.path.expanduser("~/Downloads/Offroad_Segmentation_Training_Dataset")
TRAIN_IMG_DIR  = os.path.join(DATASET_ROOT, "train", "Color_Images")
TRAIN_MASK_DIR = os.path.join(DATASET_ROOT, "train", "Segmentation")
VAL_IMG_DIR    = os.path.join(DATASET_ROOT, "val", "Color_Images")
VAL_MASK_DIR   = os.path.join(DATASET_ROOT, "val", "Segmentation")

IMAGE_SIZE    = (512, 512)
BATCH_SIZE    = 2
GRAD_ACCUM    = 2        # reduce to 2 if you get out-of-memory error
NUM_EPOCHS    = 30
LEARNING_RATE = 6e-5
NUM_CLASSES   = 10
SAVE_PATH     = "best_segformer.pth"

# Class ID → index mapping
CLASS_IDS  = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
ID_TO_IDX  = {cid: i for i, cid in enumerate(CLASS_IDS)}
CLASS_NAMES = ["Trees","Lush Bushes","Dry Grass","Dry Bushes",
               "Ground Clutter","Flowers","Logs","Rocks","Landscape","Sky"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────
class OffRoadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.processor = processor
        self.images    = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        img_path  = os.path.join(self.img_dir, img_name)

        # Try matching mask with same name
        mask_path = os.path.join(self.mask_dir, img_name)
        if not os.path.exists(mask_path):
            # Try .png extension
            base      = os.path.splitext(img_name)[0]
            mask_path = os.path.join(self.mask_dir, base + ".png")

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).resize(IMAGE_SIZE, Image.NEAREST)
        mask  = np.array(mask, dtype=np.int64)

        # Remap class IDs to 0–9
        new_mask = np.zeros_like(mask)
        for cid, idx_val in ID_TO_IDX.items():
            new_mask[mask == cid] = idx_val

        # Process image with SegFormer processor
        encoded = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"height": IMAGE_SIZE[0], "width": IMAGE_SIZE[1]}
        )
        pixel_values = encoded["pixel_values"].squeeze(0)
        label        = torch.tensor(new_mask, dtype=torch.long)
        return pixel_values, label

# ─────────────────────────────────────────
# IoU METRIC
# ─────────────────────────────────────────
def compute_iou(preds, labels, num_classes=NUM_CLASSES):
    iou_per_class = []
    preds  = preds.view(-1)
    labels = labels.view(-1)
    for cls in range(num_classes):
        pred_c  = (preds  == cls)
        true_c  = (labels == cls)
        inter   = (pred_c & true_c).sum().float()
        union   = (pred_c | true_c).sum().float()
        if union == 0:
            continue
        iou_per_class.append((inter / union).item())
    return np.mean(iou_per_class) if iou_per_class else 0.0

def compute_per_class_iou(preds, labels, num_classes=NUM_CLASSES):
    result = {}
    preds  = preds.view(-1)
    labels = labels.view(-1)
    for cls in range(num_classes):
        pred_c = (preds  == cls)
        true_c = (labels == cls)
        inter  = (pred_c & true_c).sum().float()
        union  = (pred_c | true_c).sum().float()
        result[CLASS_NAMES[cls]] = (inter / union).item() if union > 0 else None
    return result

# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
def train():
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        do_reduce_labels=False
    )

    train_ds = OffRoadDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, processor)
    val_ds   = OffRoadDataset(VAL_IMG_DIR,   VAL_MASK_DIR,   processor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"📦 Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Load SegFormer-B2 pretrained on ADE20K, replace head for 10 classes
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        id2label={i: n for i, n in enumerate(CLASS_NAMES)},
        label2id={n: i for i, n in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load("best_segformer.pth"))
    print("✅ Loaded previous best model")
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_iou        = 0.0
    train_losses    = []
    val_losses      = []
    val_ious        = []

    for epoch in range(NUM_EPOCHS):
        # ── Train
        model.train()
        total_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(pixel_values=images, labels=masks)
            
            # SegFormer returns upsampled logits
            logits = outputs.logits  # (B, num_classes, H/4, W/4)
            upsampled = nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            loss = criterion(upsampled, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ── Validate
        model.eval()
        val_loss = 0
        val_iou  = 0
        all_per_class = {n: [] for n in CLASS_NAMES}

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images = images.to(DEVICE)
                masks  = masks.to(DEVICE)

                outputs = model(pixel_values=images)
                logits  = outputs.logits
                upsampled = nn.functional.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                loss      = criterion(upsampled, masks)
                val_loss += loss.item()

                preds    = upsampled.argmax(dim=1)
                val_iou += compute_iou(preds.cpu(), masks.cpu())

                pc = compute_per_class_iou(preds.cpu(), masks.cpu())
                for name, v in pc.items():
                    if v is not None:
                        all_per_class[name].append(v)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou  = val_iou  / len(val_loader)
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        scheduler.step()

        print(f"\nEpoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_iou:.4f}")
        print("  Per-class IoU:")
        for name in CLASS_NAMES:
            vals = all_per_class[name]
            avg  = np.mean(vals) if vals else 0.0
            print(f"    {name:<18}: {avg:.4f}")

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✅ Best model saved! mIoU = {best_iou:.4f}")

    # ── Save plots
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color="red")
    plt.plot(val_losses,   label="Val Loss",   color="blue")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Loss Curve"); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_ious, label="Val mIoU", color="green")
    plt.xlabel("Epoch"); plt.ylabel("mIoU")
    plt.title("Validation IoU"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    print("\n📊 Training graphs saved → training_results.png")
    print(f"🏆 Best mIoU achieved: {best_iou:.4f}")

if __name__ == "__main__":
    train()
