import os
import cv2
import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation

# =========================
# SETTINGS
# =========================
MODEL_PATH = "best_segformer.pth"
IMAGE_DIR = "test_images"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# =========================
# LOAD MODEL
# =========================
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=10,
    ignore_mismatched_sizes=True
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# =========================
# COLOR MAP FUNCTION
# =========================
def decode_segmap(mask):
    colors = np.array([
        [0, 0, 0],        # Background
        [128, 0, 0],      # Trees
        [0, 128, 0],      # Lush Bushes
        [128, 128, 0],    # Dry Grass
        [0, 0, 128],      # Dry Bushes
        [128, 0, 128],    # Ground Clutter
        [0, 128, 128],    # Flowers
        [128, 128, 128],  # Logs
        [64, 0, 0],       # Rocks
        [192, 0, 0]       # Sky
    ])

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(len(colors)):
        color_mask[mask == i] = colors[i]

    return color_mask

# =========================
# TEST LOOP
# =========================
for img_name in os.listdir(IMAGE_DIR):

    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image.copy()

    image = cv2.resize(image, (512, 512))
    image = image.astype("float32") / 255.0
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=image).logits

    preds = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()

    color_mask = decode_segmap(preds)
    color_mask = cv2.resize(color_mask, (original.shape[1], original.shape[0]))

    overlay = cv2.addWeighted(original, 0.6, color_mask, 0.4, 0)

    save_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Saved: {save_path}")

print("✅ Testing complete. Check 'outputs' folder.")
