# Duality-AI-s-Offroad-Semantic-Scene-Segmentation

# Offroad Semantic Scene Segmentation 🚗🌵

## 📌 Project Overview

This project focuses on semantic segmentation of offroad environments using a deep learning model. The goal is to classify each pixel of an image into terrain categories such as sky, trees, rocks, grass, bushes, and other obstacles.

The model is trained on a synthetic desert dataset provided by Duality AI and evaluated on unseen images to test generalization.

---

## 🧠 Model Used

* Model: SegFormer-B2
* Pretrained on ADE20K
* Fine-tuned on desert segmentation dataset

---

## 📊 Results

* Best mIoU Score: **0.6096**

### Per-class Performance (Highlights)

* Sky: Very high accuracy
* Trees: Good segmentation
* Dry Grass & Landscape: Moderate performance
* Logs & Ground Clutter: Lower accuracy due to small object size

---

## 📁 Project Structure

```
Offroad-Semantic-Segmentation/
│
├── train.py
├── test.py
├── best_segformer.pth
├── outputs/
├── training_results.png
├── README.md
├── report.pdf
```

---

## ⚙️ How to Run

### 1. Install Dependencies

```bash
pip install torch torchvision transformers opencv-python
```

### 2. Run Testing

```bash
python3 test.py
```

### 3. Output

* Segmented images will be saved in the `outputs/` folder

---

## 🖼️ Sample Outputs

The model generates color-coded segmentation maps highlighting different terrain regions such as sky, vegetation, and obstacles.

---

## ⚠️ Challenges Faced

* Class imbalance (few samples for logs and clutter)
* Difficulty in detecting small objects
* Similar appearance between terrain classes

---

## 🚀 Future Improvements

* Use higher resolution inputs
* Apply Dice Loss and class weighting
* Improve small object detection

---

## 📌 Notes

* Dataset is not included due to size constraints
* Please download it from the official Duality AI source

---

## 👨‍💻 Author

* Rohan Gangarde
* Parth Bagul
* Prathamesh Waman
