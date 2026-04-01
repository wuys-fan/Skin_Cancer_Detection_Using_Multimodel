<div align="center">

# 🩺 Multimodal Skin Cancer Detection
**Enhancing Dermatological Diagnosis with Deep Learning and Patient Metadata Fusion**

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-ISIC%202019%20%7C%20HAM10000-green.svg)](https://challenge2019.isic-archive.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📌 Project Overview
This project presents an advanced deep learning pipeline for **Skin Cancer Detection**, classifying dermatoscopic images into 8 distinct categories (`MEL`, `NV`, `BCC`, `AK`, `BKL`, `DF`, `VASC`, `SCC`). 

Unlike standard image-only classification models, this research implements a **Multimodal Learning Approach** that fuses Visual Features (extracted via ConvNeXt) with Patient Clinical Metadata (Age, Sex, and Anatomical Site) to significantly boost diagnostic accuracy and reduce false negatives in critical cases.

### 🌟 Technical Highlights
- **State-of-the-Art Architecture:** Leverages **ConvNeXt** as the primary backbone, taking advantage of modern CNN design principles for robust feature extraction.
- **Multimodal Fusion Engine:** Supports two dedicated training modes:
  - `Image-Only`: Standard baseline model.
  - `Concat Fusion`: Late-fusion concatenation of visual embeddings with a bespoke 13-dimensional one-hot encoded metadata vector.
- **Progressive Training Strategy:** Implements a 2-stage training process: `Warm-up` (Classifier training with frozen backbone) followed by `Fine-tuning` (Unfrozen full model) to prevent catastrophic forgetting.
- **Robust Software Engineering:** Completely modularized pipeline built with PyTorch emphasizing clear Separation of Concerns (`config`, `data`, `modeling`, `trainer`, `evaluation`).

---

## 🗂️ Project Architecture & Structure

The codebase is engineered strictly following Python & PyTorch best practices:

```text
├── convnext_pipeline/
│   ├── config.py       # Centralized hyperparameters & directory paths
│   ├── data.py         # ISIC dataset definitions, DataLoaders, and Augmentations
│   ├── modeling.py     # Custom ConvNeXt, Fusion, and Cross-Attention architectures
│   ├── trainer.py      # Core Training/Validation loops and Early Stopping logic
│   ├── evaluation.py   # Metrics calculations and Confusion Matrix generation
│   └── main.py         # Elegant entry point orchestrating the entire lifecycle
└── README.md
```

## 📊 Dataset & Preprocessing
The model is trained on the combined **ISIC 2019** and **HAM10000** datasets. 
### Metadata Encoding Strategy
Clinical text data represents a massive challenge due to missing values. We mitigate this using a **13-Dimensional Missing Value Indicator Vector**:
- **Age**: Normalized (`[Age/90.0, 0.0]`) or Missing Indicator (`[0.0, 1.0]`).
- **Sex**: One-hot encoded Male/Female with an explicit Missing channel.
- **Anatomical Site**: 7 encoded locations (Torso, Extremities, Head/Neck, etc.) + 1 Missing channel.

---

## 🚀 Getting Started

### 1. Requirements
Ensure you have PyTorch installed with CUDA support.
```bash
pip install torch torchvision pandas numpy Pillow matplotlib
```

### 2. Configuration
Update the dataset paths in `convnext_pipeline/config.py` to point to your local ISIC / HAM10000 directories before execution.

### 3. Execution (Training & Evaluation)
The pipeline is fully automated. You can switch between different fusion strategies simply by changing the `--mode` argument dynamically:

```bash
# Baseline: Image Only
python convnext_pipeline/main.py --mode image

# Multimodal: Image + Metadata (Concatenation)
python convnext_pipeline/main.py --mode fusion
```

*Note: The script automatically handles checkpoint loading, warm-up epochs, learning rate scheduling (ReduceLROnPlateau), and metrics plotting.*

---

## 📈 Results & Performance

Below are the empirical evaluation metrics derived from the Test Set (2537 samples):

| 🏗️ Model Architecture | 🧩 Fusion Strategy | ⚖️ Balanced Acc. | 🎯 Overall Acc. | 🔬 Macro Precision | 🏆 Macro F1 | 🥇 Weighted F1 |
|:----------------------|:-------------------|:----------------:|:---------------:|:------------------:|:-----------:|:--------------:|
| **ConvNeXt Baseline** | None               | 79.95%           | 87.78%          | 83.83%             | 81.71%      | 87.60%         |
| **ConvNeXt Fusion**   | Concatenation      | **82.63%**       | **89.20%**      | **84.55%**         | **83.36%**  | **89.07%**     |

**💡 Key Insight from the Fusion Strategy:**
Incorporating patient metadata (Age, Sex, Anatomical Site) yielded massive improvements across the board, particularly for underrepresented classes. The most remarkable leap is in **Balanced Accuracy** (Macro-Averaged Recall), which surged from **79.95% to 82.63%**. 

Because dermatological datasets suffer from severe class imbalance, the Baseline model inherently favors the majority class. However, the `Concat Fusion` strategy mitigated this bias, successfully boosting the **Macro F1-Score from 81.71% to 83.36%**. This proves that patient metadata acts as crucial differential diagnostic context, allowing the ConvNeXt model to accurately identify rare skin lesions without sacrificing its overall high precision.


## 🔮 Future Roadmap
- [ ] Export trained weights to **ONNX / TensorRT** for high-speed inference.
- [ ] Implement robust **Explainable AI (Grad-CAM)** visualizations to pinpoint lesion ROIs (Regions of Interest) for dermatologists.
- [ ] Package the inference engine into a lightweight **FastAPI Microservice**.

---
*Created for demonstrating strong Machine Learning Engineering principles, from robust model design to highly maintainable codebases.*
