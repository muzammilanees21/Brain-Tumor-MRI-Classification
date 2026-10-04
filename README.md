# 🧠 Brain Tumor MRI Classification — Advanced Deep Learning Project

An end-to-end deep learning system that classifies brain MRI scans into **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**, using transfer learning, model explainability (Grad-CAM), and a full production-style deployment (FastAPI + Streamlit).

> ⚠️ **Disclaimer:** This is an educational/portfolio project. It is **not** a certified medical device and must never be used for real diagnostic decisions. Always consult a qualified radiologist.

---

## 1. Problem Statement

Radiologists manually reviewing brain MRI scans face high caseloads and diagnostic fatigue, which can lead to delayed or missed tumor detection. This project builds an AI-assisted screening tool that classifies an MRI scan into one of 4 categories and — critically — **explains its reasoning** via a heatmap, so it can act as a second-opinion aid rather than a black box.

## 2. Dataset

**[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** (Kaggle, Masoud Nickparvar) — a combination of Figshare, SARTAJ, and Br35H datasets, ~7,000 MRI images across 4 classes.

Download it and place it like this:

```
data/raw/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

```bash
# Option A — Kaggle CLI
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data/raw --unzip

# Option B — kagglehub (Python)
python -c "import kagglehub; print(kagglehub.dataset_download('masoudnickparvar/brain-tumor-mri-dataset'))"
```

## 3. Project Structure

```
brain-tumor-classifier/
├── notebooks/
│   └── 01_brain_tumor_eda_experimentation.ipynb   # EDA + model experimentation
├── src/
│   ├── preprocessing.py     # data loading, augmentation, class weights
│   ├── model.py              # EfficientNet/DenseNet/ResNet architectures + ensemble
│   ├── train.py               # 2-phase transfer learning training loop
│   ├── gradcam.py             # Grad-CAM explainability
│   └── evaluate.py            # confusion matrix, sensitivity/specificity, ROC-AUC
├── api/
│   └── main.py                # FastAPI inference backend
├── app/
│   └── streamlit_app.py       # Streamlit frontend UI
├── models/                     # saved .keras model weights (generated after training)
├── reports/                    # generated evaluation plots/reports
├── data/raw/                   # dataset goes here (not committed to git)
├── requirements.txt
├── Dockerfile
└── README.md
```

## 4. Methodology

### 4.1 Preprocessing
- Resize to 224×224, normalize to [0,1]
- Augmentation: rotation, zoom, shift, brightness, shear — **no horizontal flip** (anatomically asymmetric data shouldn't be mirrored)
- Class weighting to handle imbalance between tumor/no-tumor classes

### 4.2 Modeling — Two-Phase Transfer Learning
| Phase | What happens |
|---|---|
| **Phase 1** | ImageNet-pretrained backbone frozen; only the new classification head trains (fast, stabilizes weights) |
| **Phase 2** | Top ~35% of backbone unfrozen; fine-tuned end-to-end at a low learning rate (1e-5) to adapt to MRI-specific textures |

Three backbones are implemented and compared: **EfficientNetB0**, **DenseNet121**, **ResNet50**. A **soft-voting ensemble** of all three is available for the highest accuracy configuration.

### 4.3 Explainability
**Grad-CAM** heatmaps show which region of the scan drove each prediction — implemented in `src/gradcam.py` and exposed live via the `/predict/gradcam` API endpoint and the Streamlit UI.

### 4.4 Evaluation
Beyond raw accuracy, this project reports (see `src/evaluate.py`):
- Per-class precision / recall / F1
- **Sensitivity & specificity per class** (recall matters more than accuracy here — a missed tumor is far costlier than a false alarm)
- Confusion matrix
- ROC-AUC curves (one-vs-rest)

## 5. How to Run

### 5.1 Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 5.2 Train the model
```bash
cd src
python train.py --backbone efficientnet --epochs1 10 --epochs2 15
```
This saves `models/efficientnet_final.keras`, used automatically by the API.

Train the other backbones for the ensemble (optional):
```bash
python train.py --backbone densenet
python train.py --backbone resnet
```

### 5.3 Explore in the notebook
```bash
jupyter notebook notebooks/01_brain_tumor_eda_experimentation.ipynb
```
Covers: EDA, augmentation preview, baseline-CNN-vs-transfer-learning comparison, backbone comparison, fine-tuning, evaluation, Grad-CAM, and ensembling — all with explanations of *why* each step matters.

### 5.4 Run the backend (FastAPI)
```bash
uvicorn api.main:app --reload --port 8000
```
- Docs: http://localhost:8000/docs
- `GET /health` — service status
- `GET /classes` — class labels
- `POST /predict` — upload an image, get class + confidence
- `POST /predict/gradcam` — same, plus base64 Grad-CAM overlay

### 5.5 Run the frontend (Streamlit)
```bash
streamlit run app/streamlit_app.py
```
Upload an MRI scan → get prediction, confidence, class probabilities, and Grad-CAM heatmap.

### 5.6 Docker (backend only)
```bash
docker build -t brain-tumor-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models brain-tumor-api
```

## 6. Results (fill in after training on your machine)

| Model | Test Accuracy | Macro F1 | Notes |
|---|---|---|---|
| Baseline CNN (scratch) | ~82-88% | — | overfits quickly, included for comparison |
| ResNet50 | — | — | |
| DenseNet121 | — | — | |
| EfficientNetB0 | — | — | typically best single model |
| **Ensemble (all 3)** | — | — | highest accuracy, 3x inference cost |

Run `src/evaluate.py` after training to auto-generate the confusion matrix, ROC curves, and sensitivity/specificity report into `reports/`.

## 7. Real-World Considerations & Limitations

- **Not clinically validated.** Trained on a single public dataset; real deployment needs multi-institution data, radiologist-verified labels, and regulatory clearance (FDA/CE).
- **Confidence thresholding matters.** The Streamlit app flags predictions below 70% confidence for mandatory human review instead of blindly trusting the model — this is standard practice in clinical-support ML.
- **Recall > accuracy for tumor classes.** A missed tumor (false negative) is far more dangerous than a false alarm, which is why sensitivity/specificity are reported per class, not just overall accuracy.
- **Dataset bias.** Public MRI datasets often come from limited scanner types/populations; performance may not generalize to different hospitals' equipment.

## 8. Possible Extensions
- Deploy to Hugging Face Spaces / Streamlit Community Cloud for a public demo link
- Add tumor **segmentation** (not just classification) using U-Net for localization/size estimation
- Add Monte Carlo Dropout for uncertainty estimation alongside Grad-CAM
- Export to ONNX/TF-Lite for faster edge inference
