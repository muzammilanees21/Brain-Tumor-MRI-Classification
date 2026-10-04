"""
api/main.py  -  FastAPI backend for the Brain Tumor MRI classifier.

Loads the model trained on Google Colab:
    models/brain_tumor_effnet.weights.h5
    models/class_names.json          (optional, defaults below)

New in this version:
    - SQLite prediction history (GET /history, DELETE /history)
    - DICOM (.dcm) file support, in addition to JPG/PNG
    - Batch prediction endpoint (POST /predict/batch, multiple files at once)

Run:  uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import base64
import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

layers = tf.keras.layers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "brain_tumor_effnet.weights.h5"
CLASSES_PATH = MODELS_DIR / "class_names.json"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "history.db"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
DEFAULT_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
LOW_CONFIDENCE = 0.85      # below this -> "needs human review"
MAX_COLOR_SPREAD = 12.0    # MRI scans are (almost) grayscale; photos are not
THUMB_SIZE = (96, 96)

DISCLAIMER = (
    "Educational/research use only. Not a medical device and not a substitute "
    "for diagnosis by a qualified radiologist or doctor."
)

app = FastAPI(
    title="Brain Tumor MRI Classification API",
    description="EfficientNetB0 classifier: glioma / meningioma / pituitary / no tumor. " + DISCLAIMER,
    version="2.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------- DB ---
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    filename TEXT,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    tumor_detected INTEGER NOT NULL,
    tumor_probability REAL NOT NULL,
    all_probabilities TEXT NOT NULL,
    thumbnail_b64 TEXT
)
"""


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(CREATE_TABLE_SQL)
    return conn


def log_prediction(filename: str, result: dict, thumb_b64: str) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO predictions (ts, filename, predicted_class, confidence, tumor_detected, "
        "tumor_probability, all_probabilities, thumbnail_b64) VALUES (?,?,?,?,?,?,?,?)",
        (
            datetime.now(timezone.utc).isoformat(),
            filename,
            result["predicted_class"],
            result["confidence"],
            int(result["tumor_detected"]),
            result["tumor_probability"],
            json.dumps(result["all_probabilities"]),
            thumb_b64,
        ),
    )
    conn.commit()
    conn.close()


# ----------------------------------------------------------------- model ---
def build_model(weights=None):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=(*IMG_SIZE, 3)
    )
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))  # expects raw 0-255 pixels
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(DEFAULT_CLASSES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="brain_tumor_effnet")


_state = {"model": None, "classes": DEFAULT_CLASSES}


def get_model():
    if _state["model"] is None:
        if not WEIGHTS_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Trained weights not found at {WEIGHTS_PATH}. Train the model in Google "
                    "Colab and copy brain_tumor_effnet.weights.h5 into the models/ folder."
                ),
            )
        model = build_model(weights=None)
        model.load_weights(str(WEIGHTS_PATH))
        _state["model"] = model
        if CLASSES_PATH.exists():
            _state["classes"] = json.loads(CLASSES_PATH.read_text())
    return _state["model"]


# --------------------------------------------------------------- helpers ---
def _read_dicom(data: bytes) -> np.ndarray:
    try:
        import pydicom
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="DICOM support needs the pydicom package: pip install pydicom",
        )
    try:
        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array.astype("float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read DICOM file: {e}")
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    arr = arr.astype("uint8")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr[..., 0]
        arr = np.stack([arr] * 3, axis=-1)
    return arr[..., :3]


def read_image(data: bytes, filename: str = ""):
    """Returns (1,224,224,3) float32 batch in 0-255 range, 224x224 uint8 RGB, color_spread."""
    is_dicom_ext = filename.lower().endswith((".dcm", ".dicom"))

    arr = None
    if is_dicom_ext:
        arr = _read_dicom(data)
    else:
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            arr = np.array(img)
        except (UnidentifiedImageError, OSError):
            # Not a standard image - maybe a DICOM file without the right extension
            arr = _read_dicom(data)

    if arr is None or min(arr.shape[:2]) < 32:
        raise HTTPException(status_code=400, detail="Image is too small or unreadable.")

    a = arr.astype(np.int16)
    color_spread = float(
        (np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])).mean() / 2
    )
    resized = tf.image.resize(arr, IMG_SIZE).numpy().astype("float32")
    display_img = np.clip(resized, 0, 255).astype("uint8")
    return resized[None], display_img, color_spread


def make_thumbnail_b64(display_img: np.ndarray) -> str:
    thumb = cv2.resize(display_img, THUMB_SIZE)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8")


def gradcam(model, batch, pred_index):
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    last_conv = [l for l in base.layers if isinstance(l, layers.Conv2D)][-1]
    grad_base = tf.keras.Model(base.input, [last_conv.output, base.output])
    head = model.layers[model.layers.index(base) + 1:]
    with tf.GradientTape() as tape:
        conv_out, feats = grad_base(batch)
        tape.watch(conv_out)
        x = feats
        for layer in head:
            x = layer(x)
        score = x[:, pred_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.squeeze(conv_out[0] @ weights[..., tf.newaxis])
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def overlay(rgb_uint8, cam, alpha=0.4):
    cam = cv2.resize(cam, (rgb_uint8.shape[1], rgb_uint8.shape[0]))
    color = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_uint8, 1 - alpha, color, alpha, 0)


def run_prediction(data: bytes, filename: str, with_gradcam: bool, log: bool = True) -> dict:
    model = get_model()
    classes = _state["classes"]
    batch, display_img, color_spread = read_image(data, filename)

    probs = model(batch, training=False).numpy()[0]
    idx = int(np.argmax(probs))
    pred = classes[idx]
    conf = float(probs[idx])
    looks_like_mri = color_spread <= MAX_COLOR_SPREAD
    p_notumor = float(probs[classes.index("notumor")]) if "notumor" in classes else 0.0

    out = {
        "predicted_class": pred,
        "confidence": conf,
        "tumor_detected": pred != "notumor",
        "tumor_probability": 1.0 - p_notumor,
        "all_probabilities": {c: float(p) for c, p in zip(classes, probs)},
        "looks_like_mri": looks_like_mri,
        "needs_review": bool(conf < LOW_CONFIDENCE or not looks_like_mri),
        "disclaimer": DISCLAIMER,
    }

    if log:
        try:
            log_prediction(filename, out, make_thumbnail_b64(display_img))
        except Exception:
            pass  # history logging must never break a prediction request

    if with_gradcam:
        cam = gradcam(model, tf.convert_to_tensor(batch), idx)
        blended = overlay(display_img, cam)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        out["gradcam_overlay_base64"] = base64.b64encode(buf).decode("utf-8")

    return out


# ------------------------------------------------------------- endpoints ---
@app.get("/health")
def health():
    return {
        "status": "ok",
        "weights_found": WEIGHTS_PATH.exists(),
        "model_loaded": _state["model"] is not None,
        "weights_path": str(WEIGHTS_PATH),
    }


@app.get("/classes")
def classes():
    return {"classes": _state["classes"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return run_prediction(await file.read(), file.filename or "upload", with_gradcam=False)


@app.post("/predict/gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    return run_prediction(await file.read(), file.filename or "upload", with_gradcam=True)


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...), include_gradcam: bool = False):
    results = []
    for f in files:
        data = await f.read()
        try:
            r = run_prediction(data, f.filename or "upload", with_gradcam=include_gradcam)
            r["filename"] = f.filename
            r["error"] = None
        except HTTPException as e:
            r = {"filename": f.filename, "error": e.detail}
        results.append(r)
    return {"results": results}


@app.get("/history")
def get_history(limit: int = 50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, ts, filename, predicted_class, confidence, tumor_detected, "
        "tumor_probability, thumbnail_b64 FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return {
        "history": [
            {
                "id": r[0],
                "timestamp": r[1],
                "filename": r[2],
                "predicted_class": r[3],
                "confidence": r[4],
                "tumor_detected": bool(r[5]),
                "tumor_probability": r[6],
                "thumbnail_base64": r[7],
            }
            for r in rows
        ]
    }


@app.delete("/history")
def clear_history():
    conn = get_conn()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return {"status": "cleared"}