"""
preprocessing.py
-----------------
Data loading, cleaning, and augmentation pipeline for the Brain Tumor MRI
Classification project.

Dataset expected (Kaggle "Brain Tumor MRI Dataset" by Masoud Nickparvar):
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Expected folder structure after download:
    data/raw/
        Training/
            glioma/
            meningioma/
            notumor/
            pituitary/
        Testing/
            glioma/
            meningioma/
            notumor/
            pituitary/
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

try:
    import tensorflow as tf
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore[reportMissingImports]
    except ImportError:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore[reportMissingImports]
except ImportError:
    tf = None
    try:
        from keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        ImageDataGenerator = None

        def _missing_dependency_error():
            raise ImportError(
                "TensorFlow/Keras is required to run preprocessing. "
                "Install it with: pip install tensorflow"
            )

        def get_data_generators(*args, **kwargs):
            _missing_dependency_error()

        def get_class_weights(*args, **kwargs):
            _missing_dependency_error()

        def preprocess_single_image(*args, **kwargs):
            _missing_dependency_error()
    else:
        tf = None

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TRAIN_DIR = RAW_DIR / "Training"
TEST_DIR = RAW_DIR / "Testing"


def verify_dataset() -> None:
    """Sanity-check that the dataset was downloaded and placed correctly."""
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DIR}.\n"
            "Download it from Kaggle:\n"
            "  kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset\n"
            "then unzip it so you have data/raw/Training/<class>/*.jpg and "
            "data/raw/Testing/<class>/*.jpg"
        )
    for split_dir in (TRAIN_DIR, TEST_DIR):
        for cls in CLASS_NAMES:
            n = len(list((split_dir / cls).glob("*")))
            print(f"{split_dir.name}/{cls}: {n} images")


def get_data_generators(
    img_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    val_split: float = 0.15,
):
    """
    Build train/val/test ImageDataGenerators.

    NOTE: horizontal flip is intentionally OFF. Human anatomy (and tumor
    location relative to left/right hemisphere) is not symmetric in a way
    that's safe to augment naively for diagnostic imaging.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.10,
        brightness_range=[0.9, 1.1],
        shear_range=0.05,
        horizontal_flip=False,
        fill_mode="nearest",
        validation_split=val_split,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="validation",
        shuffle=False,
        seed=42,
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def get_class_weights(train_gen) -> dict:
    """
    Medical datasets are rarely balanced. Compute class weights so the loss
    function penalizes mistakes on minority classes more heavily instead of
    the model just learning to predict the majority class.
    """
    labels = train_gen.classes
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    return dict(enumerate(weights))


def preprocess_single_image(image_bytes: bytes, img_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Used by the FastAPI inference endpoint to prep an uploaded image."""
    import io
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


if __name__ == "__main__":
    verify_dataset()
