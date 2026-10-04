"""
model.py
--------
Transfer-learning model architectures for Brain Tumor MRI classification.

Three backbones are provided (EfficientNetB0, DenseNet121, ResNet50) plus
a function to build a soft-voting ensemble of all three — this is what
pushes accuracy from the "single model ~93-95%" range to 96%+ on the
Kaggle Brain Tumor MRI dataset.
"""

from typing import Tuple

try:
    import tensorflow as tf  # type: ignore[import-not-found]
    from tensorflow.keras import layers, models  # type: ignore[import-not-found]
    from tensorflow.keras.applications import (  # type: ignore[import-not-found]
        EfficientNetB0,
        DenseNet121,
        ResNet50,
    )
except ImportError:
    import keras
    from keras import layers, models
    from keras.applications import EfficientNetB0, DenseNet121, ResNet50

    tf = keras

NUM_CLASSES = 4
IMG_SHAPE = (224, 224, 3)


def _build_head(base_model, num_classes: int = NUM_CLASSES) -> models.Model:
    base_model.trainable = False  # Phase 1: frozen backbone

    inputs = layers.Input(shape=IMG_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)


def build_efficientnet(num_classes: int = NUM_CLASSES) -> models.Model:
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=IMG_SHAPE)
    return _build_head(base, num_classes)


def build_densenet(num_classes: int = NUM_CLASSES) -> models.Model:
    base = DenseNet121(include_top=False, weights="imagenet", input_shape=IMG_SHAPE)
    return _build_head(base, num_classes)


def build_resnet(num_classes: int = NUM_CLASSES) -> models.Model:
    base = ResNet50(include_top=False, weights="imagenet", input_shape=IMG_SHAPE)
    return _build_head(base, num_classes)


def unfreeze_top_layers(model: models.Model, base_model_name: str, unfreeze_pct: float = 0.35) -> None:
    """
    Phase 2 fine-tuning: unfreeze the top N% of the backbone and let it
    adapt to MRI-specific textures instead of only ImageNet features.
    Call this AFTER phase-1 head-only training, then recompile with a
    low learning rate (1e-5) before continuing training.
    """
    base_model = model.get_layer(base_model_name)
    base_model.trainable = True
    n_layers = len(base_model.layers)
    freeze_until = int(n_layers * (1 - unfreeze_pct))
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    print(f"Unfroze last {n_layers - freeze_until}/{n_layers} layers of {base_model_name}")


def compile_model(model: models.Model, lr: float = 1e-3) -> models.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


class EnsembleModel:
    """
    Soft-voting ensemble: averages the softmax probabilities of all three
    fine-tuned backbones. Used at inference time only (each model is
    trained independently via train.py).
    """

    def __init__(self, models_list, weights=None):
        self.models = models_list
        self.weights = weights or [1.0 / len(models_list)] * len(models_list)

    def predict(self, x):
        preds = [w * m.predict(x, verbose=0) for m, w in zip(self.models, self.weights)]
        return sum(preds)
