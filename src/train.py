"""
train.py
--------
Two-phase transfer learning training loop:
  Phase 1: frozen backbone, train classification head only (fast, stabilizes).
  Phase 2: unfreeze top ~35% of backbone, fine-tune end-to-end at low LR.

Usage:
    python src/train.py --backbone efficientnet --epochs1 10 --epochs2 15
    python src/train.py --backbone densenet
    python src/train.py --backbone resnet
"""

import argparse
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "TensorFlow is required to run this project. Install it first with `pip install tensorflow`."
    ) from exc

from preprocessing import get_data_generators, get_class_weights, verify_dataset
from model import (
    build_efficientnet,
    build_densenet,
    build_resnet,
    unfreeze_top_layers,
    compile_model,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

BACKBONE_MAP = {
    "efficientnet": (build_efficientnet, "efficientnetb0"),
    "densenet": (build_densenet, "densenet121"),
    "resnet": (build_resnet, "resnet50"),
}


def get_callbacks(model_name: str):
    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7),
        ModelCheckpoint(
            str(MODELS_DIR / f"{model_name}_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]


def main(backbone: str, epochs1: int, epochs2: int, batch_size: int):
    verify_dataset()
    train_gen, val_gen, test_gen = get_data_generators(batch_size=batch_size)
    class_weights = get_class_weights(train_gen)
    print("Class weights (balances rare tumor types):", class_weights)

    build_fn, layer_name = BACKBONE_MAP[backbone]
    model = build_fn()
    model = compile_model(model, lr=1e-3)

    print(f"\n=== PHASE 1: training classification head ({backbone}) ===")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs1,
        class_weight=class_weights,
        callbacks=get_callbacks(f"{backbone}_phase1"),
    )

    print(f"\n=== PHASE 2: fine-tuning top layers of {layer_name} ===")
    unfreeze_top_layers(model, layer_name, unfreeze_pct=0.35)
    model = compile_model(model, lr=1e-5)  # much lower LR for fine-tuning
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs2,
        class_weight=class_weights,
        callbacks=get_callbacks(f"{backbone}_final"),
    )

    final_path = MODELS_DIR / f"{backbone}_final.keras"
    model.save(final_path)
    print(f"\nSaved final model to {final_path}")

    print("\n=== Test set evaluation ===")
    results = model.evaluate(test_gen, return_dict=True)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=list(BACKBONE_MAP.keys()), default="efficientnet")
    parser.add_argument("--epochs1", type=int, default=10, help="frozen-head epochs")
    parser.add_argument("--epochs2", type=int, default=15, help="fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args.backbone, args.epochs1, args.epochs2, args.batch_size)
