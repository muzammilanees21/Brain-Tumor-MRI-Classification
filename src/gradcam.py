"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping) for explainability.

This is the single most important "advanced" component of a medical
imaging project: it shows WHICH region of the MRI the model focused on
to make its prediction, overlaid as a heatmap. Without this, the model
is a black box that no clinician (or recruiter) would trust.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (2017)
"""

from typing import Optional

import cv2
import numpy as np
import tensorflow as tf


def find_last_conv_layer(model: tf.keras.Model, backbone_layer_name: str) -> str:
    """Auto-detect the last convolutional layer inside the backbone.

    Conv2D layers always produce 4D output for image inputs, so a plain
    isinstance check is enough — newer Keras versions removed the
    `layer.output_shape` attribute this used to check as well.
    """
    backbone = model.get_layer(backbone_layer_name)
    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Could not find a Conv2D layer in the backbone.")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    backbone_layer_name: str,
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """
    img_array: preprocessed (1, H, W, 3) float array, already normalized.
    Returns a normalized (0-1) heatmap the same spatial size as the conv layer.

    Keras 3 will not let you build a new Model that reaches *inside* an
    already-nested submodel's output while also depending on the outer
    model's input (raises "Output with path ... is not connected to
    inputs"). The fix: build a standalone Model directly from the
    backbone's own input/output (no nesting issue there), then manually
    replay the remaining head layers (GAP, Dense, etc.) on top of it
    inside the same GradientTape.
    """
    backbone = model.get_layer(backbone_layer_name)
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model, backbone_layer_name)
    conv_layer = backbone.get_layer(last_conv_layer_name)

    backbone_grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[conv_layer.output, backbone.output],
    )

    backbone_idx = model.layers.index(backbone)
    head_layers = model.layers[backbone_idx + 1:]

    with tf.GradientTape() as tape:
        conv_outputs, backbone_outputs = backbone_grad_model(img_array)
        tape.watch(conv_outputs)
        x = backbone_outputs
        for layer in head_layers:
            x = layer(x)
        predictions = x
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    original_img: (H, W, 3) uint8 RGB image (0-255)
    heatmap: raw output from make_gradcam_heatmap (0-1, any size)
    Returns: (H, W, 3) uint8 RGB blended image
    """
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlaid = cv2.addWeighted(original_img.astype("uint8"), 1 - alpha, heatmap_color, alpha, 0)
    return overlaid