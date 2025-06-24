import tensorflow as tf
import numpy as np
import cv2

# For MobileNetV2, 'out_relu' is the last convolutional activation layer
LAST_CONV_LAYER_NAME = "out_relu"

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """Generates a Grad-CAM heatmap for a given image and model."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(LAST_CONV_LAYER_NAME).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-9) # Add epsilon for stability
    return heatmap.numpy()

def get_gradcam_overlay(image_bytes, heatmap, img_size=(224, 224)):
    """Superimposes the heatmap on the original image."""
    # Decode image from bytes using OpenCV
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, img_size)

    # Resize and apply colormap to the heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Create the superimposed image
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Encode the final image back to bytes (PNG format)
    _, img_encoded = cv2.imencode('.png', superimposed_img)
    return img_encoded.tobytes()