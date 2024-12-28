import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Define the custom ClassToken layer for Vision Transformer (ViT)
class ClassToken(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]
        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

# Load models
vit_model = load_model(
    r'Vit_Model.h5',
    custom_objects={'ClassToken': ClassToken}
)
cnn_model = load_model(r'CNN_Model.h5')

# Define class names
class_names = [
    "Barred_Spiral_Galaxies",
    "Cigar_Shaped_Smooth_Galaxies",
    "Disturbed_Galaxies",
    "Edge_On_Galaxies_With_Bulge",
    "Edge_On_Galaxies_Without_Bulge",
    "In_Between_Round_Smooth_Galaxies",
    "Merging_Galaxies",
    "Round_Smooth_Galaxies",
    "Unbarred_Loose_Spiral_Galaxies",
    "Unbarred_Tight_Spiral_Galaxies"
]

# Preprocessing function for CNN
def preprocess_for_cnn(img):
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Preprocessing function for ViT
def preprocess_for_vit(img):
    img = cv2.resize(img, (200, 200))  # Resize to 200x200
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    patch_size = 25
    patches = []
    for i in range(0, 200, patch_size):
        for j in range(0, 200, patch_size):
            patch = img[0, i:i + patch_size, j:j + patch_size, :]
            patches.append(patch.flatten())
    patches = np.array(patches)
    return np.expand_dims(patches, axis=0)

# Prediction function
def predict(image, model_choice):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if model_choice == "CNN":
        preprocessed_img = preprocess_for_cnn(img)
        model = cnn_model
    elif model_choice == "ViT":
        preprocessed_img = preprocess_for_vit(img)
        model = vit_model
    else:
        return "Invalid model choice!"
    
    predictions = model.predict(preprocessed_img)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence_score = predictions[0][predicted_class_idx] * 100  # Confidence as a percentage
    
    predicted_class = class_names[predicted_class_idx]
    return f"Predicted Class: {predicted_class}\nConfidence Score: {confidence_score:.2f}%"

# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload a Galaxy Image"),
        gr.Radio(["CNN", "ViT"], label="Select Model")
    ],
    outputs=gr.Textbox(label="Prediction and Confidence Score"),
    title="Galaxy Type Classification",
    description="Upload an image of a galaxy and select a model to classify its type. The prediction includes the galaxy type and the confidence score."
)

# Launch the interface
interface.launch()
