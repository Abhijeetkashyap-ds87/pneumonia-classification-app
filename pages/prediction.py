import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Pneumonia Classification - Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Load Model Function ---
def load_model(dropout_rate=0.5):
    model = mobilenet_v3_small(weights=None)  # No pre-trained weights for inference
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 128),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(128, 2)
    )
    parameters = torch.load("models/mobilenet_v3_small_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(parameters)
    model.eval()
    return model

# --- Normalization Stats ---
global_mean = 122.75558188357674
global_std = 18.28005783697279

# --- Image Transform ---
transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([global_mean / 255] * 3, [global_std / 255] * 3)
])

# --- Predict Function ---
def predict(image):
    image = transformation(image).unsqueeze(0)
    model = load_model()
    with torch.inference_mode():
        output = model(image)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return predicted.item(), confidence.item() * 100

# --- Streamlit UI ---
st.title("Pneumonia Detection from Chest X-ray")

st.subheader("Upload a Chest X-ray Image")
uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    display_image = image.copy()
    display_image.thumbnail((400, 400))

    col1, col2 = st.columns([3, 2])

    with col1:
        st.image(display_image, caption="Uploaded X-ray", use_container_width=False)

    with col2:
        if st.button("Predict"):
            pred, prob = predict(image)
            label = "Pneumonia" if pred == 1 else "Normal"
            st.success(f"Prediction: **{label}**\nConfidence: **{prob:.2f}%**")
