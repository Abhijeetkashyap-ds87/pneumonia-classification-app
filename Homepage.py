import streamlit as st
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Pneumonia Classification - Training Metrics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Pneumonia Detection - Model Training Overview")

# --- Model Info ---
st.markdown("#### 📌 Model Used: `MobileNetV3-Small`")
st.markdown("""
This model was chosen for its lightweight architecture and suitability for deployment on resource-constrained environments like mobile or edge devices. The final classifier layer was modified to output binary classes (Normal vs Pneumonia).
""")

# --- Training Summary ---
st.markdown("### 📈 Training & Validation Performance")
try:
    img = Image.open('assets/Screenshot 2025-08-05 at 1.24.31 PM.png')
    img.thumbnail((1000, 1000))  # Resize
    st.image(img, caption="Loss & Accuracy during Training", use_container_width=False)
except FileNotFoundError:
    st.warning('⚠️ Training summary image not found.')

# --- Confusion Matrix ---
st.markdown("### 🧮 Test Set Evaluation - Confusion Matrix")
try:
    c_img = Image.open('assets/Screenshot 2025-08-05 at 1.25.05 PM.png')
    c_img.thumbnail((800, 800))  # Resize
    st.image(c_img, caption="Confusion Matrix on Test Set", use_container_width=False)
except FileNotFoundError:
    st.warning('⚠️ Confusion matrix image not found.')

# --- Summary Footer ---
st.markdown("---")
st.markdown(" This project is part of a pneumonia detection system built using PyTorch, Torchvision, and Streamlit.")
