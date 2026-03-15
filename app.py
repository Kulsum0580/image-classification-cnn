import streamlit as st
import numpy as np
import cv2
import os
import gdown
import torch
import torch.nn as nn
from PIL import Image

MODEL_PATH  = "model/cnn_cifar10.pt"
GDRIVE_ID   = "1CiIkBhejneT2UguScocsMZqBXYhpPFTD"
IMG_SIZE    = 32
CLASS_NAMES = [
    "✈️ Airplane", "🚗 Automobile", "🐦 Bird",  "🐱 Cat",  "🦌 Deer",
    "🐶 Dog",      "🐸 Frog",       "🐴 Horse", "🚢 Ship", "🚛 Truck"
]

st.set_page_config(page_title="CNN Image Classifier", page_icon="🔍", layout="centered")
st.title("🔍 Real-Time Image Classifier")
st.markdown("Upload an image and the **CNN model** will classify it into one of 10 categories.")
st.markdown("---")

# CNN architecture must match training
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        st.info("Downloading model... please wait")
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_ID}",
            MODEL_PATH, quiet=False
        )
    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

try:
    model = load_model()
    st.success("✅ Model loaded!")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

def preprocess(pil_image):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    return img

uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp","webp"])

if uploaded_file:
    pil_image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Your Image")
        st.image(pil_image, use_container_width=True)

    img_tensor = preprocess(pil_image)
    with st.spinner("Classifying..."):
        with torch.no_grad():
            outputs = model(img_tensor)
            probs   = torch.softmax(outputs, dim=1)[0].numpy()
        top_idx    = int(np.argmax(probs))
        top_label  = CLASS_NAMES[top_idx]
        confidence = float(probs[top_idx])

    with col2:
        st.subheader("🎯 Prediction")
        st.markdown(f"## {top_label}")
        st.markdown(f"**Confidence: `{confidence * 100:.1f}%`**")
        st.progress(confidence)

    st.markdown("---")
    st.subheader("📊 All Class Probabilities")
    for label, prob in zip(CLASS_NAMES, probs):
        col_a, col_b, col_c = st.columns([2, 4, 1])
        with col_a:
            st.write(label)
        with col_b:
            st.progress(float(prob))
        with col_c:
            st.write(f"{prob*100:.1f}%")

else:
    st.info("👆 Upload an image to get started!")
    st.markdown("""
    **10 categories the model recognises:**

    | | | | | |
    |---|---|---|---|---|
    | ✈️ Airplane | 🚗 Automobile | 🐦 Bird | 🐱 Cat | 🦌 Deer |
    | 🐶 Dog | 🐸 Frog | 🐴 Horse | 🚢 Ship | 🚛 Truck |
    """)

st.markdown("---")
st.caption("Built with PyTorch · OpenCV · Streamlit")
