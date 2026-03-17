import streamlit as st
import numpy as np
import cv2
import os
import gdown
import torch
import torch.nn as nn
from PIL import Image

MODEL_PATH  = "model/cnn_cifar10.pt"
GDRIVE_ID   = "1Ci-d-SCXqZogKY-ChW6F_7ICC_oINAen"
IMG_SIZE    = 32
MEAN        = [0.4914, 0.4822, 0.4465]
STD         = [0.2023, 0.1994, 0.2010]
CLASS_NAMES = [
    "✈️ Airplane", "🚗 Automobile", "🐦 Bird",  "🐱 Cat",  "🦌 Deer",
    "🐶 Dog",      "🐸 Frog",       "🐴 Horse", "🚢 Ship", "🚛 Truck"
]

st.set_page_config(page_title="CNN Image Classifier", page_icon="🔍", layout="centered")
st.title("🔍 Real-Time Image Classifier")
st.markdown("Upload an image and the **CNN model** will classify it into one of 10 categories.")
st.markdown("---")

# ── MODEL ARCHITECTURE ────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64),
            nn.MaxPool2d(2, 2), nn.Dropout(0.2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResBlock(128),
            nn.MaxPool2d(2, 2), nn.Dropout(0.2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            ResBlock(256),
            nn.MaxPool2d(2, 2), nn.Dropout(0.2),

            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            ResBlock(512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        with st.spinner("Downloading model... please wait 30 seconds"):
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
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ── PREPROCESS ────────────────────────────────────────────────────────────────
def preprocess(pil_image):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = (img - np.array(MEAN)) / np.array(STD)
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return img

# ── UI ────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file:
    pil_image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Your Image")
        st.image(pil_image, use_container_width=True)

    img_tensor = preprocess(pil_image)
    with st.spinner("Classifying..."):
        with torch.no_grad():
            outputs    = model(img_tensor)
            probs      = torch.softmax(outputs, dim=1)[0].numpy()
        top_idx    = int(np.argmax(probs))
        top_label  = CLASS_NAMES[top_idx]
        confidence = float(probs[top_idx])

    with col2:
        st.subheader("🎯 Prediction")
        if confidence >= 0.87:
            st.markdown(f"## {top_label}")
            st.markdown(f"**Confidence: `{confidence * 100:.1f}%`**")
            st.progress(confidence)
        else:
            st.markdown("## 🤷 Unknown")
            st.markdown("**This image doesn't match any of my 10 categories!**")
            st.markdown(f"*Highest match: {top_label} at `{confidence * 100:.1f}%`*")
    st.markdown("---")
    st.subheader("🔬 What the model sees (32×32)")
    small = cv2.resize(
        np.array(pil_image.convert("RGB")),
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )
    display = cv2.resize(small, (160, 160), interpolation=cv2.INTER_NEAREST)
    st.image(display, width=160)

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
st.caption("Built with PyTorch · OpenCV · Streamlit | Accuracy: 93.24%")

