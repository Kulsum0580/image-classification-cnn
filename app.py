import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image

MODEL_PATH  = "model/cnn_cifar10.keras"
IMG_SIZE    = 32
CLASS_NAMES = [
    "✈️ Airplane", "🚗 Automobile", "🐦 Bird",  "🐱 Cat",  "🦌 Deer",
    "🐶 Dog",      "🐸 Frog",       "🐴 Horse", "🚢 Ship", "🚛 Truck"
]

st.set_page_config(page_title="CNN Image Classifier", page_icon="🔍", layout="centered")
st.title("🔍 Real-Time Image Classifier")
st.markdown("Upload an image and the **CNN model** will classify it into one of 10 categories.")
st.markdown("---")

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    st.success("✅ Model loaded!")
except Exception as e:
    st.error("❌ Model not found. Please run train.py first.")
    st.stop()

def preprocess(pil_image):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file:
    pil_image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Your Image")
        st.image(pil_image, use_column_width=True)

    img_array = preprocess(pil_image)
    with st.spinner("Classifying..."):
        probs      = model.predict(img_array, verbose=0)[0]
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

    st.markdown("---")
    st.subheader("🔬 What the model sees (32×32)")
    small = cv2.resize(np.array(pil_image.convert("RGB")),
                       (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
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
st.caption("Built with TensorFlow · Keras · OpenCV · Streamlit")


