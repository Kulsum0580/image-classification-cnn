import tensorflow as tf
from tensorflow import keras
import torch
import numpy as np

# Load keras model
print("Loading Keras model...")
keras_model = keras.models.load_model("model/cnn_cifar10.keras")

# Get all weights
weights = []
for layer in keras_model.layers:
    w = layer.get_weights()
    if w:
        weights.append(w)
        print(f"Layer: {layer.name}, weights: {[x.shape for x in w]}")

print("Saving weights...")
import pickle
with open("model/keras_weights.pkl", "wb") as f:
    pickle.dump(weights, f)
print("Done! Weights saved.")