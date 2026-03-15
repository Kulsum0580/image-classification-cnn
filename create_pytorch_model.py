import torch
import torch.nn as nn
import numpy as np
import pickle

# Load saved keras weights
with open("model/keras_weights.pkl", "rb") as f:
    weights = pickle.load(f)

print(f"Total weight groups: {len(weights)}")
for i, w in enumerate(weights):
    print(f"Group {i}: {[x.shape for x in w]}")

# Build PyTorch model
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
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CNN()

# Transfer weights from Keras to PyTorch
# Keras conv weights: (H, W, in, out) → PyTorch: (out, in, H, W)
def transfer_conv(pt_layer, keras_w):
    w = torch.tensor(keras_w[0]).permute(3, 2, 0, 1).float()
    b = torch.tensor(keras_w[1]).float()
    pt_layer.weight.data = w
    pt_layer.bias.data   = b

def transfer_bn(pt_layer, keras_w):
    # keras BN: gamma, beta, mean, var
    pt_layer.weight.data  = torch.tensor(keras_w[0]).float()  # gamma
    pt_layer.bias.data    = torch.tensor(keras_w[1]).float()  # beta
    pt_layer.running_mean = torch.tensor(keras_w[2]).float()  # mean
    pt_layer.running_var  = torch.tensor(keras_w[3]).float()  # var

def transfer_linear(pt_layer, keras_w):
    pt_layer.weight.data = torch.tensor(keras_w[0]).T.float()
    pt_layer.bias.data   = torch.tensor(keras_w[1]).float()

# Map weights in order
transfer_conv(model.features[0],  weights[0])   # conv2d
transfer_bn(model.features[1],    weights[1])   # bn
transfer_conv(model.features[3],  weights[2])   # conv2d_1
transfer_bn(model.features[4],    weights[3])   # bn_1
transfer_conv(model.features[8],  weights[4])   # conv2d_2
transfer_bn(model.features[9],    weights[5])   # bn_2
transfer_conv(model.features[11], weights[6])   # conv2d_3
transfer_bn(model.features[12],   weights[7])   # bn_3
transfer_conv(model.features[16], weights[8])   # conv2d_4
transfer_bn(model.features[17],   weights[9])   # bn_4
transfer_conv(model.features[19], weights[10])  # conv2d_5
transfer_bn(model.features[20],   weights[11])  # bn_5
transfer_linear(model.classifier[1], weights[12])  # dense
transfer_bn(model.classifier[2],     weights[13])  # bn_6
transfer_linear(model.classifier[5], weights[14])  # dense_1

# Save PyTorch model
torch.save(model.state_dict(), "model/cnn_cifar10.pt")
print("✅ PyTorch model saved to model/cnn_cifar10.pt")

# Quick test
model.eval()
dummy = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    out = model(dummy)
print(f"✅ Test output shape: {out.shape}")
print("✅ Conversion successful!")