import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import torch.nn.functional as F

# ---- Load model ----
model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 4)
model.load_state_dict(torch.load("models/best_resnet18_emotes_no_tongue.pth", map_location="cpu"))
model.eval()

# ---- Transforms (must match training) ----
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

idx_to_class = {0:"Cry", 1:"HandsUp", 2:"Still", 3:"Yawn"}

def predict_cr(frame):
    # Convert BGR (OpenCV) → RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # Transform
    img = transform(img).unsqueeze(0)  # shape: (1,3,224,224)

    # Predict
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
        probs = F.softmax(outputs, dim=1)   # still shape (1, 4)
        print(outputs)
        probs = F.softmax(outputs, dim=1)[0]
        readable = [round(p.item(), 4) for p in probs]
        print(readable)


    return pred.item()

test_predict = predict_cr(cv2.imread("data/frames/test/TongueOut/TongueOut25_f1.jpg"))
print(f"Test prediction: {test_predict}")  