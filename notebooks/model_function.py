import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import torch.nn.functional as F


## model is loaded in from the notebooks that we trained it in
## transfroms the input frame to the right format for the model
## Then feeds into prediction funciton to predict class and then show output
## used AI to help write these functionss
model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 5)
model.load_state_dict(torch.load("models/best_resnet18_emotes.pth", map_location="cpu"))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

idx_to_class = {0:"Cry", 1:"HandsUp", 2:"Still", 3:"TongueOut", 4:"Yawn"}

def predict_cr(frame):
    # Preprocessing so everything work s
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

   
    img = transform(img).unsqueeze(0)  # shape: (1,3,224,224)

    # Prediction part 
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
        probs = F.softmax(outputs, dim=1)   # still shape (1, 5)
        print(outputs)
        probs = F.softmax(outputs, dim=1)[0]
        readable = [round(p.item(), 4) for p in probs]
        print(readable)


    return pred.item()

#test 
test_predict = predict_cr(cv2.imread("data/frames/test/Still/Still25_f1.jpg"))
print(f"Test prediction: {test_predict}")  