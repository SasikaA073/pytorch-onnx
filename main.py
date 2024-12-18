import os, json
os.environ['TORCH_HOME'] = './.cache' 

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from time import time

# Load the pre-trained EfficientNet model (e.g., efficientnet_b0)
model = models.efficientnet_b0(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the required input size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Normalization mean
        std=[0.229, 0.224, 0.225]    # Normalization std
    )
])

# Load and preprocess the input image
start_time = time()
image_path = "dog.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

end_time = time()

# Load ImageNet class labels from JSON file
with open("imagenet_1000_cls_id_to_label.json", "r") as f:
    imagenet_classes = json.load(f)

# Get top-5 predictions
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(f"{imagenet_classes[str(top5_catid[i].item())]}: {top5_prob[i].item():.4f}")

print(f"\nTime taken: {end_time - start_time:.4f} seconds")