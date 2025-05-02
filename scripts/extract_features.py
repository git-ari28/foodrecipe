import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np

# Paths
CSV_PATH = "../data/recipes.csv"  # Replace with your actual CSV name
IMAGE_FOLDER = "../data/images/"   # Folder containing the images
OUTPUT_FEATURES_FILE = "image_features.npy"
OUTPUT_IDS_FILE = "image_ids.npy"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained CNN (ResNet50)
model = resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier
model.eval().to(device)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet norms
                         std=[0.229, 0.224, 0.225])
])

# Load CSV
df = pd.read_csv(CSV_PATH)

# Containers
features = []
image_ids = []

# Extract features
for i, row in tqdm(df.iterrows(), total=len(df)):
    image_name = row['Image_Name']
    image_path = os.path.join(IMAGE_FOLDER, image_name + ".jpg")

    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model(img_tensor).squeeze().cpu().numpy()
        features.append(feature)
        image_ids.append(image_name)
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")

# Save features
np.save(OUTPUT_FEATURES_FILE, np.array(features))
np.save(OUTPUT_IDS_FILE, np.array(image_ids))
print(f"Saved {len(features)} image features to {OUTPUT_FEATURES_FILE}")
