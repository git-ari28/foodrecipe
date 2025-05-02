import pandas as pd
import json
import os

CSV_PATH = '../data/recipes.csv'
IMAGE_DIR = '../data/images/'
OUTPUT_PATH = '../processed/train.json'

# Load CSV
df = pd.read_csv(CSV_PATH)

# Clean and format samples
data = []
for i, row in df.iterrows():
    image_name = row['Image_Name'] + '.jpg'
    image_path = os.path.join(IMAGE_DIR, image_name)
    title = row['Title']
    ingredients = row['Cleaned_Ingredients']
    instructions = row['Instructions']

    text = f"<name>{title}</name> <ingredients>{ingredients}</ingredients> <instructions>{instructions}</instructions>"

    data.append({
        "image_path": image_path,
        "text": text
    })

# Save JSON
with open(OUTPUT_PATH, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Saved {len(data)} samples to {OUTPUT_PATH}")

