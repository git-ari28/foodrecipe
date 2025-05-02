import pandas as pd
import os
from datasets import Dataset

def load_csv_to_hf_dataset(csv_path, image_dir):
    df = pd.read_csv(csv_path)

    data = []
    for _, row in df.iterrows():
        title = str(row["Title"]).strip()
        instructions = str(row["Instructions"]).strip()
        image_name = str(row["Image_Name"]).strip()
        image_path = os.path.join(image_dir, f"{image_name}.jpg")

        # Collect cleaned ingredients from columns starting at index 5
        cleaned_ingredients = []
        for val in row[5:]:
            if pd.notna(val):
                cleaned_ingredients.append(str(val).strip())
        ingredients_text = ', '.join(cleaned_ingredients)

        recipe_text = f"<recipe> <name>{title}</name> <ingredients>{ingredients_text}</ingredients> <instructions>{instructions}</instructions> </recipe>"

        data.append({
            "image": image_path,
            "text": recipe_text
        })

    return Dataset.from_list(data)

if __name__ == "__main__":
    csv_file = "data/recipes.csv"
    image_folder = "data/images"

    dataset = load_csv_to_hf_dataset(csv_file, image_folder)
    dataset.save_to_disk("data/processed_dataset")
