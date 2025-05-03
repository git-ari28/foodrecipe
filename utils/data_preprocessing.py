import os
import shutil
import pandas as pd
from PIL import Image
import json
import argparse

def preprocess_data(image_dir, recipes_file, output_dir):
    """
    Preprocess the food images and recipes data to create the processed dataset.
    
    Args:
        image_dir (str): Directory containing food images
        recipes_file (str): Path to recipes CSV file
        output_dir (str): Output directory for processed data
    """
    print(f"Starting preprocessing...")
    print(f"Image directory: {image_dir}")
    print(f"Recipes file: {recipes_file}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load recipes data
    try:
        recipes_df = pd.read_csv(recipes_file)
        print(f"Loaded {len(recipes_df)} recipes from CSV")
    except Exception as e:
        print(f"Error loading recipes file: {e}")
        return
    
    # Get list of image files
    try:
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(image_files)} image files")
    except Exception as e:
        print(f"Error reading image directory: {e}")
        return
    
    # Create image_data.json for processed images
    image_data = {}
    processed_count = 0
    
    for img_file in image_files:
        try:
            # Get image ID (filename without extension)
            img_id = os.path.splitext(img_file)[0]
            
            # Find corresponding recipe (if exists)
            recipe_row = None
            if 'image_id' in recipes_df.columns:
                recipe_row = recipes_df[recipes_df['image_id'] == img_id]
            
            # If no image_id column or no matching row, try using filename or id column
            if recipe_row is None or len(recipe_row) == 0:
                for col in ['filename', 'id', 'image', 'image_name']:
                    if col in recipes_df.columns:
                        recipe_row = recipes_df[recipes_df[col] == img_id]
                        if len(recipe_row) > 0:
                            break
            
            # Process image
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path)
            
            # Resize image (optional)
            # img = img.resize((224, 224))
            
            # Save processed image
            processed_img_path = os.path.join(output_dir, img_file)
            img.save(processed_img_path)
            
            # Gather image metadata
            width, height = img.size
            image_data[img_id] = {
                'filename': img_file,
                'width': width,
                'height': height,
                'has_recipe': recipe_row is not None and len(recipe_row) > 0
            }
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} images")
                
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
    
    # Save image metadata
    with open(os.path.join(output_dir, 'image_data.json'), 'w') as f:
        json.dump(image_data, f, indent=2)
    
    # Process recipes
    recipe_data = {}
    for _, row in recipes_df.iterrows():
        try:
            # Determine image ID field
            img_id = None
            for col in ['image_id', 'filename', 'id', 'image', 'image_name']:
                if col in row.index and not pd.isna(row[col]):
                    img_id = str(row[col])
                    # Remove extension if present
                    img_id = os.path.splitext(img_id)[0]
                    break
            
            if img_id is None:
                continue
                
            # Extract recipe information
            recipe = {}
            for col in recipes_df.columns:
                if col != img_id and not pd.isna(row[col]):
                    recipe[col] = row[col]
                    
            recipe_data[img_id] = recipe
        except Exception as e:
            print(f"Error processing recipe: {e}")
    
    # Save recipe data
    with open(os.path.join(output_dir, 'recipe_data.json'), 'w') as f:
        json.dump(recipe_data, f, indent=2)
    
    # Create train/val/test splits
    all_ids = list(image_data.keys())
    import random
    random.shuffle(all_ids)
    
    # 70% train, 15% validation, 15% test
    train_size = int(0.7 * len(all_ids))
    val_size = int(0.15 * len(all_ids))
    
    train_ids = all_ids[:train_size]
    val_ids = all_ids[train_size:train_size+val_size]
    test_ids = all_ids[train_size+val_size:]
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    with open(os.path.join(output_dir, 'splits.json'), 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Preprocessing complete!")
    print(f"Processed {len(image_data)} images")
    print(f"Processed {len(recipe_data)} recipes")
    print(f"Data splits: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test")
    print(f"Output saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess food images and recipes')
    parser.add_argument('--image_dir', type=str, default='data/images',
                        help='Directory containing food images')
    parser.add_argument('--recipes_file', type=str, default='data/recipes.csv',
                        help='Path to recipes CSV file')
    parser.add_argument('--output_dir', type=str, default='data/processed_dataset',
                        help='Output directory for processed data')
    
    args = parser.parse_args()
    
    preprocess_data(args.image_dir, args.recipes_file, args.output_dir)