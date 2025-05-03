import os
import torch
import argparse
from PIL import Image
import glob
from torchvision import transforms
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt

# Import project modules
from food2recipe_model import Food2RecipeModel
from dataset import InferenceDataset

def visualize_results(image_path, recipe):
    """
    Visualize the image and generated recipe
    
    Args:
        image_path (str): Path to the food image
        recipe (dict): Generated recipe with title, ingredients, and instructions
    """
    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    # Show image
    img = Image.open(image_path)
    ax[0].imshow(img)
    ax[0].set_title("Food Image")
    ax[0].axis('off')
    
    # Show recipe
    recipe_text = f"Title: {recipe['title']}\n\n"
    recipe_text += "Ingredients:\n"
    for i, ingredient in enumerate(recipe['ingredients'], 1):
        recipe_text += f"{i}. {ingredient}\n"
    
    recipe_text += "\nInstructions:\n"
    for i, instruction in enumerate(recipe['instructions'], 1):
        recipe_text += f"{i}. {instruction}\n"
    
    ax[1].text(0, 0.5, recipe_text, va='center', fontsize=12)
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def inference(args):
    """
    Generate recipes from food images
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = Food2RecipeModel(
        encoder_embedding_dim=args.encoder_dim,
        gpt_model_name=args.gpt_model
    )
    
    # Load weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.model_path))
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    
    model = model.to(device)
    model.eval()
    
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get image paths
    if os.path.isdir(args.input_path):
        image_paths = glob.glob(os.path.join(args.input_path, "*.jpg")) + \
                      glob.glob(os.path.join(args.input_path, "*.jpeg")) + \
                      glob.glob(os.path.join(args.input_path, "*.png"))
    else:
        image_paths = [args.input_path]
    
    if not image_paths:
        print("No images found in the specified path.")
        return
    
    print(f"Found {len(image_paths)} images.")
    
    # Create dataset and dataloader
    dataset = InferenceDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Generate recipes
    all_recipes = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            paths = batch['path']
            
            # Generate recipes
            recipes = model.generate(
                images=images,
                max_length=args.max_length,
                num_beams=args.num_beams,
                early_stopping=True
            )
            
            # Pair images with generated recipes
            for path, recipe in zip(paths, recipes):
                result = {
                    'image_path': path,
                    'recipe': recipe
                }
                all_recipes.append(result)
                
                # Visualize if requested
                if args.visualize:
                    visualize_results(path, recipe)
    
    # Save results
    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump(all_recipes, f, indent=2)
        print(f"Results saved to {args.output_path}")
    
    # Print results
    for result in all_recipes:
        print(f"\nImage: {result['image_path']}")
        print(f"Recipe: {result['recipe']['title']}")
        print("Ingredients:")
        for ing in result['recipe']['ingredients']:
            print(f"  - {ing}")
        print("Instructions:")
        for i, step in enumerate(result['recipe']['instructions'], 1):
            print(f"  {i}. {step}")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate recipes from food images")
    
    # Input arguments
    parser.add_argument("--input_path", type=str, required=True, 
                        help="Path to food image or directory of images")
    parser.add_argument("--output_path", type=str, default="", 
                        help="Path to save generated recipes (JSON format)")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model")
    parser.add_argument("--encoder_dim", type=int, default=768, 
                        help="Dimension of image encoder output")
    parser.add_argument("--gpt_model", type=str, default="gpt2", 
                        help="GPT-2 model name")
    
    # Generation arguments
    parser.add_argument("--max_length", type=int, default=150, 
                        help="Maximum length of generated text")
    parser.add_argument("--num_beams", type=int, default=5, 
                        help="Number of beams for beam search")
    
    # Other arguments
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, 
                        help="Number of dataloader workers")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize results")
    
    args = parser.parse_args()
    
    inference(args)