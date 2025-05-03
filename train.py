import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import time
from tqdm import tqdm
from torchvision import transforms
from transformers import AdamW, get_linear_schedule_with_warmup

# Import project modules
from data_preprocessing import load_and_preprocess_data, prepare_train_val_test_split
from dataset import Food2RecipeDataset
from food2recipe_model import Food2RecipeModel

def train(args):
    """
    Train the Food2Recipe model
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(args.data_path)
    
    # Split data
    train_df, val_df, test_df = prepare_train_val_test_split(
        df, val_size=args.val_size, test_size=args.test_size, random_state=args.seed
    )
    
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Initialize model
    print("Initializing model...")
    model = Food2RecipeModel(
        encoder_embedding_dim=args.encoder_dim,
        gpt_model_name=args.gpt_model
    )
    
    # Move model to device
    model = model.to(device)
    
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = Food2RecipeDataset(
        train_df, args.img_dir, model.tokenizer, transform=transform, max_length=args.max_length
    )
    val_dataset = Food2RecipeDataset(
        val_df, args.img_dir, model.tokenizer, transform=transform, max_length=args.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    
    # Create directories for saving models
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            # Get batch data
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Prepare labels for language modeling (shift input_ids one position to the right)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Ignore the last token
            
            # Forward pass
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Update tracking variables
            train_loss += loss.item()
            train_steps += 1
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_steps
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                # Get batch data
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Prepare labels for language modeling
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = -100
                
                # Forward pass
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # Update tracking variables
                val_loss += loss.item()
                val_steps += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_steps
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Saving best model at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), os.path.join(args.save_dir, "food2recipe_best.pth"))
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(args.save_dir, f"food2recipe_checkpoint_{epoch+1}.pth"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, "food2recipe_final.pth"))
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Food2Recipe model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="data/recipes.csv", help="Path to the recipes CSV file")
    parser.add_argument("--img_dir", type=str, default="data/images", help="Directory containing food images")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set size")
    
    # Model arguments
    parser.add_argument("--encoder_dim", type=int, default=768, help="Dimension of image encoder output")
    parser.add_argument("--gpt_model", type=str, default="gpt2", help="GPT-2 model name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    # Saving arguments
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save models")
    parser.add_argument("--save_every", type=int, default=1, help="Save model every N epochs")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train(args)