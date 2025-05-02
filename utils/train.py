import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from tqdm import tqdm

# Import your dataset class
from utils.dataset import FoodDataset

def train():
    # Hyperparameters
    batch_size = 8
    max_epochs = 5
    learning_rate = 5e-5
    max_length = 512
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model.to(device)
    
    # Paths
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Navigate up from utils folder
    csv_file = os.path.join(data_dir, 'data', 'recipes.csv')
    img_dir = os.path.join(data_dir, 'data', 'images')
    
    print(f"CSV path: {csv_file}")
    print(f"Images directory: {img_dir}")
    
    # Check if files/directories exist
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
    
    # Set up dataset with error handling
    try:
        # First check what's in the CSV
        df = pd.read_csv(csv_file)
        print(f"CSV columns: {df.columns.tolist()}")
        print(f"First few image names: {df['Image_Name'].head(3).tolist()}")
        
        # Create dataset with just valid images
        dataset = FoodDataset(csv_file=csv_file, img_dir=img_dir, tokenizer=tokenizer, max_length=max_length)
        
        # Check if we have any valid images
        if len(dataset) == 0:
            print("⚠️ No valid images found! Check your data paths and CSV contents.")
            return
            
        print(f"Dataset created with {len(dataset)} valid samples")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    except Exception as e:
        print(f"Error setting up dataset: {e}")
        return
    
    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Set model to training mode
    model.train()
    
    # Training loop
    for epoch in range(max_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}", unit="batch")
        
        for batch in progress_bar:
            try:
                images, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, _ = batch
                
                # Move tensors to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                decoder_input_ids = decoder_input_ids.to(device)
                decoder_attention_mask = decoder_attention_mask.to(device)
                
                # Forward pass through the model
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask
                )
                
                # Check if loss is None
                if outputs.loss is None:
                    print(f"Warning: Loss is None at batch {progress_bar.n}")
                    continue  # Skip this batch
                
                # Get the loss and perform backpropagation
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Accumulate loss for reporting
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue  # Skip to next batch
        
        # Print loss at the end of the epoch
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader)}")
    
    # Save the model after training
    os.makedirs('saved_model', exist_ok=True)
    model.save_pretrained('saved_model')
    tokenizer.save_pretrained('saved_model')
    print("Model saved to 'saved_model' directory")

if __name__ == "__main__":
    train()