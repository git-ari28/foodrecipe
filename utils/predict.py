import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import T5Tokenizer
from models.cnn_encoder import CNNEncoder
from models.transformer_decoder import TransformerDecoder

def predict(image_path):
    # Add error handling for the image path
    if not os.path.exists(image_path):
        # Try adding extensions if missing
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = image_path + ext
            if os.path.exists(potential_path):
                image_path = potential_path
                print(f"Found image at {image_path}")
                break
        else:
            # If no extensions matched, raise an error
            raise FileNotFoundError(f"Could not find image at {image_path}")
    
    print(f"Loading image from: {image_path}")
    
    # Load models and tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    encoder = CNNEncoder()
    decoder = TransformerDecoder()
    
    # Load saved model weights if available
    try:
        encoder.load_state_dict(torch.load('models/encoder.pth'))
        print("Loaded encoder weights")
    except Exception as e:
        print(f"Could not load encoder weights: {e}")
    
    try:
        decoder.model.load_state_dict(torch.load('models/decoder.pth'))
        print("Loaded decoder weights")
    except Exception as e:
        print(f"Could not load decoder weights: {e}")
    
    encoder.eval()
    decoder.model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    encoder.to(device)
    decoder.model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Match the normalization from training
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            print("Extracting image features...")
            features = encoder(image).unsqueeze(1)  # [1, 1, embed_size]
            print(f"Feature shape: {features.shape}")
            
            print("Generating recipe...")
            output_text = decoder.generate(features, tokenizer)[0]
        
        return output_text
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise
