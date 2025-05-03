import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class Food2RecipeDataset(Dataset):
    """Dataset for the Food2Recipe model"""
    
    def __init__(self, dataframe, img_dir, tokenizer, transform=None, max_length=512):
        """
        Args:
            dataframe (pandas.DataFrame): DataFrame with recipe data
            img_dir (string): Directory with all the images
            tokenizer: Tokenizer for text processing
            transform (callable, optional): Optional transform to be applied on an image
            max_length (int): Maximum length of tokenized text
        """
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image
        img_name = self.dataframe.iloc[idx]['Image_Name']
        if not img_name or pd.isna(img_name):
            # Use default image if no image name specified
            img_path = os.path.join(self.img_dir, "default_food.jpg")
            if not os.path.exists(img_path):
                # Create a placeholder image
                img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            else:
                img = Image.open(img_path).convert('RGB')
        else:
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                # Create a placeholder image
                img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            else:
                img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Get text
        text = self.dataframe.iloc[idx]['Formatted_Text']
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {
            'image': img,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text': text,
            'title': self.dataframe.iloc[idx]['Title'],
            'ingredients': self.dataframe.iloc[idx]['Ingredients'],
            'instructions': self.dataframe.iloc[idx]['Instructions']
        }

class InferenceDataset(Dataset):
    """Dataset for inference with only images"""
    
    def __init__(self, image_paths, transform=None):
        """
        Args:
            image_paths (list): List of image paths
            transform (callable, optional): Optional transform to be applied on an image
        """
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return {
            'image': img,
            'path': img_path
        }