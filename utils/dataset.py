import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer
import json
import os

class RecipeDataset(Dataset):
    def __init__(self, json_path, tokenizer_name='t5-small', image_transform=None):
        with open(json_path) as f:
            self.samples = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)

        # Tokenize text
        encoding = self.tokenizer(
            sample['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }