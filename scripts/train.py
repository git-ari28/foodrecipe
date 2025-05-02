import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torchvision import models, transforms
from utils.dataset import RecipeDataset

# Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 5

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset and DataLoader
dataset = RecipeDataset('../processed/train.json', image_transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Image Encoder (ResNet50)
resnet = models.resnet50(pretrained=True)
modules = list(resnet.children())[:-1]  # remove last FC layer
encoder = nn.Sequential(*modules).to(DEVICE)
for param in encoder.parameters():
    param.requires_grad = False

# Text Decoder (T5)
tokenizer = T5Tokenizer.from_pretrained('t5-small')
decoder = T5ForConditionalGeneration.from_pretrained('t5-small').to(DEVICE)

# Projection layer
decoder_embedding_size = decoder.config.d_model
projection = nn.Linear(2048, decoder_embedding_size).to(DEVICE)

# Optimizer
optimizer = torch.optim.AdamW(decoder.parameters(), lr=5e-5)

# Training Loop
for epoch in range(EPOCHS):
    decoder.train()
    total_loss = 0

    for batch in dataloader:
        images = batch['image'].to(DEVICE)
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)

        with torch.no_grad():
            features = encoder(images).squeeze()

        encoder_outputs = projection(features).unsqueeze(1)

        outputs = decoder(
            inputs_embeds=encoder_outputs,
            labels=input_ids,
            attention_mask=attention_mask
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

