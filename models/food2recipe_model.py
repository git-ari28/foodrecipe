import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
import re

from image_encoder import ImageEncoder
from text_decoder import TextDecoder

class Food2RecipeModel(nn.Module):
    """End-to-end model for food image to recipe generation"""
    
    def __init__(self, encoder_embedding_dim=768, gpt_model_name="gpt2"):
        """
        Args:
            encoder_embedding_dim (int): Dimension of the image encoder output
            gpt_model_name (str): Name of the pretrained GPT-2 model
        """
        super(Food2RecipeModel, self).__init__()
        
        # Image encoder
        self.image_encoder = ImageEncoder(embedding_dim=encoder_embedding_dim)
        
        # Text decoder
        self.text_decoder = TextDecoder(gpt_model_name=gpt_model_name, embedding_dim=encoder_embedding_dim)
        
        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
        
        # Add special tokens for recipe structure
        special_tokens = {
            'additional_special_tokens': [
                'Title:', 'Ingredients:', 'Instructions:'
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.text_decoder.gpt.resize_token_embeddings(len(self.tokenizer))
        
    def forward(self, images, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass through the network
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, height, width)
            input_ids (torch.Tensor): Input token ids for the text decoder
            attention_mask (torch.Tensor): Attention mask for the text decoder
            labels (torch.Tensor): Labels for computing the language modeling loss
            
        Returns:
            torch.Tensor: Loss if labels are provided, else logits
        """
        # Encode images
        image_embeddings = self.image_encoder(images)
        
        # Generate text with the decoder
        outputs = self.text_decoder(
            image_embeddings=image_embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def generate(self, images, max_length=150, num_beams=5, early_stopping=True):
        """
        Generate recipe text from food images
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, height, width)
            max_length (int): Maximum length of generated text
            num_beams (int): Number of beams for beam search
            early_stopping (bool): Whether to stop when all beams are finished
            
        Returns:
            list: List of generated recipe texts
        """
        # Encode images
        image_embeddings = self.image_encoder(images)
        
        # Create a prompt to help guide generation
        prompt = "Title:"
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(images.device)
        
        # Project image embeddings to GPT embedding space
        image_proj = self.text_decoder.projection(image_embeddings)
        
        batch_size = image_proj.shape[0]
        
        # Prepare input embeddings with image embeddings at the beginning
        inputs_embeds = self.text_decoder.gpt.transformer.wte(prompt_ids)
        inputs_embeds = inputs_embeds.repeat(batch_size, 1, 1)
        
        # Create extended embeddings with image projection
        extended_embeds = torch.zeros(
            (batch_size, inputs_embeds.shape[1] + 1, inputs_embeds.shape[2]),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device
        )
        
        # Place image embeddings at the beginning
        extended_embeds[:, 0, :] = image_proj
        
        # Place prompt embeddings after
        extended_embeds[:, 1:, :] = inputs_embeds
        
        # Generate with beam search
        outputs = self.text_decoder.gpt.generate(
            inputs_embeds=extended_embeds,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode generated token ids to text
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # Parse the generated recipes
        parsed_recipes = []
        for text in generated_texts:
            recipe = self.parse_generated_recipe(text)
            parsed_recipes.append(recipe)
        
        return parsed_recipes
    
    def parse_generated_recipe(self, text):
        """
        Parse the generated recipe text into title, ingredients, and instructions
        
        Args:
            text (str): Generated recipe text
            
        Returns:
            dict: Dictionary with parsed recipe components
        """
        # Extract title
        title_match = re.search(r'Title:(.*?)(?=Ingredients:|$)', text, re.DOTALL)
        title = title_match.group(1).strip() if title_match else "Unknown Recipe"
        
        # Extract ingredients
        ingredients_match = re.search(r'Ingredients:(.*?)(?=Instructions:|$)', text, re.DOTALL)
        ingredients_text = ingredients_match.group(1).strip() if ingredients_match else ""
        ingredients = [item.strip() for item in ingredients_text.split(',') if item.strip()]
        
        # Extract instructions
        instructions_match = re.search(r'Instructions:(.*?)$', text, re.DOTALL)
        instructions_text = instructions_match.group(1).strip() if instructions_match else ""
        instructions = [step.strip() for step in instructions_text.split('.') if step.strip()]
        
        return {
            'title': title,
            'ingredients': ingredients,
            'instructions': instructions
        }