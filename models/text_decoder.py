import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class TextDecoder(nn.Module):
    """Transformer-based text decoder for recipe generation"""
    
    def __init__(self, gpt_model_name="gpt2", embedding_dim=768):
        """
        Args:
            gpt_model_name (str): Name of the pretrained GPT-2 model
            embedding_dim (int): Dimension of the input image embeddings
        """
        super(TextDecoder, self).__init__()
        
        # Load pretrained GPT-2 model
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        
        # Create projection layer from image embeddings to GPT embedding space
        self.projection = nn.Linear(embedding_dim, self.gpt.config.n_embd)
        
    def forward(self, image_embeddings, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass through the network
        
        Args:
            image_embeddings (torch.Tensor): Image embeddings from the encoder
            input_ids (torch.Tensor): Input token ids for the decoder
            attention_mask (torch.Tensor): Attention mask for the decoder
            labels (torch.Tensor): Labels for computing the language modeling loss
            
        Returns:
            torch.Tensor: Loss if labels are provided, else logits
        """
        # Project image embeddings to GPT embedding space
        image_proj = self.projection(image_embeddings)
        
        batch_size = image_proj.shape[0]
        
        # Get GPT embeddings for the input ids
        if input_ids is not None:
            # Add image embeddings as a prefix to the sequence
            inputs_embeds = self.gpt.transformer.wte(input_ids)
            
            # Insert image embeddings at the beginning
            # Create new tensors with space for the image embeddings
            extended_embeds = torch.zeros(
                (batch_size, inputs_embeds.shape[1] + 1, inputs_embeds.shape[2]),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device
            )
            
            # Place image embeddings at the beginning
            extended_embeds[:, 0, :] = image_proj
            
            # Place text embeddings after
            extended_embeds[:, 1:, :] = inputs_embeds
            
            # Adjust attention mask to include the image embedding token
            if attention_mask is not None:
                extended_attention_mask = torch.ones(
                    (batch_size, attention_mask.shape[1] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                extended_attention_mask[:, 1:] = attention_mask
            else:
                extended_attention_mask = None
            
            # Run through GPT-2
            outputs = self.gpt(
                inputs_embeds=extended_embeds,
                attention_mask=extended_attention_mask,
                labels=labels
            )
        else:
            # For generation, we'll start with just the image embedding
            outputs = self.gpt(
                inputs_embeds=image_proj.unsqueeze(1)
            )
        
        return outputs