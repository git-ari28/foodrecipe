import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import torch
from rouge import Rouge
import re

def calculate_bleu(reference, hypothesis):
    """
    Calculate BLEU score for text generation evaluation
    
    Args:
        reference (str): Reference text
        hypothesis (str): Generated text
        
    Returns:
        float: BLEU score
    """
    reference_tokens = reference.lower().split()
    hypothesis_tokens = hypothesis.lower().split()
    
    return sentence_bleu([reference_tokens], hypothesis_tokens)

def calculate_meteor(reference, hypothesis):
    """
    Calculate METEOR score for text generation evaluation
    
    Args:
        reference (str): Reference text
        hypothesis (str): Generated text
        
    Returns:
        float: METEOR score
    """
    reference_tokens = reference.lower().split()
    hypothesis_tokens = hypothesis.lower().split()
    
    return meteor_score([reference_tokens], hypothesis_tokens)

def calculate_rouge(reference, hypothesis):
    """
    Calculate ROUGE scores for text generation evaluation
    
    Args:
        reference (str): Reference text
        hypothesis (str): Generated text
        
    Returns:
        dict: Dictionary with ROUGE scores
    """
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)[0]
    
    return {
        'rouge-1': scores['rouge-1']['f'],
        'rouge-2': scores['rouge-2']['f'],
        'rouge-l': scores['rouge-l']['f']
    }

def calculate_ingredient_f1(reference_ingredients, predicted_ingredients):
    """
    Calculate F1 score for ingredients
    
    Args:
        reference_ingredients (list): Reference ingredients
        predicted_ingredients (list): Predicted ingredients
        
    Returns:
        float: F1 score
    """
    # Clean and standardize ingredients
    ref_clean = [clean_ingredient(ing) for ing in reference_ingredients]
    pred_clean = [clean_ingredient(ing) for ing in predicted_ingredients]
    
    # Calculate precision and recall
    true_positives = sum(1 for ing in pred_clean if ing in ref_clean)
    precision = true_positives / len(pred_clean) if pred_clean else 0
    recall = true_positives / len(ref_clean) if ref_clean else 0
    
    # Calculate F1
    if precision + recall == 0:
        return 0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def clean_ingredient(ingredient):
    """Standardize ingredient for better matching"""
    ing = ingredient.lower()
    ing = re.sub(r'\([^)]*\)', '', ing)  # Remove parentheses
    ing = re.sub(r'\d+', '', ing)  # Remove numbers
    ing = re.sub(r'cup|tablespoon|teaspoon|tbsp|tsp|oz|pound|lb', '', ing, flags=re.IGNORECASE)  # Remove measurements
    return ing.strip()

def evaluate_recipe_generation(reference_recipes, predicted_recipes):
    """
    Evaluate recipe generation using multiple metrics
    
    Args:
        reference_recipes (list): List of reference recipes (dicts with title, ingredients, instructions)
        predicted_recipes (list): List of predicted recipes (dicts with title, ingredients, instructions)
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    metrics = {
        'title_bleu': [],
        'ingredients_f1': [],
        'instructions_bleu': [],
        'instructions_rouge_1': [],
        'instructions_rouge_l': []
    }
    
    for ref, pred in zip(reference_recipes, predicted_recipes):
        # Title evaluation
        metrics['title_bleu'].append(calculate_bleu(ref['title'], pred['title']))
        
        # Ingredients evaluation
        metrics['ingredients_f1'].append(
            calculate_ingredient_f1(ref['ingredients'], pred['ingredients'])
        )
        
        # Instructions evaluation
        ref_instructions = ' '.join(ref['instructions'])
        pred_instructions = ' '.join(pred['instructions'])
        
        metrics['instructions_bleu'].append(calculate_bleu(ref_instructions, pred_instructions))
        
        rouge_scores = calculate_rouge(ref_instructions, pred_instructions)
        metrics['instructions_rouge_1'].append(rouge_scores['rouge-1'])
        metrics['instructions_rouge_l'].append(rouge_scores['rouge-l'])
    
    # Calculate average of all metrics
    return {k: np.mean(v) for k, v in metrics.items()}