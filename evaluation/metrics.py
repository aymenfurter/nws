import torch
import math
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from collections import Counter

def perplexity(loss):
    return math.exp(loss)

def accuracy(logits, targets):
    predictions = logits.argmax(dim=-1)
    return (predictions == targets).float().mean().item()

def bleu_score(hypotheses, references):
    """
    Calculate BLEU score for a corpus of hypotheses and references.
    
    :param hypotheses: List of generated texts
    :param references: List of lists of reference texts
    :return: BLEU score
    """
    # Tokenize the hypotheses and references
    hypotheses = [hypothesis.split() for hypothesis in hypotheses]
    references = [[reference.split() for reference in ref] for ref in references]
    
    return corpus_bleu(references, hypotheses)

def rouge_score(hypotheses, references):
    """
    Calculate ROUGE scores for a corpus of hypotheses and references.
    
    :param hypotheses: List of generated texts
    :param references: List of reference texts
    :return: Dictionary of ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    
    # Aggregate scores
    aggregated_scores = {
        'rouge1': sum(score['rouge1'].fmeasure for score in scores) / len(scores),
        'rouge2': sum(score['rouge2'].fmeasure for score in scores) / len(scores),
        'rougeL': sum(score['rougeL'].fmeasure for score in scores) / len(scores),
    }
    
    return aggregated_scores

def diversity_metrics(generated_texts, n_gram=1):
    """
    Calculate diversity metrics for generated texts.
    
    :param generated_texts: List of generated texts
    :param n_gram: n-gram size for distinct-n calculation
    :return: Dictionary of diversity metrics
    """
    total_words = 0
    unique_n_grams = set()
    
    for text in generated_texts:
        words = text.split()
        total_words += len(words)
        
        # Calculate n-grams
        n_grams = [tuple(words[i:i+n_gram]) for i in range(len(words)-n_gram+1)]
        unique_n_grams.update(n_grams)
    
    # Distinct-n
    distinct_n = len(unique_n_grams) / total_words if total_words > 0 else 0
    
    # Vocabulary size
    vocab_size = len(set(word for text in generated_texts for word in text.split()))
    
    # Type-Token Ratio (TTR)
    ttr = vocab_size / total_words if total_words > 0 else 0
    
    return {
        f'distinct-{n_gram}': distinct_n,
        'vocab_size': vocab_size,
        'type_token_ratio': ttr
    }

def weight_change_metrics(old_weights, new_weights):
    """
    Calculate metrics to quantify the change in weights.
    
    :param old_weights: Original model weights (state dict)
    :param new_weights: New model weights (state dict)
    :return: Dictionary of weight change metrics
    """
    total_params = sum(p.numel() for p in old_weights.values())
    
    l1_norm = sum((new_weights[name] - param).abs().sum().item() 
                  for name, param in old_weights.items()) / total_params
    
    l2_norm = math.sqrt(sum(((new_weights[name] - param) ** 2).sum().item() 
                            for name, param in old_weights.items()) / total_params)
    
    cosine_sim = sum((new_weights[name] * param).sum().item() / 
                     (torch.norm(new_weights[name]) * torch.norm(param))
                     for name, param in old_weights.items()) / len(old_weights)
    
    # Calculate percentage of weights that changed significantly
    significant_change_threshold = 0.01  # 1% change
    significant_changes = sum(
        ((new_weights[name] - param).abs() > significant_change_threshold * param.abs()).float().mean().item()
        for name, param in old_weights.items()
    ) / len(old_weights)
    
    return {
        'l1_norm': l1_norm,
        'l2_norm': l2_norm,
        'cosine_similarity': cosine_sim,
        'significant_changes': significant_changes
    }

def perplexity_improvement(old_perplexity, new_perplexity):
    """
    Calculate the relative improvement in perplexity.
    
    :param old_perplexity: Perplexity before weight update
    :param new_perplexity: Perplexity after weight update
    :return: Relative improvement in perplexity
    """
    return (old_perplexity - new_perplexity) / old_perplexity

def parameter_efficiency(model):
    """
    Calculate parameter efficiency metrics for the model.
    
    :param model: The PyTorch model
    :return: Dictionary of parameter efficiency metrics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_efficiency': trainable_params / total_params if total_params > 0 else 0
    }

def gradient_statistics(model):
    """
    Calculate gradient statistics for the model.
    
    :param model: The PyTorch model after backward pass
    :return: Dictionary of gradient statistics
    """
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    
    if not gradients:
        return {
            'grad_norm': 0,
            'grad_mean': 0,
            'grad_std': 0,
            'grad_zero_fraction': 1
        }
    
    grad_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients]))
    grad_mean = torch.mean(torch.stack([torch.mean(g) for g in gradients]))
    grad_std = torch.std(torch.stack([torch.std(g) for g in gradients]))
    grad_zero_fraction = torch.mean(torch.stack([(g == 0).float().mean() for g in gradients]))
    
    return {
        'grad_norm': grad_norm.item(),
        'grad_mean': grad_mean.item(),
        'grad_std': grad_std.item(),
        'grad_zero_fraction': grad_zero_fraction.item()
    }