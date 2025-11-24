import numpy as np

def get_max_length(tokenizer, texts, percentile=95):
    """Calculates the optimal max length for tokenization."""
    lengths = [len(tokenizer.encode(text)) for text in texts]
    max_length = int(np.percentile(lengths, percentile))
    return max_length
