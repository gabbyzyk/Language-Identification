import time
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def evaluate_latency(model, tokenizer, texts, device, num_warmup=10, num_samples=50):
    model.eval()
    model.to(device)
    
    # Prepare input
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    batch_size = len(texts)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(**inputs)
            
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_samples):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            _ = model(**inputs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000) # Convert to ms

    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    return mean_latency, std_latency, batch_size
