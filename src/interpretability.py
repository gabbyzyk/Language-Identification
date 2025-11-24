import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from tqdm import tqdm
from termcolor import colored
from torch.utils.data import DataLoader
from utils import clean_text

def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)
            
            # Get model outputs with hidden states
            # We need to ensure output_hidden_states=True is passed
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Extract CLS token embeddings (last hidden state, first token)
            # hidden_states is a tuple, last element is the last layer
            cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            
            embeddings.append(cls_embeddings)
            labels.append(batch_labels.cpu().numpy())
            
    return np.concatenate(embeddings), np.concatenate(labels)

def get_logits(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting logits"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            batch_labels = batch['labels'].cpu().numpy()
            
            outputs = model(input_ids, attention_mask=attention_mask)
            batch_logits = outputs.logits.cpu().numpy()   # shape (batch, num_classes)
            
            all_logits.append(batch_logits)
            all_labels.append(batch_labels)
    
    return np.concatenate(all_logits), np.concatenate(all_labels)

def color_text_by_language(text, lang):
    color_map = {
        "English": "cyan",
        "French": "yellow",
        "German": "green",
        "Spanish": "magenta",
        "Chinese": "red",
        "Arabic": "blue"
    }
    return colored(text, color_map.get(lang, "white"))

def analyze_mixed_language_offset(text, tokenizer, model, class_names,
                                  window_size=10, stride=5, device="cuda"):
    cleaned_text = clean_text(text)

    # --------- Global Prediction ---------
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    main_lang = class_names[pred_idx]

    # --------- Tokenization with Offsets ---------
    encoding = tokenizer(
        cleaned_text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    n_tokens = len(token_ids)

    token_votes = collections.defaultdict(list)

    # --------- Sliding Window Prediction ---------
    for start in range(0, max(1, n_tokens - window_size + 1), stride):
        end = min(start + window_size, n_tokens)
        win_ids = token_ids[start:end]

        win_inputs = {
            "input_ids": torch.tensor([win_ids]).to(device),
            "attention_mask": torch.ones(1, len(win_ids)).to(device)
        }

        with torch.no_grad():
            win_outputs = model(**win_inputs)
            win_probs = torch.nn.functional.softmax(win_outputs.logits, dim=1)
            win_pred = torch.argmax(win_probs, dim=1).item()

        win_lang = class_names[win_pred]

        for idx in range(start, end):
            token_votes[idx].append(win_lang)

    # --------- Majority Vote per Token ---------
    final_langs = []
    for i in range(n_tokens):
        if token_votes[i]:
            final_langs.append(collections.Counter(token_votes[i]).most_common(1)[0][0])
        else:
            final_langs.append(main_lang)

    # --------- Merge Tokens into Segments ---------
    segments = []
    cur_lang = final_langs[0]
    cur_start = offsets[0][0]
    cur_end = offsets[0][1]

    for i in range(1, n_tokens):
        lang = final_langs[i]
        off_s, off_e = offsets[i]

        if lang == cur_lang:
            cur_end = off_e
        else:
            seg_text = cleaned_text[cur_start:cur_end]
            segments.append((seg_text, cur_lang))
            
            cur_lang = lang
            cur_start = off_s
            cur_end = off_e

    seg_text = cleaned_text[cur_start:cur_end]
    segments.append((seg_text, cur_lang))

    return main_lang, segments

def print_analysis(text, main_lang, segments):
    print("\n" + "="*120)
    print(f"INPUT:\n{text}\n")
    print(f"GLOBAL PREDICTION: {main_lang}")
    print("\nSEGMENTED ANALYSIS:")

    colored_output = ""
    for seg, lang in segments:
        colored_output += color_text_by_language(seg, lang) + " "
    
    print(colored_output)
    print("="*120 + "\n")

def visualize_attention(text, model, tokenizer, device):
    """
    Visualize the attention weights from the [CLS] token to all input tokens
    for the fine-tuned DistilBERT model.

    The visualization highlights which tokens the model attends to most strongly,
    revealing language-discriminative cues in single-language and mixed-language inputs.
    """

    # -------------------------
    # 1. Clean + Tokenize
    # -------------------------
    cleaned_text = clean_text(text)
    encoded = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_attention_mask=True
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # -------------------------
    # 2. Forward pass with attentions
    # -------------------------
    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)

    # -------------------------
    # 3. Extract last-layer attention
    # Shape: (batch, heads, seq_len, seq_len)
    # -------------------------
    last_layer_attn = outputs.attentions[-1][0]   # first batch → (heads, seq, seq)

    # Average over all heads → (seq_len, seq_len)
    attn_matrix = last_layer_attn.mean(dim=0).cpu().numpy()

    # Select attention FROM the [CLS] token (idx 0)
    cls_attention = attn_matrix[0]   # shape: (seq_len,)

    # -------------------------
    # 4. Convert IDs → tokens for plotting
    # -------------------------
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    # -------------------------
    # 5. Create attention heatmap (1 x seq_len)
    # -------------------------
    plt.figure(figsize=(14, 2.8))
    sns.heatmap(
        [cls_attention],
        xticklabels=tokens,
        yticklabels=["[CLS] → Tokens"],
        cmap="Reds",
        cbar=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.6}
    )

    plt.title(f"Last-Layer Attention from [CLS] — \"{text[:50]}...\"", fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def visualize_bert_attention(idx, model, tokenizer, device, testing_texts, testing_labels, class_names):
    # Get text and label
    text = testing_texts.iloc[idx]
    true_label_idx = testing_labels[idx]
    true_label = class_names[true_label_idx]
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Model inference with attentions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Predictions
    logits = outputs.logits
    pred_idx = torch.argmax(logits, dim=1).item()
    pred_label = class_names[pred_idx]
    prob = torch.softmax(logits, dim=1)[0, pred_idx].item()
    
    # Get Attention: Last Layer, Average over Heads, [CLS] token row
    # outputs.attentions is a tuple of tensors (one for each layer)
    # We take the last layer: outputs.attentions[-1] -> (batch, heads, seq_len, seq_len)
    last_layer_att = outputs.attentions[-1][0].cpu().numpy() # (heads, seq_len, seq_len)
    
    # Average over heads
    avg_att = np.mean(last_layer_att, axis=0) # (seq_len, seq_len)
    
    # Focus on [CLS] token (index 0) attending to all other tokens
    cls_att = avg_att[0] # (seq_len,)
    
    # Visualization
    print(f"\n{'='*100}")
    print(f"Example Index: {idx}")
    print(f"Text: {text}")
    print(f"True Label: {true_label} | Predicted: {pred_label} (Prob: {prob:.4f})")
    
    # Plot Heatmap
    plt.figure(figsize=(20, 3))
    # Reshape for heatmap (1, seq_len)
    sns.heatmap(cls_att.reshape(1, -1), xticklabels=tokens, yticklabels=['[CLS] Attention'], 
                cmap="Reds", cbar=True, square=False)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title(f"Attention Weights ([CLS] -> Tokens) | True: {true_label} -> Pred: {pred_label}")
    plt.tight_layout()
    plt.show()

def run_standard_case_study(model, tokenizer, device, class_names):
    cases = [
        {"text": "The quick brown fox jumps over the lazy dog.", "language": "English"},
        {"text": "La vie est belle quand on profite de chaque instant.", "language": "French"},
        {"text": "El conocimiento es poder y la educación es la clave.", "language": "Spanish"},
        {"text": "今天天气真好，我们去公园散步吧。", "language": "Chinese"},
        {"text": "اللغة العربية هي لغة جميلة جداً ولها تاريخ عريق.", "language": "Arabic"},
        {"text": "Москва — столица России, красивый и большой город.", "language": "Russian"},
        {"text": "Das Wetter ist heute sehr schön und die Sonne scheint.", "language": "German"},
        {"text": "La vita è bella e bisogna godersela al massimo.", "language": "Italian"},
        {"text": "Obrigado por tudo, você foi muito gentil comigo.", "language": "Portuguese"},
        {"text": "Merhaba dünya, bugün nasılsın?", "language": "Turkish"}
    ]

    results = []

    # Ensure model is in evaluation mode
    model.eval()

    print(f"{'Text':<60} | {'True':<10} | {'Pred':<10} | {'Conf':<6}")
    print("-" * 95)

    for case in cases:
        text = case["text"]
        true_lang = case["language"]
        
        # Preprocess using the same cleaning function
        cleaned_text = clean_text(text)
        
        # Tokenize
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            
        pred_lang = class_names[pred_idx.item()]
        
        results.append({
            "Text": text,
            "True Language": true_lang,
            "Predicted Language": pred_lang,
            "Confidence": conf.item()
        })
        
        print(f"{text[:57]+'...':<60} | {true_lang:<10} | {pred_lang:<10} | {conf.item():.4f}")

    # Display as DataFrame for better visualization
    return pd.DataFrame(results)

def run_hard_case_study(model, tokenizer, device, class_names):
    hard_cases = [
        # --- Very Short Texts ---
        {"text": "No", "type": "Short / Ambiguous", "expected": "Spanish/English/Italian..."},
        {"text": "Chat", "type": "Short / Ambiguous", "expected": "French (Cat) / English"},
        {"text": "Gift", "type": "Short / False Friend", "expected": "English (Gift) / German (Poison)"},
        
        # --- Similar Languages ---
        {"text": "Eu não sei o que fazer.", "type": "Similar (Port/Span)", "expected": "Portuguese"},
        {"text": "No sé qué hacer.", "type": "Similar (Port/Span)", "expected": "Spanish"},
        {"text": "Selamat pagi dunia.", "type": "Similar (Indo/Malay)", "expected": "Indonesian/Malay"},
        
        # --- Out-of-Domain / Noisy ---
        {"text": "😂😂😂 🚀🚀🚀", "type": "Emoji Only", "expected": "Unknown/Noise"},
        {"text": "12345 67890", "type": "Numeric", "expected": "Unknown/Noise"},
        {"text": "def train_model(x): return x * 2", "type": "Code Snippet", "expected": "English (usually)"},
        {"text": "C'est la vie 🏖️", "type": "Mixed + Emoji", "expected": "French"},
    ]

    print(f"{'Type':<22} | {'Text':<30} | {'Expected':<30} | {'Pred':<10} | {'Conf':<6}")
    print("-" * 110)

    for case in hard_cases:
        text = case["text"]
        case_type = case["type"]
        expected = case["expected"]
        
        # Preprocess
        cleaned_text = clean_text(text)
        
        if not cleaned_text.strip():
            pred_lang = "N/A (Empty)"
            conf = 0.0
        else:
            inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
                
            pred_lang = class_names[pred_idx.item()]
            conf = conf.item()
        
        # Truncate text for display
        display_text = (text[:27] + '...') if len(text) > 27 else text
        
        print(f"{case_type:<22} | {display_text:<30} | {expected:<30} | {pred_lang:<10} | {conf:.4f}")

def get_error_cases(preds, testing_labels, testing_texts, class_names, logits):
    import torch.nn.functional as F
    # Calculate probabilities
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()

    # Find indices of misclassified examples
    error_indices = np.where(preds != testing_labels)[0]

    # Select 20 random error cases (or fewer if there aren't 20)
    num_samples = min(20, len(error_indices))
    if num_samples > 0:
        selected_indices = np.random.choice(error_indices, num_samples, replace=False)
        
        error_cases = []
        for idx in selected_indices:
            text = testing_texts.iloc[idx]
            true_label = class_names[testing_labels[idx]]
            pred_label = class_names[preds[idx]]
            max_prob = np.max(probs[idx])
            
            # Truncate text
            truncated_text = text[:100] + "..." if len(text) > 100 else text
            
            error_cases.append({
                "Text (Truncated)": truncated_text,
                "True": true_label,
                "Pred": pred_label,
                "Max Prob": f"{max_prob:.4f}"
            })
        
        error_df = pd.DataFrame(error_cases)
        return error_df, selected_indices
    else:
        print("No misclassified examples found!")
        return None, []

def get_confused_pairs(cm, class_names):
    """
    Identifies the most difficult language pairs to distinguish based on the confusion matrix.
    """
    # Calculate percentage confusion matrix
    row_sums = cm.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm_norm = cm.astype('float') / row_sums[:, np.newaxis]

    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                confused_pairs.append({
                    'True Language': class_names[i],
                    'Predicted Language': class_names[j],
                    'Count': cm[i, j],
                    'Percentage': cm_norm[i, j]
                })

    # Create DataFrame
    df_confused = pd.DataFrame(confused_pairs)

    # Sort by Count descending
    df_confused_sorted = df_confused.sort_values(by='Count', ascending=False)
    
    return df_confused_sorted

def visualize_top_confused_pairs(df_confused_sorted, model, tokenizer, device, testing_labels, preds, class_names, testing_texts, top_k=2, examples_per_pair=5):
    """
    Visualizes attention weights for the top k most frequent confusion pairs.
    """
    # Get top k confused pairs
    top_pairs = df_confused_sorted.head(top_k)

    print(f"Visualizing attention for the top {top_k} confused pairs:")

    for index, row in top_pairs.iterrows():
        true_lang = row['True Language']
        pred_lang = row['Predicted Language']
        count = row['Count']
        
        print(f"\n{'#'*50}")
        print(f"Pair: True '{true_lang}' -> Predicted '{pred_lang}' (Total Count: {count})")
        print(f"{'#'*50}")
        
        # Map language names back to IDs
        true_label_id = class_names.index(true_lang)
        pred_label_id = class_names.index(pred_lang)
        
        # Find indices in the test set where this specific confusion occurred
        pair_indices = np.where((testing_labels == true_label_id) & (preds == pred_label_id))[0]
        
        indices_to_show = pair_indices[:examples_per_pair]
        
        if len(indices_to_show) == 0:
            print("No examples found.")
            continue
            
        for i, idx in enumerate(indices_to_show):
            print(f"\n--- Example {i+1}/{min(len(pair_indices), examples_per_pair)} (Index: {idx}) ---")
            visualize_bert_attention(idx, model, tokenizer, device, testing_texts, testing_labels, class_names)
            
        if len(pair_indices) > examples_per_pair:
            print(f"\n... and {len(pair_indices) - examples_per_pair} more examples not shown.")
