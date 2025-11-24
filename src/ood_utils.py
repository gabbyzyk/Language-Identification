import random
import string
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

# ====================================================
# OOD Data Generation Functions
# ====================================================

def generate_noise(n=20, length_range=(8, 20)):
    chars = string.ascii_letters + string.digits + string.punctuation
    samples = []
    for _ in range(n):
        length = random.randint(*length_range)
        s = ''.join(random.choice(chars) for _ in range(length))
        samples.append(s)
    return samples

def generate_code_snippets(n=20):
    code_pool = [
        "def foo(x): return x+1",
        "for i in range(10): print(i)",
        "console.log('Hello World');",
        "if (a > b) { return a; }",
        "public static void main(String[] args) {}",
        "x = [i for i in range(5)]",
        "print('test', value)",
        "int sum(int a, int b) { return a+b; }",
        "const result = compute(a, b);",
        "<html><body>Hello</body></html>"
    ]
    # Ensure we have enough samples if n > len(code_pool)
    return random.sample(code_pool * (n // len(code_pool) + 1), n)

def generate_unseen_lang(n=20):
    unseen_pool = [
        # Greek
        "Αυτό είναι ένα ελληνικό κείμενο που δεν εμφανίζεται στο training set.",
        # Hebrew
        "זהו משפט בעברית שלא הופיע בנתוני האימון.",
        # Ukrainian
        "Це український текст, якого не було в навчальних даних.",
        # Vietnamese
        "Đây là một câu tiếng Việt không có trong tập huấn luyện.",
        # Bengali
        "এটি একটি বাংলা বাক্য যা প্রশিক্ষণ ডাটাতে ছিল না।",
        # Polish
        "To jest zdanie w języku polskim, którego nie było w zbiorze treningowym.",
        # Swahili
        "Hii ni sentensi ya Kiswahili ambayo haipo kwenye data ya mafunzo."
    ]

    samples = []
    for _ in range(n):
        samples.append(random.choice(unseen_pool))
    return samples

def generate_short(n=20):
    pool = ["hi", "ok", "yo", "ha", "go", "no", "uh", "eh", "ya", "lo", "aa", "zz"]
    return random.sample(pool * (n // len(pool) + 1), n)

def generate_emoji(n=20):
    emoji_pool = ["😂", "😍", "🔥", "😭", "✨", "😡", "💯", "🙌", "🥺", "🤔"]
    samples = []
    for _ in range(n):
        length = random.randint(3, 10)
        s = ''.join(random.choice(emoji_pool) for _ in range(length))
        samples.append(s)
    return samples

def generate_numbers(n=20):
    return [str(random.randint(10**6, 10**12)) for _ in range(n)]

def generate_urls(n=20):
    domains = ["google.com", "example.org", "github.com", "openai.com", "univ.edu"]
    paths = ["index", "home", "api", "search", "login", "docs"]

    samples = []
    for _ in range(n):
        url = f"https://{random.choice(domains)}/{random.choice(paths)}?id={random.randint(1,9999)}"
        samples.append(url)
    return samples

def create_ood_dataset_df(n_samples=20):
    """
    Generates a DataFrame containing various types of OOD samples.
    """
    ood_data = {"type": [], "text": []}

    def add_samples(name, samples):
        for s in samples:
            ood_data["type"].append(name)
            ood_data["text"].append(s)

    add_samples("noise",        generate_noise(n_samples))
    add_samples("code",         generate_code_snippets(n_samples))
    add_samples("unseen_lang",  generate_unseen_lang(n_samples))
    add_samples("short",        generate_short(n_samples))
    add_samples("emoji",        generate_emoji(n_samples))
    add_samples("numbers",      generate_numbers(n_samples))
    add_samples("url",          generate_urls(n_samples))

    return pd.DataFrame(ood_data)

# ====================================================
# OOD Metrics Calculation
# ====================================================

def calculate_ood_metrics(id_confidences, ood_confidences):
    """
    Calculates ROC curve, AUC, and optimal threshold using Youden's J statistic.
    
    Args:
        id_confidences: Array of confidence scores for ID data.
        ood_confidences: Array of confidence scores for OOD data.
        
    Returns:
        dict: Dictionary containing metrics and curve data.
    """
    # 1. Construct Binary Labels & Scores
    # ID = 1 (Positive), OOD = 0 (Negative)
    y_true = np.concatenate([np.ones(len(id_confidences)), np.zeros(len(ood_confidences))])
    y_scores = np.concatenate([id_confidences, ood_confidences])

    # 2. Calculate ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 3. Calculate Youden's J Statistic
    # J = Sensitivity (TPR) + Specificity (1 - FPR) - 1
    # J = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    best_j = j_scores[best_idx]
    
    # 4. Performance at Threshold
    # ID samples kept (TPR at threshold)
    id_kept = np.sum(id_confidences >= best_threshold)
    id_acc = id_kept / len(id_confidences)

    # OOD samples rejected (True Negative Rate at threshold)
    ood_rejected = np.sum(ood_confidences < best_threshold)
    ood_rej_rate = ood_rejected / len(ood_confidences)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "roc_auc": roc_auc,
        "best_threshold": best_threshold,
        "best_idx": best_idx,
        "best_j": best_j,
        "id_acc": id_acc,
        "ood_rej_rate": ood_rej_rate
    }
