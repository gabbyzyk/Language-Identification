import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_target_structure(labels, class_label, fig_title="Title_1"):
    """Plotting the shares of Dataset labels."""
    # Computing the unique labels
    unique_labels = np.unique(labels)
    # Computing the number of objects within each class
    labels_count = np.bincount(labels)
    
    # Computing the shares of objects in each class
    n_obj = labels.shape[0]
    
    labels_info_share = pd.Series(
        labels_count, index=class_label.names
    ) 
    
    # Plotting a figure
    labels_info_share.plot(kind="bar")
    plt.xticks(rotation=0)
    plt.title(fig_title, fontsize=15)
    plt.xlabel("Class name")
    plt.ylabel("Proportion of objects")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def display_encodings_info(tokenizer, encodings, texts, labels, text_idx, class_label):
    # Original text and label for the sample
    original_text = texts.iloc[text_idx]  
    original_label_encoded = labels[text_idx]
    
    # Retrieve the tokenized encoding for this specific sample
    encoded_tokens = encodings["input_ids"][text_idx]
    
    # Decode the tokens back to text
    decoded_text = tokenizer.decode(encoded_tokens, skip_special_tokens=True)
    
    # Display results
    print("Original Text:", original_text)
    print("Encoded Tokens:", encoded_tokens)
    print("Decoded Text from Tokens:", decoded_text)
    print("Label (encoded):", original_label_encoded)
    print("Label (decoded):", class_label.int2str(int(original_label_encoded)))  # Convert back to the original label

def plot_training_history(history, num_epochs):
    # Initialize lists to store metrics
    train_loss = []
    train_epochs = []
    val_loss = []
    val_epochs = []
    val_acc = []
    val_f1 = []

    # Parse the history
    for entry in history:
        # Training loss is logged at specific steps
        if 'loss' in entry:
            train_loss.append(entry['loss'])
            train_epochs.append(entry['epoch'])
        
        # Validation metrics are logged at the end of each epoch
        if 'eval_loss' in entry:
            val_loss.append(entry['eval_loss'])
            val_epochs.append(entry['epoch'])
            # Check for keys, they might be prefixed with 'eval_'
            val_acc.append(entry.get('eval_accuracy', 0))
            val_f1.append(entry.get('eval_f1', 0))

    # Plotting
    plt.figure(figsize=(18, 6))

    # 1. Loss Curves (Train vs Validation)
    plt.subplot(1, 2, 1)
    plt.plot(train_epochs, train_loss, label='Training Loss', alpha=0.7)
    plt.plot(val_epochs, val_loss, label='Validation Loss', marker='o', linewidth=2, color='orange')
    plt.title('Training Dynamics: Loss Curves', fontsize=15)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 2. Validation Metrics (Accuracy & F1)
    plt.subplot(1, 2, 2)
    plt.plot(val_epochs, val_acc, label='Validation Accuracy', marker='o', linewidth=2, color='green')
    plt.plot(val_epochs, val_f1, label='Validation F1 Score', marker='s', linewidth=2, color='blue')
    plt.title('Generalization: Validation Accuracy & F1', fontsize=15)
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # Print Early Stopping Information
    stopped_epoch = len(val_epochs)
    print(f"Training finished after {stopped_epoch} epochs.")
    if stopped_epoch < num_epochs:
        print(f"Early stopping triggered at epoch {stopped_epoch} (Max epochs: {num_epochs}).")
    else:
        print(f"Training completed all {num_epochs} epochs.")

def plot_confusion_matrix_custom(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(include_values=False, cmap="Blues", xticks_rotation=90)

    plt.title("Confusion Matrix on Test Set", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return cm

def plot_msp_calibration(probs, preds, labels, n_bins=15):
    msp = probs.max(axis=1)
    correct = (preds == labels).astype(int)

    frac_pos, mean_pred = calibration_curve(correct, msp, n_bins=n_bins)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(mean_pred, frac_pos, 'o-', label='MSP Calibration')

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of correct predictions")
    plt.title("MSP Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_top5_calibration(probs, labels, class_names, top_k=5, n_bins=10):
    # Count per-class frequency
    counts = np.bincount(labels)
    top_classes = np.argsort(counts)[-top_k:]

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

    palette = sns.color_palette('husl', top_k)

    for idx, cls in enumerate(top_classes):
        y_true_binary = (labels == cls).astype(int)
        y_prob = probs[:, cls]

        frac_pos, mean_pred = calibration_curve(y_true_binary, y_prob, n_bins=n_bins)
        plt.plot(mean_pred, frac_pos, marker='o', label=class_names[cls], color=palette[idx])

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Top-5 Frequent Classes: Calibration Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_ece(probs, preds, labels, n_bins=15):
    msp = probs.max(axis=1)
    correct = (preds == labels).astype(int)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(msp, bins) - 1

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_sizes = []

    for i in range(n_bins):
        idx = bin_ids == i
        if np.sum(idx) > 0:
            avg_conf = msp[idx].mean()
            avg_acc = correct[idx].mean()
            bin_size = np.sum(idx) / len(correct)

            ece += np.abs(avg_acc - avg_conf) * bin_size

            bin_accs.append(avg_acc)
            bin_confs.append(avg_conf)
            bin_sizes.append(np.sum(idx))

        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
            bin_sizes.append(0)

    return ece, bins, bin_accs, bin_confs, bin_sizes

def plot_ece_bar(bin_accs, bin_confs, bins):
    plt.figure(figsize=(8, 4))
    x = (bins[:-1] + bins[1:]) / 2

    errors = np.abs(np.array(bin_accs) - np.array(bin_confs))

    plt.bar(x, errors, width=0.06, alpha=0.7)
    plt.xlabel("Confidence bins")
    plt.ylabel("|Accuracy − Confidence|")
    plt.title("ECE per Confidence Bin")
    plt.grid(True)
    plt.show()

def plot_embeddings(embeddings_2d, labels, title, class_names):
    plt.figure(figsize=(16, 10))
    
    # Create a DataFrame for easier plotting
    df_plot = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': [class_names[l] for l in labels]
    })
    
    try:
        sns.scatterplot(
            data=df_plot,
            x='x',
            y='y',
            hue='label',
            palette='tab20', # A palette with many colors
            s=60,
            alpha=0.7
        )
    except:
        # Fallback to matplotlib if seaborn fails or palette issue
        for label_name in class_names:
            subset = df_plot[df_plot['label'] == label_name]
            plt.scatter(subset['x'], subset['y'], label=label_name, alpha=0.7, s=60)

    plt.title(title, fontsize=18)
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Languages")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_confidence_distribution(id_confidences, ood_confidences):
    plt.figure(figsize=(12, 6))

    # Plot ID distribution
    sns.histplot(
        id_confidences, 
        color='blue', 
        label='ID Data (Validation)', 
        kde=True, 
        stat="density", 
        element="step", 
        alpha=0.3,
        bins=30
    )

    # Plot OOD distribution
    sns.histplot(
        ood_confidences, 
        color='red', 
        label='OOD Data (Noise/Code/etc.)', 
        kde=True, 
        stat="density", 
        element="step", 
        alpha=0.3,
        bins=30
    )

    plt.title("Confidence Distribution: In-Distribution vs Out-of-Distribution", fontsize=16)
    plt.xlabel("Max Softmax Probability (Confidence)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 1.05)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc, best_threshold, best_idx):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Mark the optimal threshold
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', marker='o', s=100, zorder=5, 
                label=f'Optimal Threshold: {best_threshold:.4f}')

    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
