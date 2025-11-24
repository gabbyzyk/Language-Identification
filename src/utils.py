import os
import random
import re
import unicodedata
import numpy as np
import torch
import sys
import time
import matplotlib.pyplot as plt
try:
    from config import EXPERIMENT_NAME
except ImportError:
    EXPERIMENT_NAME = "default_experiment"

def clean_text(text):
    """
    A minimalist cleaning function optimized for DistilBERT-multilingual-cased.
    Preserves case, punctuation, and digits as they are linguistic features.
    """
    # 1. Safety check
    if not isinstance(text, str):
        return ""

    # 2. Unicode Normalization (NFC)
    # Ensures that characters like "é" are represented consistently
    text = unicodedata.normalize("NFC", text)

    # 3. Remove Noise (URLs and HTML tags)
    # URLs are generally language-agnostic or biased towards English
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'<.*?>', '', text)   

    # 4. Remove Control Characters (Non-printable)
    # Keeps punctuations, digits, letters, and standard whitespace
    text = "".join(ch for ch in text if ch.isprintable() or ch.isspace())

    # 5. Whitespace Standardization
    # Collapses multiple spaces/tabs/newlines into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    print(f"Random seed set as {seed_value}")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_notebook_logging(output_dir):
    """
    Sets up output directories, monkey patches plt.show to save figures,
    and redirects stdout to a log file.
    """
    figures_dir = os.path.join(output_dir, "figures", EXPERIMENT_NAME)
    logs_dir = os.path.join(output_dir, "logs", EXPERIMENT_NAME)

    # Create directories
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"Figures will be saved to: {figures_dir}")
    print(f"Logs will be saved to: {logs_dir}")

    # ==========================================
    # Monkey Patch plt.show to save figures
    # ==========================================
    # Reset counter (using a global variable or attribute on the function to persist if needed, 
    # but here we just use a global variable in the module scope if we want, 
    # or attach it to the patched function)
    
    # We need to handle the counter. Since this function might be called once, 
    # we can initialize the counter.
    
    if not hasattr(plt.show, "_is_patched"):
        _original_show = plt.show
        # We attach the counter to the wrapper function to keep state
        
        def save_and_show(*args, **kwargs):
            if not hasattr(save_and_show, "counter"):
                save_and_show.counter = 1
                
            try:
                fig = plt.gcf()
                title = ""
                # Try to get title from axes
                if fig.axes:
                    for ax in fig.axes:
                        t = ax.get_title()
                        if t:
                            title = t
                            break
                
                # If no title on axes, check suptitle
                if not title and hasattr(fig, "_suptitle") and fig._suptitle:
                    title = fig._suptitle.get_text()
                    
                if not title:
                    title = f"figure_{save_and_show.counter}"
                    save_and_show.counter += 1
                
                # Sanitize filename
                filename = re.sub(r'[\\/*?:"<>|]', "", title)
                filename = filename.replace(" ", "_").replace("\n", "_")
                filename = filename[:100] # Limit length
                
                save_path = os.path.join(figures_dir, f"{filename}.png")
                
                # Save
                fig.savefig(save_path, bbox_inches='tight')
                print(f"Saved figure to {save_path}")
            except Exception as e:
                print(f"Failed to save figure: {e}")
            
            # Call original show
            _original_show(*args, **kwargs)

        save_and_show._is_patched = True
        plt.show = save_and_show
        print("plt.show has been patched to save figures automatically.")
    else:
        print("plt.show is already patched.")

    # ==========================================
    # Redirect stdout to log file
    # ==========================================
    class Tee(object):
        def __init__(self, name, mode):
            self.file = open(name, mode)
            self.stdout = sys.stdout
            sys.stdout = self
        def __del__(self):
            if sys.stdout is self:
                sys.stdout = self.stdout
            if not self.file.closed:
                self.file.close()
        def write(self, data):
            try:
                self.file.write(data)
                self.file.flush()
            except Exception:
                pass
            self.stdout.write(data)
        def flush(self):
            try:
                self.file.flush()
            except Exception:
                pass
            self.stdout.flush()
        def __getattr__(self, attr):
            return getattr(self.stdout, attr)

    log_file_path = os.path.join(logs_dir, "notebook_output.log")

    # Reset if already redirected (simple check)
    if hasattr(sys.stdout, 'file') and hasattr(sys.stdout, 'stdout'):
         sys.stdout = sys.stdout.stdout

    sys.stdout = Tee(log_file_path, "w")
    print(f"Stdout redirected to {log_file_path}")
