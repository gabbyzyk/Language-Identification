import torch
from torch.utils.data import Dataset
from tokenization import get_max_length
from data import load_data, preprocess_data, encode_labels, split_data

class LanguageDataset(Dataset):
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        dataset_length = len(self.labels)
        
        return dataset_length
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx])
                for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        
        return item

def create_datasets(training_texts, validation_texts, testing_texts, 
                   training_labels, validation_labels, testing_labels, 
                   tokenizer, max_length=None):
    """Tokenizes texts and creates LanguageDataset instances."""
    
    if max_length is None:
        # If not provided, calculate it from training texts
        max_length = get_max_length(tokenizer, training_texts)
        print(f"Calculated max_length: {max_length}")

    # Tokenizing the training examples
    training_encodings = tokenizer(
        list(training_texts),
        add_special_tokens=True,
        max_length=max_length,
        truncation=True, 
        padding="do_not_pad",
    )

    # Tokenizing the validation examples
    validation_encodings = tokenizer(
        list(validation_texts),
        add_special_tokens=True,
        max_length=max_length,
        truncation=True, 
        padding="do_not_pad",
    )
    # Tokenizing the testing examples
    testing_encodings = tokenizer(
        list(testing_texts),
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="do_not_pad",
    )
    
    # Initializing the datasets
    training_dataset = LanguageDataset(
        encodings=training_encodings, 
        labels=training_labels,
    )

    validation_dataset = LanguageDataset(
        encodings=validation_encodings, 
        labels=validation_labels,
    )

    testing_dataset = LanguageDataset(
        encodings=testing_encodings, 
        labels=testing_labels,
    )
    
    return training_dataset, validation_dataset, testing_dataset, max_length

def prepare_all_data(data_dir, tokenizer, random_state=42):
    """
    Orchestrates the entire data preparation pipeline.
    """
    # 1. Load
    df = load_data(data_dir)
    
    # 2. Preprocess
    df_filtered = preprocess_data(df)
    
    # 3. Encode Labels
    texts_data, labels_data_encoded, class_label = encode_labels(df_filtered)
    
    # 4. Split
    splits = split_data(texts_data, labels_data_encoded, random_state)
    (training_texts, validation_texts, testing_texts, 
     training_labels, validation_labels, testing_labels) = splits
     
    # 5. Create Datasets
    training_dataset, validation_dataset, testing_dataset, max_length = create_datasets(
        training_texts, validation_texts, testing_texts, 
        training_labels, validation_labels, testing_labels, 
        tokenizer
    )
    
    return {
        "datasets": (training_dataset, validation_dataset, testing_dataset),
        "splits": (training_texts, validation_texts, testing_texts, training_labels, validation_labels, testing_labels),
        "class_label": class_label,
        "max_length": max_length,
        "df_filtered": df_filtered
    }
