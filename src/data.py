import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import ClassLabel
from utils import clean_text

def load_data(data_dir):
    """Loads the dataset from CSV."""
    return pd.read_csv(data_dir)

def preprocess_data(df):
    """Cleans the text data."""
    df_filtered = df.copy()
    df_filtered['Text'] = df_filtered['Text'].apply(lambda x: clean_text(x))
    df_filtered = df_filtered.reset_index(drop=True)
    return df_filtered

def encode_labels(df):
    """Encodes the language labels."""
    texts_data = df["Text"]
    labels_data = df["language"]
    
    unique_labels = sorted(labels_data.unique())
    class_label = ClassLabel(names=unique_labels)
    
    labels_data_encoded = labels_data.apply(lambda x: class_label.str2int(x)).values
    
    return texts_data, labels_data_encoded, class_label

def split_data(texts_data, labels_data_encoded, random_state=42):
    """Splits data into train, validation, and test sets."""
    # Separating data into training set and validation/test sets
    (
        training_texts, 
        validation_testing_texts, 
        training_labels, 
        validation_testing_labels
    ) = train_test_split(
        texts_data,
        labels_data_encoded,
        train_size=0.8,
        random_state=random_state,
        stratify=labels_data_encoded,
    )
    # Separating validation and test sets
    (
        validation_texts, 
        testing_texts, 
        validation_labels, 
        testing_labels
    ) = train_test_split(
        validation_testing_texts,
        validation_testing_labels,
        train_size=0.5,
        random_state=random_state,
        stratify=validation_testing_labels,
    )
    
    return (training_texts, validation_texts, testing_texts, 
            training_labels, validation_labels, testing_labels)
