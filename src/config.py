import torch

DATA_DIR = "/home/zyk/Language Identification/Dataset/kaggle-Language Identification dataset/dataset.csv"
MODEL_NAME = "distilbert-base-multilingual-cased"
EXPERIMENT_NAME = "distilbert"
RANDOM_STATE = 42
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 8
NUM_WORKERS = 4
