import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from training import compute_metrics

class DistilBertClassifier:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest"
        )

    def build_model(self, num_labels, id2label_mappings):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=id2label_mappings
        )
        self.model.to(self.device)

    def train(self, train_dataset, validation_dataset, output_dir, batch_size, num_epochs, num_workers, random_state):
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() first.")

        # --- Discriminative Learning Rates Implementation ---
        # Use smaller LR for the pre-trained base (distilbert) to preserve knowledge
        # Use larger LR for the randomly initialized classifier heads to learn quickly
        LR_BASE = 2e-5
        LR_HEAD = 1e-3

        # Separate parameters
        base_params = [p for n, p in self.model.named_parameters() if "distilbert" in n]
        head_params = [p for n, p in self.model.named_parameters() if "distilbert" not in n]

        # Create parameter groups
        optimizer_grouped_parameters = [
            {"params": base_params, "lr": LR_BASE},
            {"params": head_params, "lr": LR_HEAD}
        ]

        # Initialize custom optimizer
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        # --------------------------------------------------

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=100,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            fp16=True,  # automatic mixed precision
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=3,
            warmup_ratio=0.1,
            # learning_rate=5e-5, # This will be ignored by the custom optimizer
            weight_decay=0.01,
            label_smoothing_factor=0.1,
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=True,
            lr_scheduler_type="cosine",
            report_to="none",
            seed=random_state,
        )

        # Instantiate Trainer with custom optimizer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            optimizers=(optimizer, None) # Pass the optimizer, let Trainer create the scheduler
        )

        # Start training
        trainer.train()
        
        return trainer
