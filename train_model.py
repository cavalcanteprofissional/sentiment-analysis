"""
Optional: Train your own sentiment analysis model using the IMDB dataset.
This script demonstrates fine-tuning a pre-trained model for sentiment analysis.
"""

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def tokenize_function(examples):
    """Tokenize the text examples."""
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)


def compute_metrics(pred):
    """Compute evaluation metrics."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


if __name__ == "__main__":
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    print("Loading pre-trained model and tokenizer...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir="./sentiment-model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating...")
    metrics = trainer.evaluate()
    print(f"Final metrics: {metrics}")

    print("Saving model...")
    trainer.save_model("./sentiment-model")
    tokenizer.save_pretrained("./sentiment-model")

    print("Model trained and saved! You can now use it in app.py by loading:")
    print('pipeline = pipeline("sentiment-analysis", model="./sentiment-model")')
