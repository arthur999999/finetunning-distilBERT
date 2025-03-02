import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load dataset (IMDB reviews)
dataset = load_dataset("imdb")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenize the dataset
def preprocess(example):
    return tokenizer(example['text'], truncation=True, padding=True)

encoded_dataset = dataset.map(preprocess, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_dir="./logs",
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'].shuffle().select(range(1000)),
    eval_dataset=encoded_dataset['test'].shuffle().select(range(200)),
)

# Train the model
trainer.train()

# Inference
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    sentiment = "positive" if prediction == 1 else "negative"
    return sentiment

# Test the model
example_text = "I absolutely loved this movie. The acting was fantastic!"
result = predict_sentiment(example_text)
print(f"Sentiment: {result}")

# Let me know if you want me to add data visualizations or improve training settings! 🚀
