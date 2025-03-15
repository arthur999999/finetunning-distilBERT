import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset

# Load dataset (IMDB reviews)
dataset = load_dataset("imdb")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenize the dataset
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

encoded_dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])

# Define data collator to handle padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,  # Increase for better results
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"].shuffle(seed=42).select(range(1000)),
    eval_dataset=encoded_dataset["test"].shuffle(seed=42).select(range(200)),
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

save_path = "./my_finetuned_model"  # Change this to your preferred directory
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Inference
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "positive" if prediction == 1 else "negative"

# Test the model
example_text = "I absolutely loved this movie. The acting was fantastic!"
result = predict_sentiment(example_text)
print(f"Sentiment: {result}")
