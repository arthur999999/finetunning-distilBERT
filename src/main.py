import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset

dataset = load_dataset("imdb")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

encoded_dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3, 
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"].shuffle(seed=42).select(range(1000)),
    eval_dataset=encoded_dataset["test"].shuffle(seed=42).select(range(200)),
    tokenizer=tokenizer,
    data_collator=data_collator,
)


trainer.train()

save_path = "./my_finetuned_model" 
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "positive" if prediction == 1 else "negative"

example_text = "I absolutely loved this movie. The acting was fantastic!"
result = predict_sentiment(example_text)
print(f"Sentiment: {result}")
