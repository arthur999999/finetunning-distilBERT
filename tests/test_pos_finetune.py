import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def predict_sentiment(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "positive" if prediction == 1 else "negative"

if __name__ == "__main__":
    model_path = "my_finetuned_model"
    tokenizer, model = load_model(model_path)
    
    test_texts = [
        "I absolutely loved this movie. The acting was fantastic!",
        "The film was boring and too long. I wouldn't recommend it.",
        "Amazing storyline and great performances by the actors!",
        "Terrible script, bad direction, and awful cinematography."
    ]
    
    for text in test_texts:
        sentiment = predict_sentiment(model, tokenizer, text)
        print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")
