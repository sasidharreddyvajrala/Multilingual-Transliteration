import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Path where you saved merged model
model_path = "/content/byt5-translit-merged"

# Load tokenizer and merged model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Your input text
text = "namaste duniya"

# Tokenize
inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

# Decode
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Input:", text)
print("Output:", prediction)
