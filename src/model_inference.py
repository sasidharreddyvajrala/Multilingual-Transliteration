# src/model_inference.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

def load_model_and_tokenizer(model_name="google/byt5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def generate_batch(model, tokenizer, texts: list[str], device, **gen_kwargs):
    inputs = tokenizer(
        texts,
        padding="longest",
        return_tensors="pt",
        truncation=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            early_stopping=True,
            length_penalty=1.2,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            **gen_kwargs
        )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def run_inference_on_dataset(model, tokenizer, dataset, device, batch_size=32):
    predictions = []
    references = dataset["label"]
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_texts = dataset[i:i+batch_size]["text"]
        preds = generate_batch(model, tokenizer, batch_texts, device)
        predictions.extend(preds)
    
    return predictions, references