# src/training.py

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
import numpy as np
import torch
torch.cuda.empty_cache()
import os,shutil



def get_lora_config():
    return LoraConfig(
        r=8,                        # small → low memory
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

def prepare_model_for_lora(model):
    model = prepare_model_for_kbit_training(model)
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # shows how few params are trained
    return model

def get_training_args(output_dir, num_epochs=1, batch_size=8, accum_steps=4):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=3e-6,
        weight_decay=0.01,
        fp16=False,
        predict_with_generate=True,
        generation_max_length=64,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        resume_from_checkpoint=True,
        report_to="none"
    )

def make_compute_metrics(tokenizer):
    cer_metric = evaluate.load("cer")
    pad_token_id = tokenizer.pad_token_id or 0

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels != -100, labels, pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        exact = np.mean([p == l for p, l in zip(decoded_preds, decoded_labels)])

        return {
            "cer": round(cer, 4),
            "exact_match": round(exact, 4)
        }

    return compute_metrics


def run_lora_training(model, tokenizer, train_subset, eval_subset, output_dir="./byt5-lora-local"):

    # Optional: clean output dir to avoid old checkpoints
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model = prepare_model_for_lora(model)

    def preprocess(examples):
        inputs = tokenizer(examples["text"], max_length=64, truncation=True, padding="max_length")
        labels = tokenizer(examples["label"], max_length=64, truncation=True, padding="max_length")
        labels_ids = labels["input_ids"]
        labels_ids = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels_ids
        ]

        inputs["labels"] = labels_ids
        #inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_train = train_subset.map(preprocess, batched=True, remove_columns=train_subset.column_names)
    tokenized_eval  = eval_subset.map(preprocess, batched=True, remove_columns=eval_subset.column_names)

    training_args = get_training_args(
        output_dir=output_dir,
        num_epochs=3,
        batch_size=8,           # reduce to 4 if OOM
        accum_steps=4
    )

    # Create compute_metrics with tokenizer
    compute_metrics_func = make_compute_metrics(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        #tokenizer=tokenizer,
        data_collator= DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics_func
    )
    resume_from_checkpoint="/content/byt5-lora-local/checkpoint-100"
    print("Starting LoRA fine-tuning...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save LoRA adapter
    trainer.save_model(output_dir + "/final_adapter")
    tokenizer.save_pretrained(output_dir + "/final_adapter")

    print(f"LoRA adapter saved to: {output_dir}/final_adapter")
    return trainer