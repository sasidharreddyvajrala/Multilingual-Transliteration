# src/evaluation.py

import evaluate
import numpy as np

cer_metric = evaluate.load("cer")
# wer_metric = evaluate.load("wer")   # optional

def compute_metrics(predictions: list, references: list):
    cer = cer_metric.compute(predictions=predictions, references=references)
    exact_match = np.mean([p == r for p, r in zip(predictions, references)])
    
    return {
        "cer": cer,
        "exact_match": exact_match,
        # "wer": wer_metric.compute(predictions=predictions, references=references)
    }