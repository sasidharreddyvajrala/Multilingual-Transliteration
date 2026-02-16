# Multilingual Indic Transliteration Model

**Task**: Fine-tune and deploy a multilingual transliteration model (Roman → Native script) for Hindi, Tamil, and Telugu using the Aksharantar dataset.

**Author**: Vajrala  
**Date**: February 2026

## Project Overview

The goal was to build a multilingual transliteration model supporting three Indic languages (Hindi, Tamil, Telugu) from Roman script to native script, using the Aksharantar dataset from Hugging Face.

Due to significant hardware limitations (local CPU + intermittent Colab free T4 GPU) and tight timeline (2–3 days), we focused on:

- End-to-end pipeline (data → training → optimization → deployment)
- Realistic training with LoRA on small subsets
- Multiple attempts with mT5-small and ByT5-small
- Evaluation using CER (primary metric) and exact match
- Final optimization with CTranslate2
- Interactive Gradio demo on Hugging Face Spaces

## 1. Dataset & Language Selection

**Dataset**: [ai4bharat/Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar)  
**Languages chosen**: Hindi (`hin`), Tamil (`tam`), Telugu (`tel`)

**Preprocessing steps**:
- Loaded per-language JSON files (`*_train.json`, `*_valid.json`)
- Kept only relevant columns: `english word` → `roman`, `native word` → `native`
- Dropped: `unique_identifier`, `source`, `score`
- Added language prefix: `<hin> `, `<tam> `, `<tel> `
- Created Hugging Face `Dataset` objects with columns: `text` (input), `label` (target)

**Final splits**:
- Train: ~1–2M examples (used subsets of 20k–100k for training)
- Validation: ~500 examples (stratified random subset for evaluation)

## 2. Model Training

**Approach chosen**: Option B — Pretrained sequence-to-sequence model with LoRA fine-tuning

**Models attempted**:
- google/byt5-small (byte-level)
- google/mt5-small (subword-level)

**Architecture**:
- Encoder-decoder transformer (mT5/ByT5 base)
- LoRA adapters (r=8, alpha=16, target_modules=["q", "v"])
- Trainable parameters: ~0.2% of total (~594k out of 301M)

**Training parameters (final stable run with mT5-small)**:
- Subset size: 20,000–100,000 examples
- Epochs: 1
- Learning rate: 1e-6 to 5e-5 (lower to avoid NaN)
- Batch size: 8 per device + gradient accumulation 4–8
- Optimizer: AdamW
- Warmup steps: 500–1000
- Gradient clipping: 0.5 norm
- fp16: False (stability)
- predict_with_generate: True
- Evaluation: manual after training (CER only)

**Challenges faced & solutions**:
- NaN loss → lowered LR to 1e-6, disabled fp16, added gradient clipping
- Slow / hanging evaluation → disabled predict_with_generate during training
- Decoding errors (NaN / invalid IDs) → added np.nan_to_num + clipping + try-except fallback
- Limited compute → used small subsets + LoRA

**Evaluation metrics (zero-shot baseline on mT5-small)**:
- CER: ~2.06
- Exact match: 0.0%

**After partial fine-tuning** (results varied across runs due to instability):
- CER: typically 1.0–1.8 (small improvement in some runs)
- Qualitative: fewer special tokens, partial correct characters, still noisy

(Note: Full fine-tuning was not possible due to repeated NaN / hanging issues and time constraints.)

## 3. Model Optimization

**Technique**: CTranslate2 (quantization to int8)

**Before**:
- Model size: ~1.2 GB (mT5-small)
- Inference speed: slow on CPU (~1–3 s/word)

**After CTranslate2**:
- Model size: ~303 MB (int8 quantized)
- Speed gain: ~3–8× faster inference
- Quality loss: minimal (CER drop <0.05 in tests)

**Hugging Face model weight**:
-huggingface.co/SasidharVajrala/mt5-small-ctranslate2

Conversion command:
```bash
ct2-transformers-converter --model ./merged_model --output_dir ./ctranslate2_model --quantization int8 --forc


