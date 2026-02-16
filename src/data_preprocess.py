# src/data_preprocess.py

import pandas as pd
from pathlib import Path
from datasets import Dataset

def load_and_clean_language_data(lang: str, base_path: str = "data"):
    """Load single language train + valid json, clean columns"""
    base = Path(base_path) / lang
    
    df_train = pd.read_json(base / f"{lang}_train.json", lines=True)
    df_valid = pd.read_json(base / f"{lang}_valid.json", lines=True)
    
    # rename & drop unnecessary columns
    rename_map = {
        "english word": "roman",
        "native word":  "native",
    }
    for df in [df_train, df_valid]:
        df.rename(columns=rename_map, inplace=True)
        df.drop(columns=["unique_identifier", "source", "score"], errors="ignore", inplace=True)
        df["lang"] = lang
    
    return df_train, df_valid


def prepare_multilingual_data(languages=["hin", "tam", "tel"], base_path="data"):
    """Combine all languages, add <lang> prefix"""
    dfs_train, dfs_valid = [], []
    
    for lang in languages:
        df_t, df_v = load_and_clean_language_data(lang, base_path)
        # add prefix
        df_t["input"] = "<" + lang + "> " + df_t["roman"]
        df_v["input"] = "<" + lang + "> " + df_v["roman"]
        dfs_train.append(df_t)
        dfs_valid.append(df_v)
    
    df_train_all = pd.concat(dfs_train, ignore_index=True)
    df_valid_all = pd.concat(dfs_valid, ignore_index=True)
    
    return df_train_all, df_valid_all


def create_hf_datasets(df_train: pd.DataFrame, df_valid: pd.DataFrame):
    """Convert pandas → HF Dataset with text/label columns"""
    train_ds = Dataset.from_pandas(
        df_train[["input", "native"]].rename(columns={"input": "text", "native": "label"})
    )
    valid_ds = Dataset.from_pandas(
        df_valid[["input", "native"]].rename(columns={"input": "text", "native": "label"})
    )
    return train_ds, valid_ds