
import pandas as pd
from transformers import AutoTokenizer

from src.utils import load_config


def main():
    cfg = load_config("configs/config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    train = pd.read_csv(f"{cfg['data']['raw_dir']}/train_subtask1.csv")

    lengths = train["text"].apply(lambda t: len(tokenizer.encode(str(t))))

    print(lengths.describe())
    max_len = cfg["model"]["max_len"]
    pct_truncated = (lengths > max_len).mean() * 100
    print(f"\n% entries vuot qua max_len={max_len}: {pct_truncated:.2f}%")
    print(f"Token length tai percentile 95: {lengths.quantile(0.95):.0f}")
    print(f"Token length tai percentile 99: {lengths.quantile(0.99):.0f}")

    return lengths


if __name__ == "__main__":
    main()
