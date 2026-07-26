
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from src.models import AffectModel
from src.utils import load_config, get_device

CONFIGS = [
    ("DeBERTa single-task", "microsoft/deberta-v3-base",
     "results/checkpoints/deberta_singletask_VALENCE.pth", "valence"),
    ("DeBERTa single-task", "microsoft/deberta-v3-base",
     "results/checkpoints/deberta_singletask_AROUSAL.pth", "arousal"),
    ("BERT single-task", "bert-base-uncased",
     "results/checkpoints/bert_singletask_VALENCE.pth", "valence"),
    ("BERT single-task", "bert-base-uncased",
     "results/checkpoints/bert_singletask_AROUSAL.pth", "arousal"),
]


@torch.no_grad()
def predict_single(model, texts, tokenizer, device, max_len=128, batch_size=32):
    model.eval()
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_len).to(device)
        out = model(enc["input_ids"], enc["attention_mask"])
        all_preds.append(out.cpu().numpy())
    return np.concatenate(all_preds).squeeze(-1)


def main():
    cfg = load_config("configs/config.yaml")
    device = get_device()
    test = pd.read_csv(f"{cfg['data']['raw_dir']}/test_labels_subtask1.csv")

    rows = []
    for name, backbone, path, target in CONFIGS:
        tok = AutoTokenizer.from_pretrained(backbone)
        model = AffectModel(model_name=backbone, multitask=False).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        preds = predict_single(model, test["text"].tolist(), tok, device)
        y_true = test[target].values
        mae = mean_absolute_error(y_true, preds)
        r, p = pearsonr(y_true, preds)
        rows.append({"model": name, "target": target, "MAE": mae, "r": r, "p": p})
        print(f"{name} ({target}): MAE={mae:.4f}  r={r:.4f}")
        del model
        torch.cuda.empty_cache()

    result_df = pd.DataFrame(rows)
    print("\n KET QUA SINGLE-TASK")
    print(result_df.to_string(index=False))
    return result_df


if __name__ == "__main__":
    main()
