
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from src.models import AffectModel
from src.smoothing import true_ema, constant_scale
from src.utils import load_config, get_device


def compute_deltas(group_vals: np.ndarray) -> tuple:
    if len(group_vals) < 2:
        return np.nan, np.nan
    state_change = group_vals[-1] - group_vals[-2]
    mid = len(group_vals) // 2
    if mid == 0:
        disp_change = np.nan
    else:
        first_half = group_vals[:mid].mean()
        second_half = group_vals[mid:].mean()
        disp_change = second_half - first_half
    return state_change, disp_change


def evaluate_config(df: pd.DataFrame, col: str, alpha: float = 0.1,
                     scale: float = 0.1, mode: str = "none") -> pd.DataFrame:
    rows = []
    for uid, g in df.groupby("user_id"):
        vals = g[col].values
        if mode == "true_ema":
            vals = true_ema(vals, alpha=alpha)
        elif mode == "constant_scale":
            vals = constant_scale(vals, factor=scale)
        sc, dc = compute_deltas(vals)
        rows.append({"user_id": uid, "state_change": sc, "disp_change": dc})
    return pd.DataFrame(rows).set_index("user_id")


def main():
    cfg = load_config("configs/config.yaml")
    device = get_device()
    raw_dir = cfg["data"]["raw_dir"]

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AffectModel(model_name="microsoft/deberta-v3-base", multitask=True)
    model.load_state_dict(torch.load(
        "results/checkpoints/deberta_affect_v1_seed42_wd0.pth", map_location=device))
    model.to(device).eval()

    text_source = pd.read_csv(f"{raw_dir}/train_subtask2a.csv")
    subtask2_users = pd.read_csv(f"{raw_dir}/test_subtask2.csv")["user_id"].unique()
    labels2 = pd.read_csv(f"{raw_dir}/test_labels_subtask2a_and_2b.csv").set_index("user_id")

    df = text_source[text_source["user_id"].isin(subtask2_users)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    print(f"So dong sau khi loc theo 46 user Subtask 2: {len(df)}")

    preds = np.array(model.predict_texts(df["text"].tolist(), tokenizer, device,
                                          max_len=128, batch_size=32))
    df["raw_v"] = preds[:, 0]
    df["raw_a"] = preds[:, 1]

    results_summary = []
    for target, col in [("Valence", "raw_v"), ("Arousal", "raw_a")]:
        for mode_name, mode in [("No filter", "none"),
                                 ("constant_scale (code goc)", "constant_scale"),
                                 ("true_ema (dung paper)", "true_ema")]:
            pred_df = evaluate_config(df, col, alpha=0.1, scale=0.1, mode=mode)
            merged = pred_df.join(labels2, how="inner")

            y_true_sc = merged[f"state_change_{target.lower()}"]
            y_pred_sc = merged["state_change"]
            y_true_dc = merged[f"disp_change_{target.lower()}"]
            y_pred_dc = merged["disp_change"]

            mae_2a = mean_absolute_error(y_true_sc, y_pred_sc)
            r_2a, _ = pearsonr(y_true_sc, y_pred_sc)
            mae_2b = mean_absolute_error(y_true_dc, y_pred_dc)
            r_2b, _ = pearsonr(y_true_dc, y_pred_dc)

            results_summary.append({
                "target": target, "config": mode_name,
                "MAE_2a": mae_2a, "r_2a": r_2a,
                "MAE_2b": mae_2b, "r_2b": r_2b,
            })

    result_table = pd.DataFrame(results_summary)
    print("\n KET QUA ABLATION SMOOTHING")
    print(result_table.to_string(index=False))
    print("\nLUU Y: ket qua nay dung text da thay trong train (train_subtask2a.csv),")
    print("KHONG phai test hoan toan an - chi co gia tri so sanh NOI BO giua 3 cau hinh.")
    return result_table


if __name__ == "__main__":
    main()
