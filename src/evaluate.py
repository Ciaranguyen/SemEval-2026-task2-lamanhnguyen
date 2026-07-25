"""
evaluate.py
-----------
Tính đầy đủ MAE + Pearson r (bản gốc chỉ có MSE), dùng thống nhất cho mọi
baseline lẫn model chính để ra 1 bảng so sánh công bằng.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r, p_value = pearsonr(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "Pearson_r": r, "p_value": p_value}


def evaluate_va(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, subtask: str) -> pd.DataFrame:
    v_metrics = compute_metrics(y_true[:, 0], y_pred[:, 0])
    a_metrics = compute_metrics(y_true[:, 1], y_pred[:, 1])
    row = {
        "subtask": subtask, "model": model_name,
        "valence_MAE": v_metrics["MAE"], "valence_r": v_metrics["Pearson_r"], "valence_p": v_metrics["p_value"],
        "arousal_MAE": a_metrics["MAE"], "arousal_r": a_metrics["Pearson_r"], "arousal_p": a_metrics["p_value"],
    }
    return pd.DataFrame([row])


def build_comparison_table(results: list) -> pd.DataFrame:
    table = pd.concat(results, ignore_index=True)
    table = table.sort_values(["subtask", "valence_MAE"]).reset_index(drop=True)
    return table
