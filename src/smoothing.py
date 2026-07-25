"""
smoothing.py
------------
Cung cấp 2 cách "làm mượt" để so sánh:
  1. constant_scale() - cách code gốc trên GitHub thực sự làm (chỉ nhân hệ số
     cố định, KHÔNG đệ quy theo thời gian)
  2. true_ema() - đúng công thức (1) trong paper: P_t = alpha*y_t + (1-alpha)*P_{t-1}
"""

import numpy as np
import pandas as pd


def constant_scale(predictions: np.ndarray, factor: float) -> np.ndarray:
    return predictions * factor


def true_ema(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for t in range(1, len(values)):
        smoothed[t] = alpha * values[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def apply_true_ema_per_user(df: pd.DataFrame, value_col: str, user_col: str = "user_id",
                             time_col: str = "timestamp", alpha: float = 0.1) -> pd.Series:
    df_sorted = df.sort_values([user_col, time_col])
    result = df_sorted.groupby(user_col)[value_col].transform(
        lambda s: true_ema(s.values, alpha=alpha)
    )
    return result.reindex(df.index)
