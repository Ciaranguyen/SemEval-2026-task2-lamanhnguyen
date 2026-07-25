"""
check_token_length.py
----------------------
Kiểm tra phân phối ĐỘ DÀI TOKEN THẬT (sau khi qua tokenizer DeBERTa-v3) trên
toàn bộ dữ liệu train, để có SỐ LIỆU THẬT bảo vệ lựa chọn max_len=128 trong
báo cáo (mục 4.3 - Phân tích độ phức tạp), thay vì chỉ suy luận từ số từ.

LƯU Ý: script này cần tải tokenizer từ HuggingFace nên phải chạy trên máy có
mạng internet đầy đủ (Colab) — môi trường sandbox lúc soạn code này không có
quyền truy cập huggingface.co nên KHÔNG thể chạy thử tại đây.

Cách chạy trên Colab:
    python -m src.check_token_length
"""

import pandas as pd
from transformers import AutoTokenizer

from src.utils import load_config


def check_token_length(cfg: dict):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    cols = cfg["data"]["columns"]

    all_lengths = []
    for fname in cfg["data"]["train_files"]:
        path = f"{cfg['data']['raw_dir']}/{fname}"
        df = pd.read_csv(path)
        texts = df[cols["text"]].dropna().astype(str).tolist()
        lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in texts]
        all_lengths.extend(lengths)
        print(f"{fname}: {len(texts)} văn bản, độ dài token trung bình = {sum(lengths)/len(lengths):.1f}")

    s = pd.Series(all_lengths)
    print("\n=== PHÂN PHỐI ĐỘ DÀI TOKEN TOÀN BỘ TẬP TRAIN ===")
    print(s.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    for max_len in [64, 128, 192, 256]:
        pct_truncated = (s > max_len).mean() * 100
        print(f"max_len={max_len}: {pct_truncated:.2f}% văn bản bị CẮT CỤT (vượt quá giới hạn)")


if __name__ == "__main__":
    config = load_config()
    check_token_length(config)
