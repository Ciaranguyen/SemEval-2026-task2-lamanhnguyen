"""
data.py
-------
Toàn bộ logic liên quan đến DỮ LIỆU — ĐÃ CẬP NHẬT theo cấu trúc cột THẬT,
xác nhận bằng cách đọc trực tiếp train_subtask1.csv / train_subtask2a.csv:

Cột thật có sẵn: user_id, text_id, text, timestamp, collection_phase,
                 is_words, valence, arousal
(Subtask 2a có thêm: state_change_valence, state_change_arousal)

LƯU Ý QUAN TRỌNG: cột `is_words` phân biệt 2 loại input mà Figure 1 của BTC
gọi là "essays or feeling words":
  - is_words = False -> đoạn văn đầy đủ (essay), trung bình ~59 từ, tối đa 225 từ
  - is_words = True  -> chỉ là danh sách từ cảm xúc rời rạc, vd "Tired, Calm, Happy"
Mặc định pipeline dùng CẢ HAI loại (giống cách paper gốc xử lý), nhưng có thể
lọc riêng qua tham số `only_essays` nếu muốn thực nghiệm so sánh.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def load_and_merge_train_data(raw_dir: str, train_files: list, cols: dict,
                               only_essays: bool = False) -> pd.DataFrame:
    """Gộp dữ liệu train từ nhiều subtask, chỉ giữ 3 cột cần thiết: text, valence, arousal."""
    frames = []
    text_col, v_col, a_col, iw_col = cols["text"], cols["valence"], cols["arousal"], cols["is_words"]

    for fname in train_files:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            print(f"⚠️  Không tìm thấy {fpath}, bỏ qua.")
            continue
        df = pd.read_csv(fpath)
        required = {text_col, v_col, a_col}
        if not required.issubset(df.columns):
            print(f"⚠️  {fname} thiếu cột {required - set(df.columns)}, bỏ qua.")
            continue

        sub = df[[text_col, v_col, a_col] + ([iw_col] if iw_col in df.columns else [])].copy()
        sub.columns = ["text", "valence", "arousal"] + (["is_words"] if iw_col in df.columns else [])
        frames.append(sub)
        print(f"   -> Đã thêm {len(sub)} dòng từ {fname}")

    if not frames:
        raise FileNotFoundError(
            "Không tìm thấy dữ liệu train nào trong raw_dir. Hãy đặt các file .csv vào data/raw/."
        )

    full_df = pd.concat(frames, ignore_index=True)
    full_df = clean_dataframe(full_df)

    if only_essays and "is_words" in full_df.columns:
        before = len(full_df)
        full_df = full_df[full_df["is_words"] == False].reset_index(drop=True)  # noqa: E712
        print(f"   -> Lọc chỉ giữ essay (is_words=False): {before} -> {len(full_df)} dòng")

    print(f"✅ TỔNG CỘNG: {len(full_df)} dòng dữ liệu train sạch.")
    return full_df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text", "valence", "arousal"]).copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]
    return df.reset_index(drop=True)


def split_train_val(df: pd.DataFrame, val_split: float, seed: int):
    return train_test_split(df, test_size=val_split, random_state=seed)


class AffectDataset(Dataset):
    """PyTorch Dataset cho bài toán regression 2 đầu ra (valence, arousal)."""

    def __init__(self, texts, valences, arousals, tokenizer, max_len: int):
        self.texts = list(texts)
        self.valences = list(valences)
        self.arousals = list(arousals)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "targets": torch.tensor([self.valences[idx], self.arousals[idx]], dtype=torch.float),
        }
