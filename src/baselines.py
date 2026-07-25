"""
baselines.py
------------
Trục 1 — các baseline theo độ phức tạp tăng dần:
  Bậc 1: LexiconBaseline       - không học, tra từ điển NRC-VAD
  Bậc 2: TfidfRidgeBaseline    - có học, không hiểu ngữ cảnh
  Bậc 3: OfficialBertFrozenBaseline - CHÍNH THỨC của ban tổ chức SemEval-2026
         Task 2: Ridge Regression trên embedding BERT-base-uncased ĐÃ ĐÓNG BĂNG
         (frozen, không fine-tune), lấy trung bình cộng token embedding.
         (Xác nhận qua repo UKPLab/semeval26_valence_arousal_from_text — gọi là
         "linear(BERT)"). Đây là baseline CHÍNH THỨC, KHÔNG phải random baseline.

LƯU Ý: thang đo THẬT (xác nhận từ train_subtask1.csv, không phải giả định):
  Valence: 5 mức Likert {-2, -1, 0, 1, 2}
  Arousal: 3 mức Likert {0, 1, 2} — KHÔNG đối xứng quanh 0
"""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


class LexiconBaseline:
    """Bậc 1 — rule-based, dùng từ điển NRC-VAD. NRC-VAD gốc cho điểm [0,1];
    cần calibrate() để đưa về đúng thang thật của task (Valence [-2,2], Arousal [0,2])."""

    def __init__(self, lexicon_path: str):
        self.lexicon = self._load_lexicon(lexicon_path)
        self._calib_v = (1.0, 0.0)
        self._calib_a = (1.0, 0.0)

    @staticmethod
    def _load_lexicon(path: str) -> dict:
        lex = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                word, v, a = parts[0].lower(), float(parts[1]), float(parts[2])
                lex[word] = (v, a)
        return lex

    def calibrate(self, texts, valences, arousals):
        raw = self.predict_raw(texts)
        a_v, b_v = np.polyfit(raw[:, 0], valences, 1)
        a_a, b_a = np.polyfit(raw[:, 1], arousals, 1)
        self._calib_v, self._calib_a = (a_v, b_v), (a_a, b_a)
        return self

    def predict_raw(self, texts) -> np.ndarray:
        return self._score_texts(texts)

    def _score_texts(self, texts) -> np.ndarray:
        preds = []
        for text in texts:
            words = str(text).lower().split()
            scores = [self.lexicon[w] for w in words if w in self.lexicon]
            if scores:
                v = np.mean([s[0] for s in scores])
                a = np.mean([s[1] for s in scores])
            else:
                v, a = 0.5, 0.5
            preds.append([v, a])
        return np.array(preds)

    def predict(self, texts) -> np.ndarray:
        raw = self.predict_raw(texts)
        a_v, b_v = self._calib_v
        a_a, b_a = self._calib_a
        v = a_v * raw[:, 0] + b_v
        a = a_a * raw[:, 1] + b_a
        return np.stack([v, a], axis=1)


class TfidfRidgeBaseline:
    """Bậc 2 — TF-IDF + Ridge Regression. Có học, không hiểu ngữ cảnh/thứ tự từ."""

    def __init__(self, max_features: int = 5000, alpha: float = 1.0):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model_v = Ridge(alpha=alpha)
        self.model_a = Ridge(alpha=alpha)

    def fit(self, texts, valences, arousals):
        X = self.vectorizer.fit_transform(texts)
        self.model_v.fit(X, valences)
        self.model_a.fit(X, arousals)
        return self

    def predict(self, texts) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        v = self.model_v.predict(X)
        a = self.model_a.predict(X)
        return np.stack([v, a], axis=1)


class OfficialBertFrozenBaseline:
    """
    Bậc 3 — Baseline CHÍNH THỨC của ban tổ chức SemEval-2026 Task 2: "linear(BERT)".

    Cơ chế: dùng BERT-base-uncased CHỈ để trích xuất đặc trưng (KHÔNG fine-tune,
    trọng số bị đóng băng — frozen), lấy trung bình cộng (mean pooling) embedding
    của các token, rồi train 1 Ridge Regression đơn giản trên vector đặc trưng cố
    định đó. Đây chính là baseline mà số liệu MAE=1.041 (Valence) / 0.622 (Arousal)
    trong Table 1 của paper gốc tham chiếu tới — KHÔNG phải baseline ngẫu nhiên.
    """

    def __init__(self, backbone_name: str = "bert-base-uncased", alpha: float = 1.0,
                 max_len: int = 128, device: str = None):
        from transformers import AutoTokenizer, AutoModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name).to(self.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False   # đóng băng hoàn toàn, đúng tinh thần "frozen"
        self.max_len = max_len
        self.model_v = Ridge(alpha=alpha)
        self.model_a = Ridge(alpha=alpha)

    @torch.no_grad()
    def _embed(self, texts, batch_size: int = 32) -> np.ndarray:
        """Mean pooling embedding — trung bình cộng vector của MỌI token thật
        (bỏ qua token padding, dùng attention_mask để loại trừ)."""
        all_vecs = []
        texts = [str(t) for t in texts]
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                  truncation=True, max_length=self.max_len).to(self.device)
            out = self.backbone(**enc).last_hidden_state          # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()    # (B, T, 1)
            summed = (out * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = (summed / counts).cpu().numpy()
            all_vecs.append(mean_pooled)
        return np.vstack(all_vecs)

    def fit(self, texts, valences, arousals):
        X = self._embed(texts)
        self.model_v.fit(X, valences)
        self.model_a.fit(X, arousals)
        return self

    def predict(self, texts) -> np.ndarray:
        X = self._embed(texts)
        v = self.model_v.predict(X)
        a = self.model_a.predict(X)
        return np.stack([v, a], axis=1)
