"""
models.py
---------
Kiến trúc mô hình — hỗ trợ CẢ multi-task (2 head, dùng cho V1/V3/Ensemble)
LẪN single-task (1 head, dùng cho Trục 1 so sánh: BERT-single-task,
DeBERTa-single-task) để cô lập chính xác hiệu quả của multi-task learning,
tách bạch khỏi hiệu quả riêng của việc đổi backbone BERT -> DeBERTa.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class AffectModel(nn.Module):
    """
    Backbone Transformer (BERT hoặc DeBERTa, cấu hình qua model_name) + head
    tuyến tính. Nếu multitask=True: 2 head riêng (Valence, Arousal), chia sẻ
    chung 1 backbone. Nếu multitask=False: chỉ 1 head, dùng target_dim để chọn
    dự đoán Valence (0) hay Arousal (1) khi train.
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 dropout: float = 0.1, multitask: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.multitask = multitask
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        if multitask:
            self.v_head = nn.Linear(hidden_size, 1)
            self.a_head = nn.Linear(hidden_size, 1)
        else:
            # single-task: chỉ 1 head duy nhất, ra 1 giá trị
            self.single_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_embedding)

        if self.multitask:
            v = self.v_head(x)
            a = self.a_head(x)
            return torch.cat((v, a), dim=-1)   # shape (batch, 2)
        else:
            return self.single_head(x)          # shape (batch, 1)

    @torch.no_grad()
    def predict_texts(self, texts, tokenizer, device, max_len=128, batch_size=16):
        self.eval()
        all_preds = []
        clean_texts = [str(t) if t == t else "" for t in texts]
        for i in range(0, len(clean_texts), batch_size):
            batch = clean_texts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=max_len).to(device)
            out = self(enc["input_ids"], enc["attention_mask"])
            all_preds.append(out.cpu().numpy())
        import numpy as np
        return np.vstack(all_preds) if all_preds else np.zeros((0, 2 if self.multitask else 1))


class UncertaintyWeightedLoss(nn.Module):
    """
    Trục 2 — chiến lược multi-task loss NÂNG CAO, thay thế cách cộng trọng số
    cố định lambda_v * Loss_V + lambda_a * Loss_A.

    Ý tưởng (Kendall, Gal & Cipolla, 2018 — "Multi-Task Learning Using
    Uncertainty to Weigh Losses"): thay vì tự chọn lambda cố định, để mô hình
    TỰ HỌC 1 tham số log(sigma^2) cho mỗi nhiệm vụ, đại diện cho độ bất định
    (uncertainty) của nhiệm vụ đó. Loss cuối cùng:

        L = (1 / (2*sigma_v^2)) * Loss_V + log(sigma_v)
          + (1 / (2*sigma_a^2)) * Loss_A + log(sigma_a)

    Nhiệm vụ nào có độ bất định cao (khó học, nhiễu nhiều) sẽ tự động được
    giảm trọng số, tránh việc 1 nhiệm vụ "lấn át" nhiệm vụ kia trong lúc học
    chung — đây chính là hướng "học 2 objective hỗ trợ nhau" thay vì chỉ đơn
    thuần cộng trọng số cố định.
    """

    def __init__(self):
        super().__init__()
        # log_sigma^2 khởi tạo = 0 (tức sigma^2 = 1, tương đương trọng số ban đầu bằng nhau)
        self.log_sigma_v = nn.Parameter(torch.zeros(1))
        self.log_sigma_a = nn.Parameter(torch.zeros(1))

    def forward(self, loss_v: torch.Tensor, loss_a: torch.Tensor) -> torch.Tensor:
        precision_v = torch.exp(-self.log_sigma_v)
        precision_a = torch.exp(-self.log_sigma_a)
        total = precision_v * loss_v + self.log_sigma_v \
              + precision_a * loss_a + self.log_sigma_a
        return total.squeeze()

    def get_learned_weights(self) -> dict:
        """Trả về trọng số hiệu dụng hiện tại của mỗi nhiệm vụ, để log/phân tích."""
        with torch.no_grad():
            return {
                "weight_valence": torch.exp(-self.log_sigma_v).item(),
                "weight_arousal": torch.exp(-self.log_sigma_a).item(),
            }
