"""
train.py
--------
Script train dùng chung cho MỌI biến thể trong Trục 1 (v1, v3, bert_singletask,
deberta_singletask) và MỌI chiến lược loss trong Trục 2 (fixed_weight,
uncertainty_weighting) — chỉ khác nhau qua config, không lặp code.

Cách chạy:
    python -m src.train --variant v1
    python -m src.train --variant deberta_singletask --target valence
    python -m src.train --variant v1 --loss_strategy uncertainty_weighting
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data import load_and_merge_train_data, split_train_val, AffectDataset
from src.models import AffectModel, UncertaintyWeightedLoss
from src.utils import load_config, set_seed, get_device


def train_one_model(cfg: dict, variant: str, target: str = None,
                     loss_strategy: str = None) -> str:
    """
    variant: tên trong cfg['training'] (v1, v3, bert_singletask, deberta_singletask)
    target: chỉ dùng khi model là single-task -> 'valence' hoặc 'arousal'
    loss_strategy: 'fixed_weight' hoặc 'uncertainty_weighting' (chỉ áp dụng cho multi-task)
    """
    set_seed(cfg["data"]["seed"])
    device = get_device()
    train_cfg = cfg["training"][variant]
    multitask = train_cfg.get("multitask", True)
    backbone_name = train_cfg.get("backbone", cfg["model"]["name"])
    loss_strategy = loss_strategy or cfg["multitask_loss"]["strategy"]

    print(f"⚙️  Thiết bị: {device} | Biến thể: {variant} | Backbone: {backbone_name} "
          f"| Multi-task: {multitask} | Loss strategy: {loss_strategy if multitask else 'N/A (single-task)'}")

    os.makedirs(os.path.dirname(train_cfg["save_path"]), exist_ok=True)

    # ---- Data ----
    df = load_and_merge_train_data(cfg["data"]["raw_dir"], cfg["data"]["train_files"], cfg["data"]["columns"])
    df_train, df_val = split_train_val(df, cfg["data"]["val_split"], cfg["data"]["seed"])

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    train_ds = AffectDataset(df_train["text"].values, df_train["valence"].values,
                              df_train["arousal"].values, tokenizer, cfg["model"]["max_len"])
    val_ds = AffectDataset(df_val["text"].values, df_val["valence"].values,
                            df_val["arousal"].values, tokenizer, cfg["model"]["max_len"])
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"])

    # ---- Model ----
    dropout = train_cfg.get("dropout", cfg["model"]["dropout"])
    model = AffectModel(backbone_name, dropout=dropout, multitask=multitask).to(device)

    criterion = nn.MSELoss()
    uw_loss_fn = None
    params = list(model.parameters())
    if multitask and loss_strategy == "uncertainty_weighting":
        uw_loss_fn = UncertaintyWeightedLoss().to(device)
        params += list(uw_loss_fn.parameters())

    optimizer = AdamW(params, lr=train_cfg["lr"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * train_cfg["epochs"]
    )

    target_idx = {"valence": 0, "arousal": 1}.get(target) if target else None
    if not multitask and target_idx is None:
        raise ValueError("Model single-task cần chỉ rõ --target valence hoặc --target arousal")

    best_val_loss = float("inf")
    for epoch in range(train_cfg["epochs"]):
        model.train()
        loop = tqdm(train_loader, desc=f"[{variant}] Epoch {epoch+1}/{train_cfg['epochs']}")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attn)

            if multitask:
                loss_v = criterion(outputs[:, 0], targets[:, 0])
                loss_a = criterion(outputs[:, 1], targets[:, 1])
                if loss_strategy == "uncertainty_weighting":
                    loss = uw_loss_fn(loss_v, loss_a)
                else:  # fixed_weight
                    lam_v, lam_a = cfg["multitask_loss"]["lambda_v"], cfg["multitask_loss"]["lambda_a"]
                    loss = lam_v * loss_v + lam_a * loss_a
            else:
                loss = criterion(outputs.squeeze(-1), targets[:, target_idx])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loop.set_postfix(loss=loss.item())

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)
                outputs = model(input_ids, attn)
                if multitask:
                    vl = criterion(outputs[:, 0], targets[:, 0]) + criterion(outputs[:, 1], targets[:, 1])
                else:
                    vl = criterion(outputs.squeeze(-1), targets[:, target_idx])
                val_loss += vl.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"   📉 Val loss: {avg_val_loss:.4f}")
        if multitask and loss_strategy == "uncertainty_weighting":
            print(f"   ⚖️  Trọng số học được: {uw_loss_fn.get_learned_weights()}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), train_cfg["save_path"])
            print(f"   💾 Đã lưu checkpoint tốt nhất -> {train_cfg['save_path']}")

    print(f"🎉 Xong {variant}. Best Val loss = {best_val_loss:.4f}")
    return train_cfg["save_path"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                         choices=["v1", "v3", "bert_singletask", "deberta_singletask"])
    parser.add_argument("--target", choices=["valence", "arousal"], default=None)
    parser.add_argument("--loss_strategy", choices=["fixed_weight", "uncertainty_weighting"], default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train_one_model(config, args.variant, args.target, args.loss_strategy)
