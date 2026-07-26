
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from scipy.stats import pearsonr, mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from src.models import AffectModel
from src.utils import load_config, get_device


def bootstrap_ci_mae_diff(seen_errors: np.ndarray, unseen_errors: np.ndarray,
                           n_boot: int = 2000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        s_sample = rng.choice(seen_errors, size=len(seen_errors), replace=True)
        u_sample = rng.choice(unseen_errors, size=len(unseen_errors), replace=True)
        diffs.append(u_sample.mean() - s_sample.mean())
    diffs = np.array(diffs)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def report_group(group: pd.DataFrame, name: str, target: str = "valence") -> dict:
    mae = group[f"abs_err_{target[0]}"].mean()
    r, _ = pearsonr(group[target], group[f"pred_{target[0]}"])
    print(f"{name} (n={len(group)}): MAE={mae:.4f}  r={r:.4f}")
    return {"name": name, "n": len(group), "mae": mae, "r": r}


def analyze_tfidf_ridge(train: pd.DataFrame, test: pd.DataFrame, cfg: dict):
    print("\n TF-IDF + Ridge: Seen vs Unseen")
    vectorizer = TfidfVectorizer(max_features=cfg["baselines"]["tfidf_max_features"])
    X_train = vectorizer.fit_transform(train["text"].astype(str))
    X_test = vectorizer.transform(test["text"].astype(str))

    alpha = cfg["baselines"]["ridge_alpha"]
    model_v = Ridge(alpha=alpha).fit(X_train, train["valence"])
    model_a = Ridge(alpha=alpha).fit(X_train, train["arousal"])

    test = test.copy()
    test["pred_v"] = model_v.predict(X_test)
    test["pred_a"] = model_a.predict(X_test)
    test["abs_err_v"] = np.abs(test["valence"] - test["pred_v"])
    test["abs_err_a"] = np.abs(test["arousal"] - test["pred_a"])

    seen = test[test["is_seen_user"] == True]
    unseen = test[test["is_seen_user"] == False]

    r_seen = report_group(seen, "Seen", "valence")
    r_unseen = report_group(unseen, "Unseen", "valence")

    ci_low, ci_high = bootstrap_ci_mae_diff(seen["abs_err_v"].values, unseen["abs_err_v"].values)
    stat, p_mw = mannwhitneyu(seen["abs_err_v"], unseen["abs_err_v"])
    print(f"Bootstrap 95% CI (Unseen-Seen) MAE_Valence: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Mann-Whitney U p-value: {p_mw:.4f}")
    return {"seen": r_seen, "unseen": r_unseen, "ci": (ci_low, ci_high), "p": p_mw}


@torch.no_grad()
def analyze_deberta(test: pd.DataFrame, checkpoint_path: str, device):
    print("\n===== DeBERTa multi-task: Seen vs Unseen =====")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AffectModel(model_name="microsoft/deberta-v3-base", multitask=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    preds = np.array(model.predict_texts(test["text"].tolist(), tokenizer, device,
                                          max_len=128, batch_size=32))
    test = test.copy()
    test["pred_v"] = preds[:, 0]
    test["pred_a"] = preds[:, 1]
    test["abs_err_v"] = np.abs(test["valence"] - test["pred_v"])
    test["abs_err_a"] = np.abs(test["arousal"] - test["pred_a"])

    seen = test[test["is_seen_user"] == True]
    unseen = test[test["is_seen_user"] == False]

    r_seen = report_group(seen, "Seen", "valence")
    r_unseen = report_group(unseen, "Unseen", "valence")

    ci_low, ci_high = bootstrap_ci_mae_diff(seen["abs_err_v"].values, unseen["abs_err_v"].values)
    stat, p_mw = mannwhitneyu(seen["abs_err_v"], unseen["abs_err_v"])
    print(f"Bootstrap 95% CI (Unseen-Seen) MAE_Valence: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Mann-Whitney U p-value: {p_mw:.4f}")
    return {"seen": r_seen, "unseen": r_unseen, "ci": (ci_low, ci_high), "p": p_mw}


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    device = get_device()

    train = pd.read_csv(f"{cfg['data']['raw_dir']}/train_subtask1.csv")
    test = pd.read_csv(f"{cfg['data']['raw_dir']}/test_labels_subtask1.csv")

    tfidf_results = analyze_tfidf_ridge(train, test, cfg)

    deberta_ckpt = "results/checkpoints/deberta_affect_v1_seed42_wd0.pth"
    deberta_results = analyze_deberta(test, deberta_ckpt, device)

    print("\n TOM TAT")
    print(f"TF-IDF+Ridge : Seen MAE={tfidf_results['seen']['mae']:.4f}  "
          f"Unseen MAE={tfidf_results['unseen']['mae']:.4f}  p={tfidf_results['p']:.4f}")
    print(f"DeBERTa MTL  : Seen MAE={deberta_results['seen']['mae']:.4f}  "
          f"Unseen MAE={deberta_results['unseen']['mae']:.4f}  p={deberta_results['p']:.4f}")
