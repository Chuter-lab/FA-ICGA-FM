"""Evaluation suite for FA-ICGA-FM.

Implements:
  E1 - Linear probe evaluation
  E4 - Few-shot (5/10/25-shot) evaluation
  E5 - Attention map / GradCAM explainability
  E6 - IQA head evaluation
  G2 - CAL (Connectivity-Area-Length) metric for vessel segmentation
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              cohen_kappa_score, confusion_matrix)
from sklearn.preprocessing import label_binarize


def extract_features(backbone, loader, device, n_max=None):
    """Extract feature vectors from a backbone model."""
    feats, labels = [], []
    backbone.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if n_max and i * loader.batch_size >= n_max:
                break
            imgs = batch[0].to(device)
            lab  = batch[1]
            if hasattr(backbone, "features"):
                f = backbone.features(imgs)
            elif hasattr(backbone, "encode_full"):
                f = backbone.encode_full(imgs)
            else:
                f = backbone(imgs)
            if isinstance(f, (tuple, list)):
                f = f[0]
            if f.dim() > 2:
                f = f[:, 0]
            feats.append(f.cpu().float())
            labels.append(lab)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def linear_probe_eval(train_feats, train_labels, test_feats, test_labels, n_classes):
    """Fit logistic regression on frozen features, compute AUC-ROC and accuracy.

    Returns dict with auc, acc, f1, kappa.
    """
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(train_feats, train_labels)
    preds = clf.predict(test_feats)
    proba = clf.predict_proba(test_feats)

    acc = float(accuracy_score(test_labels, preds))
    kappa = float(cohen_kappa_score(test_labels, preds))
    f1 = float(f1_score(test_labels, preds, average="macro", zero_division=0))

    try:
        if n_classes == 2:
            auc = float(roc_auc_score(test_labels, proba[:, 1]))
        else:
            y_bin = label_binarize(test_labels, classes=list(range(n_classes)))
            auc = float(roc_auc_score(y_bin, proba, multi_class="ovr",
                                      average="macro"))
    except Exception:
        auc = float("nan")

    return {"auc": auc, "acc": acc, "f1": f1, "kappa": kappa}


def few_shot_eval(backbone, dataset_fn, n_shots_list, device, n_seeds=10,
                  n_classes=24):
    """5/10/25-shot nearest-neighbor + linear probe evaluation (E4).

    dataset_fn(n) -> (n labeled examples per class) train set.
    Returns dict: {shot: {metric: mean±std}}.
    """
    results = {}
    for n_shots in n_shots_list:
        accs = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            support_feats, support_labels = [], []
            query_feats,   query_labels   = [], []

            train_feats = dataset_fn("train_feats")
            train_labels = dataset_fn("train_labels")
            test_feats  = dataset_fn("test_feats")
            test_labels = dataset_fn("test_labels")

            for cls in range(n_classes):
                idx = np.where(train_labels == cls)[0]
                if len(idx) == 0:
                    continue
                chosen = rng.choice(idx, size=min(n_shots, len(idx)), replace=False)
                support_feats.append(train_feats[chosen])
                support_labels.extend([cls] * len(chosen))

            if len(support_feats) == 0:
                continue
            sf = np.vstack(support_feats)
            sl = np.array(support_labels)
            clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
            try:
                clf.fit(sf, sl)
                preds = clf.predict(test_feats)
                accs.append(float(accuracy_score(test_labels, preds)))
            except Exception:
                pass

        if accs:
            results[n_shots] = {"mean": float(np.mean(accs)),
                                 "std":  float(np.std(accs)),
                                 "n_seeds": len(accs)}
        else:
            results[n_shots] = {"mean": float("nan"), "std": float("nan"),
                                 "n_seeds": 0}
    return results


def compute_dice(pred_mask, gt_mask, eps=1e-6):
    """Compute Dice coefficient between binary masks."""
    pred = (pred_mask > 0.5).float()
    gt   = gt_mask.float()
    inter = (pred * gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def compute_cal(pred_mask, gt_mask):
    """Topology-aware CAL metric (Connectivity, Area, Length) for vessel seg (G2).

    Returns dict with connectivity, area, length, CAL (mean).
    Reference: RECOVERY-FA19 paper (PMC7648732).
    """
    try:
        from skimage.measure import label as sk_label
        from skimage.morphology import skeletonize
    except ImportError:
        return {"connectivity": float("nan"), "area": float("nan"),
                "length": float("nan"), "CAL": float("nan")}

    pred_b = (pred_mask > 0.5).astype(bool)
    gt_b   = gt_mask.astype(bool)

    def _connectivity(mask):
        if not mask.any():
            return 0
        return int(sk_label(mask).max())

    pred_conn = _connectivity(pred_b)
    gt_conn   = _connectivity(gt_b)
    conn_score = 1.0 - abs(pred_conn - gt_conn) / max(gt_conn, 1)

    area_pred = float(pred_b.sum())
    area_gt   = float(gt_b.sum())
    area_score = 1.0 - abs(area_pred - area_gt) / max(area_gt, 1)

    skel_pred = skeletonize(pred_b).sum()
    skel_gt   = skeletonize(gt_b).sum()
    len_score = 1.0 - abs(skel_pred - skel_gt) / max(skel_gt, 1)

    cal = float(np.mean([conn_score, area_score, len_score]))
    return {"connectivity": float(np.clip(conn_score, 0, 1)),
            "area": float(np.clip(area_score, 0, 1)),
            "length": float(np.clip(len_score, 0, 1)),
            "CAL": float(np.clip(cal, 0, 1))}


def gradcam_attention(model, img_tensor, target_class=None):
    """Compute GradCAM attention map for ConvNeXt/ResNet (E5).

    img_tensor: (1, 3, H, W). Returns (H', W') numpy array.
    """
    grads = []
    acts  = []

    def backward_hook(module, grad_input, grad_output):
        grads.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        acts.append(output.detach())

    target_layer = None
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d,)):
            target_layer = m

    if target_layer is None:
        return np.zeros((7, 7))

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    model.eval()
    out = model(img_tensor)
    if target_class is None:
        target_class = out.argmax(1).item()
    out[0, target_class].backward()

    fh.remove()
    bh.remove()

    if not grads or not acts:
        return np.zeros((7, 7))

    alpha = grads[0].mean(dim=(2, 3), keepdim=True)
    cam   = (alpha * acts[0]).sum(dim=1).squeeze().relu()
    cam   = cam.cpu().numpy()
    cam  -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def vit_attention_rollout(model, img_tensor):
    """ViT attention rollout across all layers (E5).

    img_tensor: (1, 3, H, W). Returns (14, 14) attention map.
    """
    attentions = []
    hooks = []

    def hook_fn(module, input, output):
        if hasattr(output, "shape") and output.dim() == 4:
            attentions.append(output.detach())

    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            hooks.append(m.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        model(img_tensor)

    for h in hooks:
        h.remove()

    if not attentions:
        return np.zeros((14, 14))

    rollout = torch.eye(attentions[0].shape[-1])
    for attn in attentions:
        if attn.dim() == 4:
            a = attn.mean(0).mean(0)
            a = a + torch.eye(a.shape[-1])
            a = a / a.sum(dim=-1, keepdim=True)
            rollout = a @ rollout

    grid = rollout[0, 1:].reshape(14, 14).cpu().numpy()
    grid -= grid.min()
    if grid.max() > 0:
        grid /= grid.max()
    return grid


def compute_ece(proba, labels, n_bins=10):
    """Expected Calibration Error and reliability diagram data (H23).

    proba: (N, C) softmax probabilities. labels: (N,) integer class indices.
    Returns dict with ece, bin_confs, bin_accs, bin_counts.
    """
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correct     = (predictions == labels).astype(float)
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs, bin_accs, bin_counts = [], [], []
    ece = 0.0
    n = max(len(labels), 1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        cnt  = mask.sum()
        bin_counts.append(int(cnt))
        if cnt == 0:
            bin_confs.append(0.0)
            bin_accs.append(0.0)
        else:
            bc = float(confidences[mask].mean())
            ba = float(correct[mask].mean())
            bin_confs.append(bc)
            bin_accs.append(ba)
            ece += (cnt / n) * abs(bc - ba)
    return {"ece": float(ece), "bin_confs": bin_confs,
            "bin_accs": bin_accs, "bin_counts": bin_counts}


def compute_conformal_sets(proba, labels, cal_proba, cal_labels, alpha=0.05):
    """Split-conformal prediction sets with coverage guarantee (H16).

    Calibrates on cal_* and produces prediction sets on test proba.
    Returns empirical coverage and mean set size.
    """
    scores = 1.0 - cal_proba[np.arange(len(cal_labels)), cal_labels]
    n = len(cal_labels)
    q_level = np.ceil((n + 1) * (1 - alpha)) / max(n, 1)
    q_hat = float(np.quantile(scores, min(q_level, 1.0)))
    sets = [(proba[i] >= 1.0 - q_hat).sum() for i in range(len(proba))]
    covered = sum(
        1 for i in range(len(labels)) if proba[i, labels[i]] >= 1.0 - q_hat
    ) / max(len(labels), 1)
    return {
        "q_hat": q_hat,
        "coverage": float(covered),
        "mean_set_size": float(np.mean(sets)) if sets else float("nan"),
        "target_coverage": 1.0 - alpha,
    }
