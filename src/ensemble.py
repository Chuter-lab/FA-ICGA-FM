"""Greedy ensemble of backbone predictions (F1).

Greedy forward selection: start with best single model, add models only if
they improve held-out AUC. Reports ensemble vs. best single model.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def _auc(proba, labels, n_classes):
    try:
        if n_classes == 2:
            return roc_auc_score(labels, proba[:, 1])
        y_b = label_binarize(labels, classes=list(range(n_classes)))
        return roc_auc_score(y_b, proba, multi_class="ovr", average="macro")
    except Exception:
        return float("nan")


def greedy_ensemble(model_probas, val_labels, n_classes):
    """Greedy forward-selection ensemble over dict of {name: proba_array}.

    Returns:
        ensemble_proba: (N, C) array from best greedy ensemble
        selected: list of model names in ensemble
        improvement: AUC gain vs. best single model
    """
    names = list(model_probas.keys())
    if not names:
        return None, [], 0.0

    best_single = max(names, key=lambda n: _auc(model_probas[n], val_labels, n_classes))
    best_auc    = _auc(model_probas[best_single], val_labels, n_classes)
    selected    = [best_single]
    current     = model_probas[best_single].copy()

    for _ in range(len(names) - 1):
        best_gain, best_add, best_proba = 0.0, None, None
        for n in names:
            if n in selected:
                continue
            candidate = (current * len(selected) + model_probas[n]) / (len(selected) + 1)
            auc = _auc(candidate, val_labels, n_classes)
            if auc - _auc(current, val_labels, n_classes) > best_gain:
                best_gain, best_add, best_proba = (
                    auc - _auc(current, val_labels, n_classes), n, candidate
                )
        if best_add is None:
            break
        selected.append(best_add)
        current = best_proba

    ensemble_auc = _auc(current, val_labels, n_classes)
    return current, selected, ensemble_auc - best_auc
