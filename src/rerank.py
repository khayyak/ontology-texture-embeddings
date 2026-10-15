"""Ontology-guided re-ranking (leakage-free), bilinear semantic term.
  final(q,c) = alpha * visual(q,c) + (1-alpha) * (p_q^T M p_c)
Same attribute: 1, same family: 0.5, different family: 0.
The profiles p are model predictions (zero-shot CLIP or SigLIP), never ground-truth labels.
Pass M = identity for the ablation that isolates the ontology's contribution."""
import numpy as np

def _row_minmax(S):
    lo = S.min(axis=1, keepdims=True); hi = S.max(axis=1, keepdims=True)
    return (S - lo) / (hi - lo + 1e-12)

def combined_scores(V, P, M, alpha):
    """(N,N) re-ranking scores; self set to -inf.
      V:(N,D) L2-normalised visual embeddings; P:(N,C) profiles; M:(C,C) ontology."""
    S_vis = V @ V.T
    S_sem = P @ M @ P.T
    S = alpha * _row_minmax(S_vis) + (1 - alpha) * _row_minmax(S_sem)
    np.fill_diagonal(S, -np.inf)
    return S

def retrieval_from_scores(S, labels, top_k=(1, 5, 10)):
    y = np.asarray(labels); N = len(y)
    order = np.argsort(-S, axis=1)
    rel = (y[order] == y[:, None]) & (order != np.arange(N)[:, None])
    n_per = {c: int((y == c).sum()) for c in np.unique(y)}
    pk = {k: np.zeros(N) for k in top_k}
    ap = np.full(N, np.nan)
    for i in range(N):
        r = rel[i]
        for k in top_k:
            pk[k][i] = r[:k].mean()
        nr = n_per[y[i]] - 1
        if nr <= 0:
            continue
        hits = np.where(r)[0]
        ap[i] = np.sum([(j + 1) / (idx + 1) for j, idx in enumerate(hits)]) / nr
    def macro(v):
        vals = []
        for c in np.unique(y):
            vc = v[y == c]; vc = vc[~np.isnan(vc)]
            if vc.size: vals.append(vc.mean())
        return float(np.mean(vals)) if vals else 0.0
    out = {f"P@{k}": macro(pk[k]) for k in top_k}
    out["mAP"] = macro(ap)
    return out
