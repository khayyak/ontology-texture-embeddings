"""Baseline evaluation of raw embeddings: retrieval, classification, clustering.
All embeddings L2-normalised (cosine == dot)."""
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import config

def retrieval_metrics(emb, labels, top_k=config.TOP_K):
    X = normalize(emb.astype(np.float32)); y = np.asarray(labels); N = len(y)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)
    ranked_rel = (y[order] == y[:, None])
    ranked_rel &= (order != np.arange(N)[:, None])   # drop self from relevance
    n_per_class = {c: int((y == c).sum()) for c in np.unique(y)}
    p_at_k = {k: np.zeros(N) for k in top_k}
    ap = np.full(N, np.nan)
    for i in range(N):
        rel = ranked_rel[i]
        for k in top_k:
            p_at_k[k][i] = rel[:k].mean()
        n_rel = n_per_class[y[i]] - 1
        if n_rel <= 0:
            continue
        hits = np.where(rel)[0]
        ap[i] = np.sum([(j + 1) / (idx + 1) for j, idx in enumerate(hits)]) / n_rel
    def macro(vals):
        out = []
        for c in np.unique(y):
            vc = vals[y == c]; vc = vc[~np.isnan(vc)]
            if vc.size: out.append(vc.mean())
        return float(np.mean(out)) if out else 0.0
    res = {f"P@{k}": macro(p_at_k[k]) for k in top_k}
    res["mAP"] = macro(ap)
    return res

def knn_loo_accuracy(emb, labels, k=config.KNN_K):
    X = normalize(emb.astype(np.float32)); y = np.asarray(labels)
    knn = KNeighborsClassifier(n_neighbors=k + 1, metric="euclidean")
    knn.fit(X, y)
    _, idx = knn.kneighbors(X, n_neighbors=k + 1)
    idx = idx[:, 1:]
    preds = np.array([np.bincount(y[row], minlength=config.N_CLASSES).argmax() for row in idx])
    return float((preds == y).mean())

def linear_probe_accuracy(tr_emb, tr_y, te_emb, te_y):
    Xtr = normalize(tr_emb.astype(np.float32)); Xte = normalize(te_emb.astype(np.float32))
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, np.asarray(tr_y))
    return float(clf.score(Xte, np.asarray(te_y)))

def clustering_metrics(emb, labels, n_clusters=config.N_CLASSES, seed=config.SEED):
    X = normalize(emb.astype(np.float32)); y = np.asarray(labels)
    pred = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(X)
    return {"NMI": float(normalized_mutual_info_score(y, pred)),
            "ARI": float(adjusted_rand_score(y, pred))}
