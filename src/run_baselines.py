"""Baseline results for every backbone on DTD.  Run: python run_baselines.py
Writes results/baselines.csv"""
import csv, numpy as np
import config, data, embeddings as emb, evaluate as ev

def main():
    print("Loading DTD (whole set for retrieval/clustering) ...")
    all_paths, all_labels, class_names = data.load_all()
    print(f"  {len(all_paths)} images, {len(class_names)} classes")
    (tr_p, tr_y), (te_p, te_y), _ = data.load_classification_splits()
    print(f"  train {len(tr_p)} / test {len(te_p)}")
    rows = []
    for name, (family, loader_key) in config.BACKBONES.items():
        print(f"\n=== {name} ({family}) ===")
        X_all = emb.get_embeddings(name, loader_key, all_paths, tag="all")
        X_tr = emb.get_embeddings(name, loader_key, tr_p, tag="clf_train")
        X_te = emb.get_embeddings(name, loader_key, te_p, tag="clf_test")
        ret = ev.retrieval_metrics(X_all, all_labels)
        knn = ev.knn_loo_accuracy(X_all, all_labels)
        probe = ev.linear_probe_accuracy(X_tr, tr_y, X_te, te_y)
        clu = ev.clustering_metrics(X_all, all_labels)
        row = {"backbone": name, "family": family, "dim": X_all.shape[1],
               **{k: round(v, 4) for k, v in ret.items()},
               "kNN_acc": round(knn, 4), "linear_probe_acc": round(probe, 4),
               "NMI": round(clu["NMI"], 4), "ARI": round(clu["ARI"], 4)}
        rows.append(row); print("  ", row)
    out = config.RESULTS_DIR / "baselines.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved {out}")
    cols = list(rows[0].keys())
    print("\n" + " | ".join(cols))
    for r in rows: print(" | ".join(str(r[c]) for c in cols))

if __name__ == "__main__":
    np.random.seed(config.SEED); main()
