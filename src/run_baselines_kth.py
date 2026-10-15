"""Baseline results for every backbone on KTH-TIPS2-b. Mirrors run_baselines.py.
Run:  python run_baselines_kth.py
Writes results/baselines_kth.csv
"""
import csv, numpy as np
import config, data_kth as data, ontology_kth as onto
import embeddings as emb, evaluate as ev

config.N_CLASSES = onto.N_FAMILIES  # minlength of the kNN vote count (np.bincount extends it to all labels)
BACKBONES = {  # separate KTH cache tags so they never collide with DTD caches
    "dinov2_vitb14_kth": ("vision", "dinov2"),
    "clip_vitb32_kth":   ("vlm",    "clip"),
    "siglip_vitb16_kth": ("vlm",    "siglip"),
}

def main():
    all_paths, all_labels, class_names = data.load_all()
    print(f"KTH-TIPS2-b: {len(all_paths)} images, {len(class_names)} materials")
    (tr_p, tr_y), (te_p, te_y), _ = data.load_classification_splits()
    print(f"  clf split: train {len(tr_p)} (sample a) / test {len(te_p)} (b,c,d)")

    rows = []
    for name, (family, key) in BACKBONES.items():
        print(f"\n=== {name} ({family}) ===")
        X_all = emb.get_embeddings(name, key, all_paths, tag="kth_all")
        X_tr  = emb.get_embeddings(name, key, tr_p, tag="kth_clf_train")
        X_te  = emb.get_embeddings(name, key, te_p, tag="kth_clf_test")
        ret = ev.retrieval_metrics(X_all, all_labels)
        knn = ev.knn_loo_accuracy(X_all, all_labels)
        probe = ev.linear_probe_accuracy(X_tr, tr_y, X_te, te_y)
        clu = ev.clustering_metrics(X_all, all_labels, n_clusters=len(class_names))
        row = {"backbone": name, "family": family, "dim": X_all.shape[1],
               **{k: round(v,4) for k,v in ret.items()},
               "kNN_acc": round(knn,4), "linear_probe_acc": round(probe,4),
               "NMI": round(clu["NMI"],4), "ARI": round(clu["ARI"],4)}
        rows.append(row); print("  ", row)

    out = config.RESULTS_DIR / "baselines_kth.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved {out}")

if __name__ == "__main__":
    np.random.seed(config.SEED); main()
