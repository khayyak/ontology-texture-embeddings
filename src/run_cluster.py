"""Ontology-guided clustering on DTD (multi-seed). Run: python run_cluster.py --vlm clip|siglip --seeds 10
Writes results/cluster_<vlm>.csv"""
import argparse, csv, numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
import config, data, ontology as onto, embeddings as emb
from run_rerank import vlm_profiles

def kmeans_multiseed(X, y_class, y_family, seeds):
    acc = {m: [] for m in ("NMI_class", "ARI_class", "NMI_family", "ARI_family")}
    for s in seeds:
        for k, y, tag in ((config.N_CLASSES, y_class, "class"), (8, y_family, "family")):
            pred = KMeans(n_clusters=k, random_state=s, n_init=10).fit_predict(X)
            acc[f"NMI_{tag}"].append(nmi(y, pred)); acc[f"ARI_{tag}"].append(ari(y, pred))
    return {m: (float(np.mean(v)), float(np.std(v))) for m, v in acc.items()}

def stack(V, P=None, lam=1.0):
    parts = [StandardScaler().fit_transform(V)]
    if P is not None: parts.append(lam * StandardScaler().fit_transform(P))
    return np.hstack(parts)

def main(vlm, n_seeds):
    seeds = list(range(n_seeds))
    _, labels, class_names = data.load_all(); y = np.asarray(labels)
    fam_names = sorted(set(onto.ATTR_TO_FAMILY.values())); fid = {f: i for i, f in enumerate(fam_names)}
    y_fam = np.array([fid[onto.ATTR_TO_FAMILY[class_names[c]]] for c in y])
    M, _ = onto.similarity_matrix(class_names)
    V = normalize(np.load(config.CACHE_DIR / "dinov2_vitb14__all.npy").astype(np.float32))
    vlm_img = np.load(config.CACHE_DIR / f"{config.VLM_CACHE[vlm]}__all.npy")
    P = vlm_profiles(vlm_img, emb.encode_texts(vlm, onto.prompts(class_names)))
    P_onto = P @ M.T; P_onto /= P_onto.sum(1, keepdims=True) + 1e-12
    print(f"=== VLM = {vlm} | KMeans x {n_seeds} seeds (mean +/- std) ===")
    rows = []
    def run(tag, X):
        res = kmeans_multiseed(X, y, y_fam, seeds)
        row = {"vlm": vlm, "features": tag}
        for m, (mu, sd) in res.items(): row[m] = f"{mu:.4f}"; row[m + "_std"] = f"{sd:.4f}"
        rows.append(row)
        print(f"  {tag:20s} " + "  ".join(f"{m}={mu:.3f}+/-{sd:.3f}" for m, (mu, sd) in res.items()))
    run("V (baseline)", stack(V))
    for lam in (0.5, 1.0):
        run(f"V+P (lam={lam})", stack(V, P, lam))
        run(f"V+P_onto (lam={lam})", stack(V, P_onto, lam))
    out = config.RESULTS_DIR / f"cluster_{vlm}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--vlm", choices=config.VLMS, default="clip")
    ap.add_argument("--seeds", type=int, default=10); a = ap.parse_args(); main(a.vlm, a.seeds)
