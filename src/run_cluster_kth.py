"""Ontology-guided clustering (multi-seed) on KTH-TIPS2-b. Mirrors run_cluster.py.
Run:  python run_cluster_kth.py --vlm clip|siglip --seeds 10
Writes results/cluster_kth_<vlm>.csv
"""
import argparse, csv, numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
import config, data_kth as data, ontology_kth as onto
import embeddings as emb
from run_rerank_kth import vlm_profiles, VLM_KTH_CACHE

def kmeans_multiseed(X, yc, yf, nc, nf, seeds):
    acc = {m: [] for m in ("NMI_class","ARI_class","NMI_family","ARI_family")}
    for s in seeds:
        for k, y, tag in ((nc, yc, "class"), (nf, yf, "family")):
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
    M, names = onto.similarity_matrix(class_names); NC = len(class_names)
    fam_names = sorted(set(onto.ATTR_TO_FAMILY.values())); fid = {f:i for i,f in enumerate(fam_names)}
    y_fam = np.array([fid[onto.ATTR_TO_FAMILY[names[c]]] for c in y]); NF = len(fam_names)

    V = normalize(np.load(config.CACHE_DIR / "dinov2_vitb14_kth__kth_all.npy").astype(np.float32))
    vlm_img = np.load(config.CACHE_DIR / f"{VLM_KTH_CACHE[vlm]}__kth_all.npy")
    T_rich = emb.encode_texts(vlm, onto.prompts(class_names))
    P = vlm_profiles(vlm_img, T_rich)
    P_onto = P @ M.T; P_onto /= P_onto.sum(1, keepdims=True) + 1e-12

    print(f"=== VLM={vlm} | KTH-TIPS2-b | KMeans x {n_seeds} seeds ===")
    rows = []
    def run(tag, X):
        res = kmeans_multiseed(X, y, y_fam, NC, NF, seeds)
        row = {"vlm": vlm, "features": tag}
        for m,(mu,sd) in res.items(): row[m]=f"{mu:.4f}"; row[m+"_std"]=f"{sd:.4f}"
        rows.append(row)
        print(f"  {tag:20s} " + "  ".join(f"{m}={mu:.3f}+/-{sd:.3f}" for m,(mu,sd) in res.items()))
    run("V (baseline)", stack(V))
    for lam in (0.5, 1.0):
        run(f"V+P (lam={lam})", stack(V, P, lam))
        run(f"V+P_onto (lam={lam})", stack(V, P_onto, lam))
    out = config.RESULTS_DIR / f"cluster_kth_{vlm}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Saved {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--vlm", choices=("clip","siglip"), default="clip")
    ap.add_argument("--seeds", type=int, default=10); a = ap.parse_args(); main(a.vlm, a.seeds)
