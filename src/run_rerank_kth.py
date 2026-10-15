"""Ontology-guided retrieval re-ranking and zero-shot prompts on KTH-TIPS2-b. Mirrors run_rerank.py.
Run:  python run_rerank_kth.py --vlm clip|siglip
Writes results/rerank_kth_<vlm>.csv
"""
import argparse, csv, numpy as np
from sklearn.preprocessing import normalize
import config, data_kth as data, ontology_kth as onto
import embeddings as emb, rerank

def softmax(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True); e = np.exp(z); return e/e.sum(axis=axis, keepdims=True)
def vlm_profiles(img, text_vecs, temp=100.0):
    I = normalize(img.astype(np.float32)); T = normalize(text_vecs.astype(np.float32))
    return softmax(temp * (I @ T.T))

VLM_KTH_CACHE = {"clip": "clip_vitb32_kth", "siglip": "siglip_vitb16_kth"}

def main(vlm):
    _, labels, class_names = data.load_all()
    y = np.asarray(labels)
    M, names = onto.similarity_matrix(class_names)
    NC = len(class_names)
    fam = np.array([onto.ATTR_TO_FAMILY[n] for n in names])
    fam_of_y = np.array([fam[c] for c in y])

    V = normalize(np.load(config.CACHE_DIR / "dinov2_vitb14_kth__kth_all.npy").astype(np.float32))
    vlm_img = np.load(config.CACHE_DIR / f"{VLM_KTH_CACHE[vlm]}__kth_all.npy")
    bare = [f"a photo of a {n.replace('_',' ')} texture" for n in names]
    rich = onto.prompts(class_names)
    T_bare = emb.encode_texts(vlm, bare); T_rich = emb.encode_texts(vlm, rich)

    print(f"=== VLM = {vlm} (KTH-TIPS2-b) ===")
    for tag, T in (("bare", T_bare), ("ontology-enriched", T_rich)):
        P = vlm_profiles(vlm_img, T)
        acc = (P.argmax(1) == y).mean()
        fam_acc = (fam[P.argmax(1)] == fam_of_y).mean()
        print(f"[A] {vlm} zero-shot ({tag:17s}): exact-acc={acc:.3f}  family-acc={fam_acc:.3f}")

    P = vlm_profiles(vlm_img, T_rich)
    Sb = V @ V.T; np.fill_diagonal(Sb, -np.inf)
    base = rerank.retrieval_from_scores(Sb, y)
    print("[B] visual-only:", {k: round(v,3) for k,v in base.items()})
    rows = [{"vlm": vlm, "method": "visual_only", "alpha": "-", **base}]
    for alpha in (0.8,0.7,0.6,0.5,0.4):
        for tag, Mmat in (("identity", np.eye(NC, dtype=np.float32)), ("ontology", M)):
            r = rerank.retrieval_from_scores(rerank.combined_scores(V, P, Mmat, alpha), y)
            rows.append({"vlm": vlm, "method": f"profiles_{tag}", "alpha": alpha, **r})
            print(f"  alpha={alpha} {tag:8s}:", {k: round(v,3) for k,v in r.items()})
    out = config.RESULTS_DIR / f"rerank_kth_{vlm}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Saved {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--vlm", choices=("clip","siglip"), default="clip")
    main(ap.parse_args().vlm)
