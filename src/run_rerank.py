"""Ontology-guided retrieval re-ranking and zero-shot prompts on DTD.
Run: python run_rerank.py --vlm clip|siglip   Writes results/rerank_<vlm>.csv"""
import argparse, csv, numpy as np
from sklearn.preprocessing import normalize
import config, data, ontology as onto, embeddings as emb, rerank

def softmax(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True); e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)

def vlm_profiles(img, text_vecs, temp=100.0):
    I = normalize(img.astype(np.float32)); T = normalize(text_vecs.astype(np.float32))
    return softmax(temp * (I @ T.T))

def main(vlm):
    _, labels, class_names = data.load_all()
    y = np.asarray(labels)
    M, names = onto.similarity_matrix(class_names)
    assert list(names) == list(class_names)
    fam = np.array([onto.ATTR_TO_FAMILY[c] for c in class_names])
    V = normalize(np.load(config.CACHE_DIR / "dinov2_vitb14__all.npy").astype(np.float32))
    vlm_img = np.load(config.CACHE_DIR / f"{config.VLM_CACHE[vlm]}__all.npy")
    bare = [f"a photo of a {a} texture" for a in class_names]
    rich = onto.prompts(class_names)
    T_bare = emb.encode_texts(vlm, bare); T_rich = emb.encode_texts(vlm, rich)
    print(f"=== VLM = {vlm} ===")
    for tag, T in (("bare", T_bare), ("ontology-enriched", T_rich)):
        P = vlm_profiles(vlm_img, T)
        acc = (P.argmax(1) == y).mean(); fam_acc = (fam[P.argmax(1)] == fam[y]).mean()
        print(f"[A] {vlm} zero-shot ({tag:17s}): exact-acc={acc:.3f}  family-acc={fam_acc:.3f}")
    P = vlm_profiles(vlm_img, T_rich)
    print(f"\n[B] retrieval re-ranking with {vlm} profiles")
    Sb = V @ V.T; np.fill_diagonal(Sb, -np.inf)
    base = rerank.retrieval_from_scores(Sb, y)
    print("  visual-only        :", {k: round(v, 3) for k, v in base.items()})
    rows = [{"vlm": vlm, "method": "visual_only", "alpha": "-", **base}]
    for alpha in (0.8, 0.7, 0.6, 0.5, 0.4):
        for tag, Mmat in (("identity", np.eye(config.N_CLASSES, dtype=np.float32)), ("ontology", M)):
            r = rerank.retrieval_from_scores(rerank.combined_scores(V, P, Mmat, alpha), y)
            rows.append({"vlm": vlm, "method": f"profiles_{tag}", "alpha": alpha, **r})
            print(f"  alpha={alpha} {tag:8s}:", {k: round(v, 3) for k, v in r.items()})
    out = config.RESULTS_DIR / f"rerank_{vlm}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--vlm", choices=config.VLMS, default="clip")
    main(ap.parse_args().vlm)
