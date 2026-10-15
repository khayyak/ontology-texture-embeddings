"""Diagnostic of DINOv2 retrieval on DTD. Splits DINOv2 top-10 errors into
within-family (ontology can't fix) vs cross-family (ontology can demote), and computes
the P@10 ceiling. Ground-truth labels are used for analysis only. Run after run_baselines.py."""
import numpy as np
from sklearn.preprocessing import normalize
import config, data, ontology as onto

def main():
    _, labels, class_names = data.load_all()
    y = np.asarray(labels)
    emb = np.load(config.CACHE_DIR / "dinov2_vitb14__all.npy")
    X = normalize(emb.astype(np.float32))
    fam = np.array([onto.ATTR_TO_FAMILY[c] for c in class_names])
    S = X @ X.T; np.fill_diagonal(S, -np.inf); N = len(y)
    top = np.argsort(-S, axis=1)[:, :10]
    same_class = (y[top] == y[:, None])
    same_fam = (fam[y[top]] == fam[y][:, None])
    wrong = ~same_class
    within = wrong & same_fam
    cross = wrong & ~same_fam
    nw = wrong.sum()
    print(f"DINOv2 top-10 over {N} queries")
    print(f"  baseline P@10        : {same_class.mean():.3f}")
    print(f"  errors total         : {nw} slots ({nw/(N*10):.1%} of top-10)")
    print(f"  within-family errors : {within.sum()/nw:.1%} of errors  (not fixable by the ontology)")
    print(f"  cross-family errors  : {cross.sum()/nw:.1%} of errors  (can be demoted by the ontology)")
    ceiling = (same_class | cross).mean()
    print(f"  P@10 ceiling (no cross-family errors) : {ceiling:.3f}")
    print(f"  max possible ontology gain on P@10     : +{ceiling - same_class.mean():.3f}")
    print("\n  most cross-family-confused query classes:")
    pcc = np.array([cross[y == c].sum() / max((y == c).sum()*10, 1) for c in range(config.N_CLASSES)])
    for c in np.argsort(-pcc)[:8]:
        print(f"    {class_names[c]:14s} {pcc[c]:.1%} cross-family in top-10")

if __name__ == "__main__":
    main()
