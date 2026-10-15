#!/usr/bin/env python
"""
run_extensions.py: experiments on DTD and KTH-TIPS2-b

All experiments read the image embeddings cached by run_baselines.py and
run_baselines_kth.py (cache/*.npy); no image features are re-extracted. Prompt text
embeddings are computed once and cached as cache/ext_text__*.npy.

Run from the src/ folder, for example:

    python run_extensions.py --exp sanity
    python run_extensions.py --exp alpha_sweep --out ../results_new
    python run_extensions.py --exp all --out ../results_new --resume

Experiments:
  sanity           check that the caches reproduce the reported main results
  alpha_sweep      re-ranking vs alpha, identity vs ontology, CLIP and SigLIP
  profile_source   bare vs ontology-enriched prompts as the profile source
  lambda_sweep     clustering vs lambda (V, V+P, V+P_onto)
  main_clustering  main clustering results (10 seeds, lambda 0.5 and 1.0)
  random_control   real ontology vs size-matched random ontologies
  perturbation     performance vs number of randomly re-assigned attributes
  alt_assign       DTD only: borderline attributes moved to an alternative family
  family_weight    same-family weight w in M (0 = identity ... 1)
  auto_ontology    ontology derived automatically from text embeddings of the names
  kth_fair         KTH only: retrieval / kNN under the sample-split protocol
  bootstrap        95% bootstrap CIs, McNemar and Wilcoxon tests
  classification   linear probe and kNN on the fused features z(x)
  per_family       per-class / per-family breakdown
  silhouette       family- and class-level silhouette of V, V+P and V+P_onto
  qualitative      DTD retrieval examples before / after re-ranking

Every experiment writes its CSV(s) to --out as soon as it finishes; --resume skips
experiments whose CSV already exists.
"""
import argparse, csv, hashlib, json, time, warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

import config
import rerank
import ontology as ONTO_DTD
import ontology_kth as ONTO_KTH

warnings.filterwarnings("ignore", category=FutureWarning)

# Settings
CACHE = Path(config.CACHE_DIR)
TEMP = 100.0                              # same as run_rerank.py
ALPHA_MAIN = 0.7                          # alpha used for the main re-ranking results
LAMBDA_MAIN = {"dtd": 1.0, "kth": 0.5}    # lambda used for the main clustering results
VLMS = ("clip", "siglip")
PRIMARY = "siglip"                        # profiles used for the main results

FILES = {
    "dtd": {"dinov2": "dinov2_vitb14__all.npy", "clip": "clip_vitb32__all.npy",
            "siglip": "siglip_vitb16__all.npy"},
    "kth": {"dinov2": "dinov2_vitb14_kth__kth_all.npy", "clip": "clip_vitb32_kth__kth_all.npy",
            "siglip": "siglip_vitb16_kth__kth_all.npy"},
}

# Main results of run_baselines*.py, run_rerank*.py and run_cluster*.py, used by --exp sanity.
EXPECTED = {
    "dtd": {"vis": {"P@1": 0.7915, "P@10": 0.7030, "mAP": 0.4412},
            "zs": {("clip", "bare"): 0.399, ("clip", "rich"): 0.470,
                   ("siglip", "bare"): 0.623, ("siglip", "rich"): 0.555},
            "rr_identity": {"P@1": 0.791, "P@10": 0.717, "mAP": 0.540},
            "rr_ontology": {"P@1": 0.797, "P@10": 0.723, "mAP": 0.528},
            "clu": {"V": 0.0973, "V+P_onto": 0.2838}},
    "kth": {"vis": {"P@1": 0.9998, "P@10": 0.9959, "mAP": 0.8259},
            "zs": {("clip", "bare"): 0.577, ("clip", "rich"): 0.606,
                   ("siglip", "bare"): 0.824, ("siglip", "rich"): 0.806},
            "rr_identity": {"P@1": 0.997, "P@10": 0.985, "mAP": 0.861},
            "rr_ontology": {"P@1": 0.999, "P@10": 0.993, "mAP": 0.852},
            "clu": {"V": 0.6941, "V+P_onto": 0.7722}},
}

# DTD sensitivity analysis: the five borderline assignments listed in ontology.py.
# Value = the alternative family tested for each attribute.
ALTERNATIVES = {
    "honeycombed": "HOLES_POROUS",   # net vs holes
    "marbled":     "SWIRLY_CURVED",  # veined vs swirly
    "zigzagged":   "SWIRLY_CURVED",  # straight vs non-straight line patterns
    "cobwebbed":   "GRID_NET",       # fibrous vs web-like mesh
    "scaly":       "GRID_NET",       # relief vs regular repeated tiling
}

DEC = 6   # decimals written to the CSVs (6 avoids double rounding when tables use 3)

SET = dict(seeds_main=10, seeds_sweep=5, n_init=10, R_random=20, R_perturb=8,
           seeds_perturb=3, B_boot=2000,
           alphas=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
           lambdas=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
           weights=[0.0, 0.25, 0.5, 0.75, 1.0],
           k_perturb={"dtd": [1, 2, 3, 5, 8, 12], "kth": [1, 2, 3, 5]})
QUICK = dict(seeds_main=2, seeds_sweep=2, n_init=2, R_random=3, R_perturb=2,
             seeds_perturb=1, B_boot=200, alphas=[0.5, 0.7, 1.0], lambdas=[0.5, 1.0],
             weights=[0.0, 0.5, 1.0], k_perturb={"dtd": [1, 5], "kth": [1, 3]})

OUT = PRED = None

# Utilities
def log(*a):
    print(*a, flush=True)


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    log(f"  -> saved {path}")


def ahash(*objs):
    h = hashlib.md5()
    for o in objs:
        h.update(np.ascontiguousarray(o).tobytes() if isinstance(o, np.ndarray) else str(o).encode())
    return h.hexdigest()[:10]


# Data
_DS = {}


def load_ds(name):
    """Load labels, ontology and cached embeddings for 'dtd' or 'kth' (in cache order)."""
    if name in _DS:
        return _DS[name]
    if name == "dtd":
        import data as dm
        onto = ONTO_DTD
        paths, labels, cls = dm.load_all()
        names = [str(c) for c in cls]
        sample = None
        split = dm.load_classification_splits()
    else:
        import data_kth as dm
        onto = ONTO_KTH
        recs, cls = dm._scan("data")              # same order as data_kth.load_all()
        paths = [r[0] for r in recs]
        labels = [r[1] for r in recs]
        sample = np.array([r[2] for r in recs])
        names = [onto._norm(c) for c in cls]
        split = dm.load_classification_splits()
    paths = [str(p) for p in paths]
    y = np.asarray(labels)
    M, mnames = onto.similarity_matrix(cls)
    if [str(n) for n in mnames] != names:
        raise RuntimeError("class order differs from the ontology order")
    fam_names = sorted(set(onto.ATTR_TO_FAMILY.values()))
    fam_of_class = np.array([fam_names.index(onto.ATTR_TO_FAMILY[n]) for n in names])
    img = {}
    for key, fn in FILES[name].items():
        f = CACHE / fn
        if not f.exists():
            raise FileNotFoundError(f"missing cache {f}; run run_baselines{'_kth' if name == 'kth' else ''}.py first")
        X = np.load(f).astype(np.float32)
        if X.shape[0] != len(y):
            raise ValueError(f"{f} has {X.shape[0]} rows but the dataset has {len(y)} images")
        img[key] = X
    (trp, _), (tep, _), _ = split
    pos = {p: i for i, p in enumerate(paths)}
    D = dict(name=name, onto=onto, paths=paths, y=y, cls=list(cls), names=names,
             sample=sample, M=M.astype(np.float32), fam_names=fam_names,
             fam_of_class=fam_of_class, yfam=fam_of_class[y], nc=len(names),
             nf=len(fam_names), V=normalize(img["dinov2"]), img=img,
             tr=np.array([pos[str(p)] for p in trp]), te=np.array([pos[str(p)] for p in tep]))
    log(f"[{name}] {len(y)} images, {D['nc']} classes, {D['nf']} families")
    _DS[name] = D
    return D


def prompts_for(D, kind):
    if kind == "bare":   # identical to run_rerank.py / run_rerank_kth.py
        return [f"a photo of a {n.replace('_', ' ')} texture" for n in D["names"]]
    return D["onto"].prompts(D["cls"])   # ontology-enriched


def text_emb(D, vlm, kind):
    f = CACHE / f"ext_text__{D['name']}__{vlm}__{kind}.npy"
    if f.exists():
        return np.load(f).astype(np.float32)
    import embeddings as E
    T = np.asarray(E.encode_texts(vlm, prompts_for(D, kind)), dtype=np.float32)
    np.save(f, T)
    return T


def profiles(D, vlm=PRIMARY, kind="rich"):
    """p(x) = softmax(tau * cos(g(x), h(t_k))), as vlm_profiles() in run_rerank.py."""
    key = ("P", vlm, kind)
    if key not in D:
        I = normalize(D["img"][vlm])
        T = normalize(text_emb(D, vlm, kind))
        Z = TEMP * (I @ T.T)
        Z = Z - Z.max(axis=1, keepdims=True)
        E_ = np.exp(Z)
        D[key] = E_ / E_.sum(axis=1, keepdims=True)
    return D[key]


def build_M(assign, w=0.5):
    a = np.asarray(assign)
    M = (a[:, None] == a[None, :]).astype(np.float32) * np.float32(w)
    np.fill_diagonal(M, 1.0)
    return M


def smooth(P, M):
    Q = P @ M.T
    return Q / (Q.sum(axis=1, keepdims=True) + 1e-12)


# Retrieval
def vis_scores(D):
    if "S_vis" not in D:
        S = D["V"] @ D["V"].T
        np.fill_diagonal(S, -np.inf)
        D["S_vis"] = S
    return D["S_vis"]


def rr_scores(D, M, alpha=ALPHA_MAIN, vlm=PRIMARY, kind="rich"):
    return rerank.combined_scores(D["V"], profiles(D, vlm, kind), M, alpha)


def same_sample_mask(D):
    """True = item may appear in the gallery (excludes self and the query's physical sample)."""
    s = D["sample"]
    return s[:, None] != s[None, :]


def retrieval_eval(Sc, y, valid=None, top_k=(1, 5, 10), chunk=1000):
    """Vectorised equivalent of rerank.retrieval_from_scores (same ranking, same macro
    averaging). Returns (macro dict, per-query dict). `valid` removes items from both the
    ranking and the relevant set."""
    N = len(y)
    pk = {k: np.zeros(N) for k in top_k}
    ap = np.full(N, np.nan)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        Sb = Sc[s:e]
        if valid is not None:
            Sb = np.where(valid[s:e], Sb, -np.inf)
        order = np.argsort(-Sb, axis=1)
        rel = (y[order] == y[s:e, None]) & (order != np.arange(s, e)[:, None])
        if valid is not None:
            rel &= np.take_along_axis(valid[s:e], order, axis=1)
        for k in top_k:
            pk[k][s:e] = rel[:, :k].mean(axis=1)
        cum = np.cumsum(rel, axis=1)
        prec = cum / np.arange(1, N + 1)[None, :]
        nrel = rel.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ap[s:e] = np.where(nrel > 0, (prec * rel).sum(axis=1) / nrel, np.nan)
    per = {f"P@{k}": pk[k] for k in top_k}
    per["mAP"] = ap
    macro = {}
    for m, v in per.items():
        vals = [np.nanmean(v[y == c]) for c in np.unique(y) if np.any(~np.isnan(v[y == c]))]
        macro[m] = float(np.mean(vals))
    return macro, per


def knn_eval(Sc, y, yfam_of_label, k=5, valid=None):
    """Majority-vote kNN on a score matrix (self excluded). Returns exact & family accuracy."""
    Sb = Sc if valid is None else np.where(valid, Sc, -np.inf)
    top = np.argpartition(-Sb, kth=k, axis=1)[:, :k]
    nb = y[top]
    nlab = int(y.max()) + 1
    pred = np.array([np.bincount(r, minlength=nlab).argmax() for r in nb])
    return float((pred == y).mean()), float((yfam_of_label[pred] == yfam_of_label[y]).mean())


# Clustering
def zfeat(D, kind, lam=1.0, vlm=PRIMARY, pkind="rich", M=None):
    """z(x) = [std(f(x)); lam * std(q(x))], as stack() in run_cluster.py."""
    parts = [StandardScaler().fit_transform(D["V"])]
    if kind != "V":
        P = profiles(D, vlm, pkind)
        Q = P if kind == "V+P" else smooth(P, M)
        parts.append(lam * StandardScaler().fit_transform(Q))
    return np.hstack(parts)


def ftag(kind, lam=None, vlm=PRIMARY, pkind="rich", M=None):
    if kind == "V":
        return "V"
    t = f"{'VP' if kind == 'V+P' else 'VPO'}_{vlm}_{pkind}_l{lam}"
    return t if kind == "V+P" else t + "_M" + ahash(M)


def cluster_preds(D, kind, k, seeds, lam=None, vlm=PRIMARY, pkind="rich", M=None):
    """K-means labels per seed, cached on disk (re-used across experiments)."""
    tag = ftag(kind, lam, vlm, pkind, M)
    out, X = [], None
    for s in seeds:
        f = PRED / f"{D['name']}__{tag}__k{k}__ni{SET['n_init']}__s{s}.npy"
        if f.exists():
            out.append(np.load(f))
            continue
        if X is None:
            X = zfeat(D, kind, lam, vlm, pkind, M)
        p = KMeans(n_clusters=k, random_state=s, n_init=SET["n_init"]).fit_predict(X)
        np.save(f, p.astype(np.int16))
        out.append(p)
    return out


def scores_vs(preds, target):
    return np.array([ARI(target, p) for p in preds]), np.array([NMI(target, p) for p in preds])


def clu_summary(D, kind, seeds, lam=None, M=None, vlm=PRIMARY, pkind="rich",
                fam_targets=None, levels=("class", "family")):
    """Returns dict of mean/std of ARI/NMI at class level and vs each family target."""
    res = {}
    if "class" in levels:
        pc = cluster_preds(D, kind, D["nc"], seeds, lam, vlm, pkind, M)
        a, n = scores_vs(pc, D["y"])
        res.update(ARI_class=a.mean(), ARI_class_sd=a.std(), NMI_class=n.mean(),
                   NMI_class_sd=n.std(), _ARI_class=a)
    if "family" in levels:
        pf = cluster_preds(D, kind, D["nf"], seeds, lam, vlm, pkind, M)
        targets = fam_targets or {"fam": D["yfam"]}
        for tn, tgt in targets.items():
            a, n = scores_vs(pf, tgt)
            res.update({f"ARI_{tn}": a.mean(), f"ARI_{tn}_sd": a.std(),
                        f"NMI_{tn}": n.mean(), f"NMI_{tn}_sd": n.std(), f"_ARI_{tn}": a})
    return res


def public(d):
    return {k: (round(float(v), DEC) if isinstance(v, (float, np.floating)) else v)
            for k, v in d.items() if not k.startswith("_")}


# 0. Sanity check
def check(label, got, exp, tol):
    ok = abs(got - exp) <= tol
    log(f"  [{'PASS' if ok else 'FAIL'}] {label:40s} got {got:.4f}  expected {exp:.4f}")
    return ok


def run_sanity(ds):
    D = load_ds(ds)
    E = EXPECTED[ds]
    ok = True
    log(f"\n=== sanity [{ds}]: do the caches reproduce the main results? ===")
    ok &= bool(np.allclose(build_M(D["fam_of_class"], 0.5), D["M"]))
    log(f"  [{'PASS' if ok else 'FAIL'}] ontology M == Wu-Palmer 1/0.5/0 matrix")
    m, _ = retrieval_eval(vis_scores(D), D["y"])
    for k, v in E["vis"].items():
        ok &= check(f"visual-only DINOv2 {k}", m[k], v, 0.0015)
    for (vlm, kind), v in E["zs"].items():
        acc = float((profiles(D, vlm, kind).argmax(1) == D["y"]).mean())
        ok &= check(f"zero-shot {vlm} {kind}", acc, v, 0.0015)
    for tag, M in (("identity", np.eye(D["nc"], dtype=np.float32)), ("ontology", D["M"])):
        m, _ = retrieval_eval(rr_scores(D, M), D["y"])
        for k, v in E[f"rr_{tag}"].items():
            ok &= check(f"re-rank SigLIP a=0.7 {tag} {k}", m[k], v, 0.0015)
    seeds = range(SET["seeds_main"])
    r = clu_summary(D, "V", seeds, levels=("family",))
    ok &= check("clustering V  ARI(family)", r["ARI_fam"], E["clu"]["V"], 0.002)
    r = clu_summary(D, "V+P_onto", seeds, LAMBDA_MAIN[ds], D["M"], levels=("family",))
    ok &= check(f"clustering V+P_onto(l={LAMBDA_MAIN[ds]}) ARI(family)", r["ARI_fam"],
                E["clu"]["V+P_onto"], 0.002)
    if QUICK_MODE:
        log("  (quick mode: fewer seeds/inits, so clustering checks are expected to differ)")
    log(f"=== sanity [{ds}] {'ALL PASS' if ok else 'SOME CHECKS FAILED'} ===")
    write_csv(OUT / f"sanity_{ds}.csv", [{"dataset": ds, "all_pass": ok}])


# 1. Alpha sweep
def run_alpha_sweep(ds):
    D = load_ds(ds)
    rows = []
    m, _ = retrieval_eval(vis_scores(D), D["y"])
    rows.append({"dataset": ds, "vlm": "-", "M": "visual-only", "alpha": 1.0, **public(m)})
    for vlm in VLMS:
        for a in SET["alphas"]:
            for tag, M in (("identity", np.eye(D["nc"], dtype=np.float32)), ("ontology", D["M"])):
                m, _ = retrieval_eval(rr_scores(D, M, a, vlm), D["y"])
                rows.append({"dataset": ds, "vlm": vlm, "M": tag, "alpha": a, **public(m)})
                log(f"  {vlm:6s} a={a:.1f} {tag:8s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
    write_csv(OUT / f"alpha_sweep_{ds}.csv", rows)


# 2. Profile source
def run_profile_source(ds):
    D = load_ds(ds)
    rows = []
    seeds = range(SET["seeds_sweep"])
    lam = LAMBDA_MAIN[ds]
    yfl = D["fam_of_class"]
    for vlm in VLMS:
        for kind in ("bare", "rich"):
            P = profiles(D, vlm, kind)
            pred = P.argmax(1)
            row = {"dataset": ds, "vlm": vlm, "prompts": kind,
                   "zs_exact": round(float((pred == D["y"]).mean()), DEC),
                   "zs_family": round(float((yfl[pred] == D["yfam"]).mean()), DEC)}
            for tag, M in (("identity", np.eye(D["nc"], dtype=np.float32)), ("ontology", D["M"])):
                m, _ = retrieval_eval(rr_scores(D, M, ALPHA_MAIN, vlm, kind), D["y"])
                row.update({f"rr_{tag}_{k}": round(v, DEC) for k, v in m.items()})
            for kd in ("V+P", "V+P_onto"):
                r = clu_summary(D, kd, seeds, lam, D["M"] if kd == "V+P_onto" else None, vlm, kind)
                row.update({f"{kd}_ARI_fam": round(r["ARI_fam"], DEC), f"{kd}_ARI_class": round(r["ARI_class"], DEC)})
            rows.append(row)
            log("  " + ", ".join(f"{k}={v}" for k, v in row.items()))
    write_csv(OUT / f"profile_source_{ds}.csv", rows)


# 3. Lambda sweep
def run_lambda_sweep(ds):
    D = load_ds(ds)
    seeds = range(SET["seeds_sweep"])
    rows = []
    r = clu_summary(D, "V", seeds)
    rows.append({"dataset": ds, "features": "V", "lambda": 0.0, **public(r)})
    for lam in SET["lambdas"]:
        for kd in ("V+P", "V+P_onto"):
            r = clu_summary(D, kd, seeds, lam, D["M"] if kd == "V+P_onto" else None)
            rows.append({"dataset": ds, "features": kd, "lambda": lam, **public(r)})
            log(f"  lam={lam:<4} {kd:9s} ARI_fam={r['ARI_fam']:.3f}+/-{r['ARI_fam_sd']:.3f}  "
                f"ARI_class={r['ARI_class']:.3f}+/-{r['ARI_class_sd']:.3f}")
    write_csv(OUT / f"lambda_sweep_{ds}.csv", rows)


def run_main_clustering(ds):
    """Main clustering table (10 seeds): V, V+P and V+P_onto at lambda 0.5 and 1.0."""
    D = load_ds(ds)
    seeds = range(SET["seeds_main"])
    rows = [{"dataset": ds, "features": "V", "lambda": "-", **public(clu_summary(D, "V", seeds))}]
    for lam in (0.5, 1.0):
        for kd in ("V+P", "V+P_onto"):
            r = clu_summary(D, kd, seeds, lam, D["M"] if kd == "V+P_onto" else None)
            rows.append({"dataset": ds, "features": kd, "lambda": lam, **public(r)})
    for r in rows:
        log("  " + ", ".join(f"{k}={v}" for k, v in r.items()))
    write_csv(OUT / f"main_clustering_{ds}.csv", rows)


# 4. Random control
def run_random_control(ds):
    """Real ontology vs ontologies with the same family sizes but random membership.
    Class-level and retrieval metrics do not depend on the family definition (non-circular)."""
    D = load_ds(ds)
    rng = np.random.default_rng(2024)
    seeds = range(SET["seeds_sweep"])
    lam = LAMBDA_MAIN[ds]
    rows = []

    def evaluate(kind_label, draw, M, assign):
        m, _ = retrieval_eval(rr_scores(D, M), D["y"])
        tg = {"fam_true": D["yfam"]}
        if assign is not None:
            tg["fam_own"] = np.asarray(assign)[D["y"]]
        r = clu_summary(D, "V+P_onto", seeds, lam, M, fam_targets=tg)
        row = {"dataset": ds, "kind": kind_label, "draw": draw,
               "rr_P@1": round(m["P@1"], DEC), "rr_P@10": round(m["P@10"], DEC), "rr_mAP": round(m["mAP"], DEC),
               "ARI_class": round(r["ARI_class"], DEC), "NMI_class": round(r["NMI_class"], DEC),
               "ARI_fam_true": round(r["ARI_fam_true"], DEC),
               "ARI_fam_own": round(r.get("ARI_fam_own", r["ARI_fam_true"]), DEC),
               "families_agreement_ARI": round(ARI(D["fam_of_class"], assign), DEC) if assign is not None else 1.0}
        rows.append(row)
        log(f"  {kind_label:8s} #{draw:<3} rrP@10={row['rr_P@10']:.3f} ARIcls={row['ARI_class']:.3f} "
            f"ARIfam(true)={row['ARI_fam_true']:.3f} ARIfam(own)={row['ARI_fam_own']:.3f}")

    evaluate("real", 0, D["M"], D["fam_of_class"])
    # references without an ontology (identity M / raw profiles) and visual-only
    m, _ = retrieval_eval(rr_scores(D, np.eye(D["nc"], dtype=np.float32)), D["y"])
    r = clu_summary(D, "V+P", seeds, lam)
    rows.append({"dataset": ds, "kind": "no_ontology", "draw": 0, "rr_P@1": round(m["P@1"], DEC),
                 "rr_P@10": round(m["P@10"], DEC), "rr_mAP": round(m["mAP"], DEC),
                 "ARI_class": round(r["ARI_class"], DEC), "NMI_class": round(r["NMI_class"], DEC),
                 "ARI_fam_true": round(r["ARI_fam"], DEC), "ARI_fam_own": "", "families_agreement_ARI": ""})
    for i in range(SET["R_random"]):
        a = D["fam_of_class"][rng.permutation(D["nc"])]
        evaluate("random", i + 1, build_M(a), a)
    write_csv(OUT / f"random_control_{ds}.csv", rows)
    real = rows[0]
    rand = [r for r in rows if r["kind"] == "random"]
    log(f"\n  summary [{ds}]: real vs {len(rand)} random ontologies (p = share of random >= real)")
    for k in ("rr_P@1", "rr_P@10", "rr_mAP", "ARI_class", "ARI_fam_true", "ARI_fam_own"):
        v = np.array([r[k] for r in rand], float)
        p = (1 + np.sum(v >= real[k])) / (len(v) + 1)
        log(f"    {k:13s} real={real[k]:.4f}  random={v.mean():.4f}+/-{v.std():.4f}  p={p:.3f}")


# 5. Perturbation
def run_perturbation(ds):
    D = load_ds(ds)
    rng = np.random.default_rng(7)
    seeds = range(SET["seeds_perturb"])
    lam = LAMBDA_MAIN[ds]
    base = D["fam_of_class"]
    rows = []

    def measure(k, draw, assign):
        M = build_M(assign)
        m, _ = retrieval_eval(rr_scores(D, M), D["y"])
        r = clu_summary(D, "V+P_onto", seeds, lam, M, levels=("family",))
        rows.append({"dataset": ds, "n_reassigned": k, "draw": draw, "rr_P@10": round(m["P@10"], DEC),
                     "rr_mAP": round(m["mAP"], DEC), "ARI_fam_true": round(r["ARI_fam"], DEC)})
        log(f"  k={k:<3} #{draw} rrP@10={m['P@10']:.3f} ARIfam={r['ARI_fam']:.3f}")

    measure(0, 0, base)
    for k in SET["k_perturb"][ds]:
        for d in range(SET["R_perturb"]):
            while True:
                a = base.copy()
                for i in rng.choice(D["nc"], size=k, replace=False):
                    a[i] = rng.choice([f for f in range(D["nf"]) if f != base[i]])
                if len(np.unique(a)) == D["nf"]:
                    break
            measure(k, d + 1, a)
    write_csv(OUT / f"perturbation_{ds}.csv", rows)


# 6. Alternative assignments
def run_alt_assign(ds):
    if ds != "dtd":
        return
    D = load_ds(ds)
    seeds = range(SET["seeds_main"])
    lam = LAMBDA_MAIN[ds]
    fam_idx = {f: i for i, f in enumerate(D["fam_names"])}
    top10 = np.argsort(-vis_scores(D), axis=1)[:, :10]
    y = D["y"]
    variants = [("original", {})] + [(a, {a: f}) for a, f in ALTERNATIVES.items()] + [("all five", dict(ALTERNATIVES))]
    m_id, _ = retrieval_eval(rr_scores(D, np.eye(D["nc"], dtype=np.float32)), y)
    rows = []
    for vname, change in variants:
        assign = D["fam_of_class"].copy()
        for attr, fam in change.items():
            assign[D["names"].index(attr)] = fam_idx[fam]
        yf = assign[y]
        wrong = y[top10] != y[:, None]
        cross = wrong & (yf[top10] != yf[:, None])
        M = build_M(assign)
        m, _ = retrieval_eval(rr_scores(D, M), y)
        tg = {"alt": yf}
        rV = clu_summary(D, "V", seeds, fam_targets=tg, levels=("family",))
        rP = clu_summary(D, "V+P", seeds, lam, fam_targets=tg, levels=("family",))
        rO = clu_summary(D, "V+P_onto", seeds, lam, M, fam_targets=tg)
        row = {"variant": vname, "moved_to": "; ".join(f"{a}->{f}" for a, f in change.items()) or "-",
               "cross_family_share": round(float(cross.sum() / wrong.sum()), DEC),
               "P@10_ceiling": round(float(((~wrong) | cross).mean()), DEC),
               "rr_identity_P@10": round(m_id["P@10"], DEC), "rr_ontology_P@1": round(m["P@1"], DEC),
               "rr_ontology_P@10": round(m["P@10"], DEC), "rr_ontology_mAP": round(m["mAP"], DEC),
               "ARI_fam_V": round(rV["ARI_alt"], DEC), "ARI_fam_VP": round(rP["ARI_alt"], DEC),
               "ARI_fam_VPO": round(rO["ARI_alt"], DEC), "ARI_fam_VPO_sd": round(rO["ARI_alt_sd"], DEC),
               "ARI_class_VPO": round(rO["ARI_class"], DEC)}
        rows.append(row)
        log("  " + ", ".join(f"{k}={v}" for k, v in row.items()))
    write_csv(OUT / f"alt_assign_{ds}.csv", rows)


# 7. Family weight
def run_family_weight(ds):
    D = load_ds(ds)
    seeds = range(SET["seeds_sweep"])
    lam = LAMBDA_MAIN[ds]
    rows = []
    for w in SET["weights"]:
        M = build_M(D["fam_of_class"], w)
        m, _ = retrieval_eval(rr_scores(D, M), D["y"])
        r = clu_summary(D, "V+P_onto", seeds, lam, M)
        rows.append({"dataset": ds, "w": w, **{f"rr_{k}": round(v, DEC) for k, v in m.items()},
                     "ARI_fam": round(r["ARI_fam"], DEC), "ARI_fam_sd": round(r["ARI_fam_sd"], DEC),
                     "ARI_class": round(r["ARI_class"], DEC), "ARI_class_sd": round(r["ARI_class_sd"], DEC)})
        log(f"  w={w:<4} rrP@10={m['P@10']:.3f} rrmAP={m['mAP']:.3f} ARIfam={r['ARI_fam']:.3f} ARIcls={r['ARI_class']:.3f}")
    write_csv(OUT / f"family_weight_{ds}.csv", rows)


# 8. Automatic ontology
def agglo(T, k, linkage="average"):
    from sklearn.cluster import AgglomerativeClustering
    if linkage == "ward":   # Ward needs Euclidean; T is L2-normalised, so this tracks cosine
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(T)
    try:
        ag = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage=linkage)
    except TypeError:
        ag = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage=linkage)
    return ag.fit_predict(T)


def run_auto_ontology(ds):
    """Families obtained automatically by clustering the text embeddings of the class names
    (no images, no labels). Compared with the hand-curated ontology."""
    D = load_ds(ds)
    seeds = range(SET["seeds_sweep"])
    lam = LAMBDA_MAIN[ds]
    rows, fams = [], {}
    cands = [("hand", D["fam_of_class"])]
    from sklearn.cluster import KMeans
    for tv in VLMS:
        T = normalize(text_emb(D, tv, "bare"))
        # average linkage (cosine) can give one giant group plus singletons, so two
        # methods that favour balanced groups are added: Ward linkage and K-means.
        cands.append((f"auto-{tv}-avg", agglo(T, D["nf"])))
        cands.append((f"auto-{tv}-ward", agglo(T, D["nf"], linkage="ward")))
        cands.append((f"auto-{tv}-kmeans", KMeans(D["nf"], n_init=50, random_state=0).fit_predict(T)))
    for name, assign in cands:
        M = build_M(assign)
        m, _ = retrieval_eval(rr_scores(D, M), D["y"])
        r = clu_summary(D, "V+P_onto", seeds, lam, M,
                        fam_targets={"fam_hand": D["yfam"], "fam_own": np.asarray(assign)[D["y"]]})
        sizes = sorted(np.bincount(np.asarray(assign)).tolist(), reverse=True)
        rows.append({"dataset": ds, "ontology": name,
                     "family_sizes": "-".join(str(s) for s in sizes if s > 0),
                     "agreement_with_hand_ARI": round(ARI(D["fam_of_class"], assign), DEC),
                     "agreement_with_hand_NMI": round(NMI(D["fam_of_class"], assign), DEC),
                     **{f"rr_{k}": round(v, DEC) for k, v in m.items()},
                     "ARI_class": round(r["ARI_class"], DEC), "ARI_fam_hand": round(r["ARI_fam_hand"], DEC),
                     "ARI_fam_own": round(r["ARI_fam_own"], DEC)})
        fams[name] = {str(g): [D["names"][i] for i in np.where(np.asarray(assign) == g)[0]]
                      for g in np.unique(assign)}
        log("  " + ", ".join(f"{k}={v}" for k, v in rows[-1].items()))
    write_csv(OUT / f"auto_ontology_{ds}.csv", rows)
    with open(OUT / f"auto_families_{ds}.json", "w") as f:
        json.dump(fams, f, indent=1)
    log(f"  -> saved {OUT / f'auto_families_{ds}.json'} (family members)")


# 9. KTH-TIPS2-b sample-split protocol
def run_kth_fair(ds):
    if ds != "kth":
        return
    D = load_ds(ds)
    valid = same_sample_mask(D)
    rows = []
    for bb in ("dinov2", "clip", "siglip"):
        X = normalize(D["img"][bb])
        S = X @ X.T
        np.fill_diagonal(S, -np.inf)
        for proto, vm in (("standard", None), ("fair", valid)):
            m, _ = retrieval_eval(S, D["y"], valid=vm)
            acc, facc = knn_eval(S, D["y"], D["fam_of_class"], 5, vm)
            rows.append({"kind": "visual-only", "backbone": bb, "M": "-", "alpha": "-", "protocol": proto,
                         **public(m), "kNN": round(acc, DEC), "kNN_family": round(facc, DEC)})
            log(f"  {bb:6s} {proto:8s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()) + f" kNN={acc:.3f}")
    for vlm in VLMS:
        for a in SET["alphas"]:
            for tag, M in (("identity", np.eye(D["nc"], dtype=np.float32)), ("ontology", D["M"])):
                m, _ = retrieval_eval(rr_scores(D, M, a, vlm), D["y"], valid=valid)
                rows.append({"kind": "re-rank", "backbone": f"dinov2+{vlm}", "M": tag, "alpha": a,
                             "protocol": "fair", **public(m)})
                log(f"  fair {vlm:6s} a={a:.1f} {tag:8s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
    write_csv(OUT / "kth_fair.csv", rows)


# 10. Statistics
def boot_macro(d, y, B, rng):
    """Stratified (per-class) bootstrap of a macro-averaged per-query difference."""
    idxs = [np.where(y == c)[0] for c in np.unique(y)]
    est = float(np.mean([np.nanmean(d[i]) for i in idxs]))
    acc = np.zeros(B)
    for ix in idxs:
        r = rng.integers(0, len(ix), size=(B, len(ix)))
        acc += np.nanmean(d[ix][r], axis=1)
    bs = acc / len(idxs)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2 * min(float((bs <= 0).mean()), float((bs >= 0).mean())))
    return est, float(lo), float(hi), p


def mcnemar(a, b):
    from scipy.stats import binomtest
    n01 = int(np.sum(a & ~b)); n10 = int(np.sum(~a & b))
    n = n01 + n10
    return 1.0 if n == 0 else float(binomtest(min(n01, n10), n, 0.5).pvalue)


def run_bootstrap(ds):
    from scipy.stats import wilcoxon, t as tdist
    D = load_ds(ds)
    rng = np.random.default_rng(11)
    B = SET["B_boot"]
    y = D["y"]
    rows = []
    protos = [("standard", None)] + ([("fair", same_sample_mask(D))] if ds == "kth" else [])
    for proto, vm in protos:
        _, pv = retrieval_eval(vis_scores(D), y, valid=vm)
        _, pi = retrieval_eval(rr_scores(D, np.eye(D["nc"], dtype=np.float32)), y, valid=vm)
        _, po = retrieval_eval(rr_scores(D, D["M"]), y, valid=vm)
        for cname, A, Bm in (("ontology - identity", po, pi), ("ontology - visual", po, pv),
                             ("identity - visual", pi, pv)):
            for met in ("P@1", "P@10", "mAP"):
                est, lo, hi, p = boot_macro(A[met] - Bm[met], y, B, rng)
                rows.append({"dataset": ds, "test": f"retrieval ({proto}) SigLIP a=0.7", "comparison": cname,
                             "metric": met, "estimate": round(est, DEC), "ci_low": round(lo, DEC),
                             "ci_high": round(hi, DEC), "p": round(p, DEC), "method": f"stratified bootstrap B={B}"})
    for vlm in VLMS:
        pb = profiles(D, vlm, "bare").argmax(1); pr = profiles(D, vlm, "rich").argmax(1)
        for lvl, cb, cr in (("exact", pb == y, pr == y),
                            ("family", D["fam_of_class"][pb] == D["yfam"], D["fam_of_class"][pr] == D["yfam"])):
            est, lo, hi, _ = boot_macro(cr.astype(float) - cb.astype(float), np.zeros_like(y), B, rng)
            rows.append({"dataset": ds, "test": f"zero-shot {vlm}", "comparison": "enriched - bare",
                         "metric": f"{lvl} accuracy", "estimate": round(est, DEC), "ci_low": round(lo, DEC),
                         "ci_high": round(hi, DEC), "p": round(mcnemar(cr, cb), 6), "method": "bootstrap CI; exact McNemar p"})
    seeds = range(SET["seeds_main"])
    lam = LAMBDA_MAIN[ds]
    r = {k: clu_summary(D, k, seeds, None if k == "V" else lam, D["M"] if k == "V+P_onto" else None)
         for k in ("V", "V+P", "V+P_onto")}
    for a_, b_ in (("V+P_onto", "V+P"), ("V+P_onto", "V"), ("V+P", "V")):
        for lvl in ("fam", "class"):
            x, z = r[a_][f"_ARI_{lvl}"], r[b_][f"_ARI_{lvl}"]
            d = x - z
            h = tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            try:
                p = float(wilcoxon(x, z).pvalue)
            except ValueError:
                p = 1.0
            rows.append({"dataset": ds, "test": f"clustering (lambda={lam}, {len(d)} seeds)",
                         "comparison": f"{a_} - {b_}", "metric": f"ARI {'family' if lvl == 'fam' else 'class'}",
                         "estimate": round(float(d.mean()), DEC), "ci_low": round(float(d.mean() - h), DEC),
                         "ci_high": round(float(d.mean() + h), DEC), "p": round(p, DEC),
                         "method": "paired over seeds: t-CI; Wilcoxon p"})
    for rw in rows:
        log(f"  {rw['test']:38s} {rw['comparison']:20s} {rw['metric']:16s} {rw['estimate']:+.4f} "
            f"[{rw['ci_low']:+.4f}, {rw['ci_high']:+.4f}] p={rw['p']}")
    write_csv(OUT / f"bootstrap_{ds}.csv", rows)


# 11. Classification
def split_feats(D, kind, lam, M=None, vlm=PRIMARY, pkind="rich"):
    tr, te = D["tr"], D["te"]
    sc = StandardScaler().fit(D["V"][tr])
    Xtr, Xte = [sc.transform(D["V"][tr])], [sc.transform(D["V"][te])]
    if kind != "V":
        P = profiles(D, vlm, pkind)
        Q = P if kind == "V+P" else smooth(P, M)
        s2 = StandardScaler().fit(Q[tr])
        Xtr.append(lam * s2.transform(Q[tr]))
        Xte.append(lam * s2.transform(Q[te]))
    return np.hstack(Xtr), np.hstack(Xte)


def run_classification(ds):
    from sklearn.linear_model import LogisticRegression
    D = load_ds(ds)
    y, yfl = D["y"], D["fam_of_class"]
    rows = []
    configs = [("V", None)] + [(k, l) for l in sorted({0.5, 1.0, LAMBDA_MAIN[ds]}) for k in ("V+P", "V+P_onto")]
    for kind, lam in configs:
        M = D["M"] if kind == "V+P_onto" else None
        Xtr, Xte = split_feats(D, kind, lam, M)
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, y[D["tr"]])
        pr = clf.predict(Xte)
        X = zfeat(D, kind, lam, M=M)
        sq = (X * X).sum(1)
        S = 2 * (X @ X.T) - sq[:, None] - sq[None, :]
        np.fill_diagonal(S, -np.inf)
        acc, facc = knn_eval(S, y, yfl, 5)
        row = {"dataset": ds, "features": kind, "lambda": lam if lam is not None else "-",
               "probe_exact": round(float((pr == y[D["te"]]).mean()), DEC),
               "probe_family": round(float((yfl[pr] == yfl[y[D["te"]]]).mean()), DEC),
               "knn_exact": round(acc, DEC), "knn_family": round(facc, DEC)}
        if ds == "kth":
            a2, f2 = knn_eval(S, y, yfl, 5, same_sample_mask(D))
            row.update(knn_fair_exact=round(a2, DEC), knn_fair_family=round(f2, DEC))
        rows.append(row)
        log("  " + ", ".join(f"{k}={v}" for k, v in row.items()))
    write_csv(OUT / f"classification_{ds}.csv", rows)


# 12. Per-class / per-family
def run_per_family(ds):
    D = load_ds(ds)
    y, yf = D["y"], D["yfam"]
    top10 = np.argsort(-vis_scores(D), axis=1)[:, :10]
    wrong = y[top10] != y[:, None]
    cross = wrong & (yf[top10] != yf[:, None])
    _, pv = retrieval_eval(vis_scores(D), y)
    _, pi = retrieval_eval(rr_scores(D, np.eye(D["nc"], dtype=np.float32)), y)
    _, po = retrieval_eval(rr_scores(D, D["M"]), y)
    P = profiles(D, PRIMARY, "rich").argmax(1)
    rows = []
    for c in range(D["nc"]):
        s = y == c
        rows.append({"dataset": ds, "class": D["names"][c], "family": D["fam_names"][D["fam_of_class"][c]],
                     "cross_family_share_top10": round(float(cross[s].sum() / (s.sum() * 10)), DEC),
                     "error_share_top10": round(float(wrong[s].sum() / (s.sum() * 10)), DEC),
                     "P@10_visual": round(float(pv["P@10"][s].mean()), DEC),
                     "P@10_identity": round(float(pi["P@10"][s].mean()), DEC),
                     "P@10_ontology": round(float(po["P@10"][s].mean()), DEC),
                     "zs_siglip_rich_exact": round(float((P[s] == c).mean()), DEC)})
    write_csv(OUT / f"per_class_{ds}.csv", rows)
    frows = []
    for fi, fn in enumerate(D["fam_names"]):
        rs = [r for r in rows if r["family"] == fn]
        frows.append({"dataset": ds, "family": fn, "n_classes": len(rs),
                      **{k: round(float(np.mean([r[k] for r in rs])), DEC)
                         for k in ("cross_family_share_top10", "P@10_visual", "P@10_identity", "P@10_ontology")}})
        f = frows[-1]
        log(f"  {fn:20s} cross={f['cross_family_share_top10']:.3f} P@10 vis={f['P@10_visual']:.3f} "
            f"id={f['P@10_identity']:.3f} onto={f['P@10_ontology']:.3f}")
    write_csv(OUT / f"per_family_{ds}.csv", frows)


# 13. Silhouette
def run_silhouette(ds):
    from sklearn.metrics import silhouette_score
    D = load_ds(ds)
    lam = LAMBDA_MAIN[ds]
    feats = {"V": zfeat(D, "V"), "V+P": zfeat(D, "V+P", lam), "V+P_onto": zfeat(D, "V+P_onto", lam, M=D["M"])}
    sil = []
    for k, X in feats.items():
        sil.append({"dataset": ds, "features": k,
                    "silhouette_family": round(float(silhouette_score(X, D["yfam"], random_state=0,
                                                                      sample_size=min(4000, len(X)))), DEC),
                    "silhouette_class": round(float(silhouette_score(X, D["y"], random_state=0,
                                                                     sample_size=min(4000, len(X)))), DEC)})
        log(f"  silhouette {k:9s} family={sil[-1]['silhouette_family']:.4f} class={sil[-1]['silhouette_class']:.4f}")
    write_csv(OUT / f"silhouette_{ds}.csv", sil)


# 14. Qualitative
def run_qualitative(ds):
    if ds != "dtd":
        return
    D = load_ds(ds)
    y, yf = D["y"], D["yfam"]
    tv = np.argsort(-vis_scores(D), axis=1)[:, :5]
    to = np.argsort(-rr_scores(D, D["M"]), axis=1)[:, :5]
    p5v = (y[tv] == y[:, None]).mean(1)
    p5o = (y[to] == y[:, None]).mean(1)
    xv = ((yf[tv] != yf[:, None])).sum(1)
    xo = ((yf[to] != yf[:, None])).sum(1)
    gain = p5o - p5v
    # Selection rule: the 3 largest P@5 gains from distinct classes,
    # ties broken by more cross-family intruders removed, then by image index; plus the
    # single largest P@5 loss (a failure case).
    order = sorted(range(len(y)), key=lambda i: (-gain[i], -(xv[i] - xo[i]), i))
    picked, used = [], set()
    for i in order:
        if gain[i] <= 0:
            break
        if y[i] not in used:
            picked.append(("improved", i)); used.add(y[i])
        if len(picked) == 3:
            break
    worst = sorted(range(len(y)), key=lambda i: (gain[i], i))[0]
    if gain[worst] < 0:
        picked.append(("degraded", worst))
    rows = []
    for kind, q in picked:
        rows.append({"case": kind, "query": int(q), "query_class": D["names"][y[q]],
                     "P@5_visual": round(float(p5v[q]), 2), "P@5_ontology": round(float(p5o[q]), 2),
                     "cross_family_visual": int(xv[q]), "cross_family_ontology": int(xo[q]),
                     "top5_visual": " ".join(map(str, tv[q])), "top5_ontology": " ".join(map(str, to[q]))})
        log("  " + ", ".join(f"{k}={v}" for k, v in rows[-1].items() if not k.startswith("top5")))
    write_csv(OUT / "qualitative_dtd.csv", rows)


# Driver
EXPERIMENTS = {
    "sanity":          (run_sanity, ("dtd", "kth"), "sanity_{ds}"),
    "alpha_sweep":     (run_alpha_sweep, ("dtd", "kth"), "alpha_sweep_{ds}"),
    "profile_source":  (run_profile_source, ("dtd", "kth"), "profile_source_{ds}"),
    "lambda_sweep":    (run_lambda_sweep, ("dtd", "kth"), "lambda_sweep_{ds}"),
    "main_clustering": (run_main_clustering, ("dtd", "kth"), "main_clustering_{ds}"),
    "random_control":  (run_random_control, ("dtd", "kth"), "random_control_{ds}"),
    "perturbation":    (run_perturbation, ("dtd", "kth"), "perturbation_{ds}"),
    "alt_assign":      (run_alt_assign, ("dtd",), "alt_assign_{ds}"),
    "family_weight":   (run_family_weight, ("dtd", "kth"), "family_weight_{ds}"),
    "auto_ontology":   (run_auto_ontology, ("dtd", "kth"), "auto_ontology_{ds}"),
    "kth_fair":        (run_kth_fair, ("kth",), "kth_fair"),
    "bootstrap":       (run_bootstrap, ("dtd", "kth"), "bootstrap_{ds}"),
    "classification":  (run_classification, ("dtd", "kth"), "classification_{ds}"),
    "per_family":      (run_per_family, ("dtd", "kth"), "per_family_{ds}"),
    "silhouette":      (run_silhouette, ("dtd", "kth"), "silhouette_{ds}"),
    "qualitative":     (run_qualitative, ("dtd",), "qualitative_dtd"),
}
QUICK_MODE = False


def main():
    global OUT, PRED, QUICK_MODE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="sanity", help="experiment name, comma-separated list, or 'all'")
    ap.add_argument("--dataset", default="both", choices=("dtd", "kth", "both"))
    ap.add_argument("--out", default=str(Path(config.RESULTS_DIR) / "ext"))
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test only")
    ap.add_argument("--resume", action="store_true", help="skip experiments whose CSV exists")
    ap.add_argument("--cache", default=None, help="existing K-means label cache to reuse (default: <out>/kmeans_cache)")
    a = ap.parse_args()
    if a.quick:
        SET.update(QUICK)
        QUICK_MODE = True
        log("Quick mode: smoke-test settings, not the settings of the reported results")
    OUT = Path(a.out)
    PRED = Path(a.cache) if a.cache else OUT / "kmeans_cache"
    for d in (OUT, PRED):
        d.mkdir(parents=True, exist_ok=True)
    exps = list(EXPERIMENTS) if a.exp == "all" else [e.strip() for e in a.exp.split(",")]
    for e in exps:
        if e not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment '{e}'. Choose from: {', '.join(EXPERIMENTS)}")
    dss = ("dtd", "kth") if a.dataset == "both" else (a.dataset,)
    for e in exps:
        run, allowed, csvname = EXPERIMENTS[e]
        for ds in dss:
            if ds not in allowed:
                continue
            if a.resume and (OUT / (csvname.format(ds=ds) + ".csv")).exists():
                log(f"[skip] {e} [{ds}] (CSV exists)")
                continue
            t0 = time.time()
            log(f"\n##### {e} [{ds}] #####")
            run(ds)
            log(f"##### {e} [{ds}] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
