"""KTH-TIPS2-b loader.

KTH-TIPS2-b layout after extraction:
    <root>/KTH-TIPS2-b/<material>/sample_<a|b|c|d>/*.png
11 materials x 4 physical samples x 108 images = 4752 images.

Interface mirrors data.py (DTD):
  load_all()                  -> (paths, labels, class_names)  [whole set]
  load_classification_splits()-> (train, test, class_names)
Standard KTH-TIPS2-b protocol: train on ONE sample, test on the other three.
We use sample 'a' as train and b/c/d as test.
"""
import os, glob
KTH_DIRNAME = "KTH-TIPS2-b"

def _find_root(root="data"):
    # accept either <root>/KTH-TIPS2-b/... or a direct path to it
    for cand in (os.path.join(root, KTH_DIRNAME), root):
        if os.path.isdir(cand) and any(
            os.path.isdir(os.path.join(cand, d)) for d in os.listdir(cand)):
            # heuristic: material dirs contain sample_* subdirs
            for d in os.listdir(cand):
                p = os.path.join(cand, d)
                if os.path.isdir(p) and glob.glob(os.path.join(p, "sample_*")):
                    return cand
    raise FileNotFoundError(
        f"Could not find {KTH_DIRNAME} under '{root}'. Extract the dataset so that "
        f"'{root}/{KTH_DIRNAME}/<material>/sample_a/...' exists.")

def _norm(s):
    return s.strip().lower().replace(" ", "_").replace("-", "_")

def _scan(root):
    base = _find_root(root)
    materials = sorted(d for d in os.listdir(base)
                       if os.path.isdir(os.path.join(base, d)))
    class_names = [_norm(m) for m in materials]
    name_to_id = {n: i for i, n in enumerate(class_names)}
    records = []  # (path, label, sample_letter)
    for m in materials:
        for sample_dir in sorted(glob.glob(os.path.join(base, m, "sample_*"))):
            letter = os.path.basename(sample_dir).split("_")[-1].lower()
            for img in sorted(glob.glob(os.path.join(sample_dir, "*"))):
                if img.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    records.append((img, name_to_id[_norm(m)], letter))
    if not records:
        raise RuntimeError("No images found under KTH-TIPS2-b.")
    return records, class_names

def load_all(root="data"):
    recs, class_names = _scan(root)
    paths  = [r[0] for r in recs]
    labels = [r[1] for r in recs]
    return paths, labels, class_names

def load_classification_splits(root="data", train_sample="a"):
    recs, class_names = _scan(root)
    tr = [(p, y) for (p, y, s) in recs if s == train_sample]
    te = [(p, y) for (p, y, s) in recs if s != train_sample]
    tr_p, tr_y = [x[0] for x in tr], [x[1] for x in tr]
    te_p, te_y = [x[0] for x in te], [x[1] for x in te]
    return (tr_p, tr_y), (te_p, te_y), class_names
