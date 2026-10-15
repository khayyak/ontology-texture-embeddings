# When Does an Attribute Ontology Help Frozen Foundation-Model Embeddings?

Code and ontologies for the article
*When Does an Attribute Ontology Help Frozen Foundation-Model Embeddings? Evidence from Texture Retrieval, Classification, and Clustering*
(El Kharrachi Khayya, Ahmed El Oirrak; submitted to *Array*).

Frozen backbones (DINOv2 ViT-B/14, CLIP ViT-B/32, SigLIP ViT-B/16) are combined with a hand-curated
texture-attribute ontology in three training-free ways: ontology-enriched prompts, retrieval
re-ranking, and ontology-guided clustering. The ontology is applied only to the models' predicted
attribute profiles, never to ground-truth labels.

## Contents

| File | What it does |
|---|---|
| `src/ontology.py` | DTD ontology: 47 attributes in 8 families, attribute descriptions, similarity matrix M |
| `src/ontology_kth.py` | KTH-TIPS2-b ontology: 11 materials in 4 families |
| `src/config.py` | Paths, backbones and shared settings |
| `src/data.py`, `src/data_kth.py` | Dataset loaders (DTD partition 1; KTH-TIPS2-b, train on sample a, test on b/c/d) |
| `src/embeddings.py` | Frozen image and text encoders, with embedding caching |
| `src/evaluate.py` | Retrieval, kNN, linear-probe and clustering metrics |
| `src/rerank.py` | Ontology-guided re-ranking: alpha * visual + (1 - alpha) * p_q^T M p_c |
| `src/run_baselines.py`, `src/run_baselines_kth.py` | Extract and cache embeddings; baseline results |
| `src/run_rerank.py`, `src/run_rerank_kth.py` | Zero-shot prompts (bare vs enriched) and re-ranking |
| `src/run_cluster.py`, `src/run_cluster_kth.py` | Ontology-guided clustering (V, V+P, V+P_onto) |
| `src/diagnose.py` | Within-family vs cross-family errors of DINOv2 retrieval on DTD |
| `src/run_extensions.py` | All further experiments and statistical tests (see below) |

## Datasets (not redistributed)

- DTD: https://www.robots.ox.ac.uk/~vgg/data/dtd/ (official partition 1)
- KTH-TIPS2-b: https://www.csc.kth.se/cvap/databases/kth-tips/ (`kth-tips2-b_col_200x200.tar`)

DTD is downloaded automatically by torchvision into `src/data/` on the first run. KTH-TIPS2-b must be
extracted so that `src/data/KTH-TIPS2-b/<material>/sample_a/...` exists. All scripts are run from
inside `src/` (the folders `data/`, `cache/` and `results/` are created there).

## Environment

The reported runs used an NVIDIA Tesla T4 GPU with Python 3.13, PyTorch 2.11.0, open_clip 3.3.0,
scikit-learn 1.6.1, NumPy 2.1.3 and SciPy 1.16.3.

```
pip install -r requirements.txt
```

## Running

From the `src/` folder:

```
python run_baselines.py               # DTD: extract and cache embeddings, baseline results
python run_baselines_kth.py           # KTH-TIPS2-b: same
python run_rerank.py --vlm siglip     # also --vlm clip; *_kth.py for KTH-TIPS2-b
python run_cluster.py --vlm siglip --seeds 10
python diagnose.py
python run_extensions.py --exp sanity
python run_extensions.py --exp all --out results/ext
```

`--exp sanity` checks that the cached embeddings reproduce the main results and should print
`ALL PASS` before the other experiments are run. `run_extensions.py` reads only the cached embeddings.
K-means labels are cached in `<out>/kmeans_cache` and can be reused with `--cache`. Use
`--dataset dtd` or `--dataset kth` to run one dataset, and `--resume` to skip finished experiments.

## Experiments in `run_extensions.py`

| `--exp` | Content | In the article |
|---|---|---|
| `profile_source` | Bare vs ontology-enriched prompts as the profile source | Table 2, Fig. 4 |
| `alpha_sweep` | Re-ranking vs alpha, identity vs ontology matrix, CLIP and SigLIP | Table 3, Fig. 5 |
| `kth_fair` | KTH-TIPS2-b retrieval and kNN under the sample-split protocol | Tables 1 and 3 |
| `main_clustering` | Clustering with V, V+P and V+P_onto (10 seeds) | Table 4 |
| `lambda_sweep` | Clustering vs lambda | Fig. 6 |
| `bootstrap` | Stratified bootstrap CIs, exact McNemar and Wilcoxon tests | Table 5 |
| `classification` | Linear probe and kNN on the fused features | Table 6 |
| `random_control` | Real ontology vs 20 size-matched random ontologies | Table 7, Fig. 7 |
| `perturbation` | Randomly re-assigned attributes | Fig. 8 |
| `alt_assign` | Borderline DTD attributes moved to an alternative family | Table 8, Fig. 3 |
| `family_weight` | Same-family weight w in M | Fig. 9 |
| `auto_ontology` | Families derived automatically from text embeddings | Table 9 |
| `per_family` | Per-class and per-family breakdown | Fig. 10 |
| `silhouette` | Family- and class-level silhouette coefficients | Section 6.5 |
| `qualitative` | Selection of the DTD retrieval examples | Fig. 11 |

## Notes

- KTH-TIPS2-b sample-split retrieval: the gallery of a query from sample index s contains only images
  of the other three sample indices.
- `run_extensions.py` writes its results with 6 decimals; the baseline, re-ranking and clustering
  scripts write 4.

## License

MIT (see `LICENSE`).
