"""Hand-curated material ontology for KTH-TIPS2-b.

11 materials grouped into 4 material families. Same interface as ontology.py (DTD),
so the same pipeline runs unchanged. Names are normalised (lowercase, underscores)
to match the dataset folder names.

Borderline assignments:
  - aluminium_foil is a singleton family (reflective metal).
  - cork could sit with FABRIC (soft) or PLANT_ORGANIC (natural); placed in PLANT_ORGANIC.
"""
import numpy as np

def _norm(s):
    return s.strip().lower().replace(" ", "_").replace("-", "_")

FAMILIES = {
    "FABRIC":        ["corduroy", "cotton", "linen", "wool"],
    "BAKED_FOOD":    ["brown_bread", "white_bread", "cracker"],
    "PLANT_ORGANIC": ["cork", "wood", "lettuce_leaf"],
    "METAL":         ["aluminium_foil"],
}

DESCRIPTIONS = {
    "aluminium_foil": "crinkled reflective aluminium foil, shiny metallic surface",
    "brown_bread":    "porous brown bread crust and crumb",
    "white_bread":    "soft porous white bread texture",
    "cracker":        "dry crisp cracker with small holes",
    "corduroy":       "ribbed corduroy fabric with parallel raised cords",
    "cotton":         "woven cotton fabric, soft matte weave",
    "linen":          "woven linen fabric with a visible thread grid",
    "wool":           "fuzzy knitted wool fibres",
    "cork":           "granular cork surface with small speckles",
    "wood":           "natural wood grain with directional lines",
    "lettuce_leaf":   "veined green lettuce leaf surface",
}

ATTR_TO_FAMILY = {a: f for f, attrs in FAMILIES.items() for a in attrs}
ALL_ATTRS = sorted(ATTR_TO_FAMILY)
N_FAMILIES = len(FAMILIES)

def _check():
    assert len(ATTR_TO_FAMILY) == 11, f"expected 11, got {len(ATTR_TO_FAMILY)}"

def similarity_matrix(class_names=None, cross_family=0.0):
    """(11,11) attribute-similarity matrix aligned to class_names (normalised)."""
    names = [_norm(c) for c in class_names] if class_names is not None else ALL_ATTRS
    for n in names:
        if n not in ATTR_TO_FAMILY:
            raise KeyError(f"class '{n}' not in KTH ontology; update FAMILIES.")
    k = len(names)
    M = np.zeros((k, k), dtype=np.float32)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = 1.0 if a == b else (0.5 if ATTR_TO_FAMILY[a] == ATTR_TO_FAMILY[b] else cross_family)
    return M, names

def prompts(class_names=None, template="a {name} texture, {desc}"):
    names = [_norm(c) for c in class_names] if class_names is not None else ALL_ATTRS
    return [template.format(name=a.replace("_", " "), desc=DESCRIPTIONS[a]) for a in names]

_check()
