"""Hand-curated ontology over the 47 DTD texture attributes.
2-level taxonomy root -> family -> attribute; 8 perceptual families.

Borderline assignments (alternatives tested by run_extensions.py --exp alt_assign):
  honeycombed (net vs holes -> GRID_NET), marbled (veined vs swirly -> CRACKED_VEINED),
  zigzagged (-> LINEAR_DIRECTIONAL), cobwebbed (-> FIBROUS_SOFT), scaly (-> BUMPY_RELIEF).
"""
import numpy as np

FAMILIES = {
    "LINEAR_DIRECTIONAL": ["banded", "striped", "lined", "grooved", "pleated",
                            "stratified", "zigzagged"],
    "DOTTED_SPOTTED":     ["dotted", "polka-dotted", "freckled", "flecked",
                            "sprinkled", "studded"],
    "GRID_NET":           ["chequered", "grid", "crosshatched", "meshed", "woven",
                            "knitted", "waffled", "honeycombed", "interlaced",
                            "braided", "lacelike"],
    "HOLES_POROUS":       ["perforated", "porous", "pitted", "potholed"],
    "BUMPY_RELIEF":       ["bumpy", "bubbly", "blotchy", "scaly", "crystalline"],
    "CRACKED_VEINED":     ["cracked", "veined", "marbled", "smeared", "stained"],
    "SWIRLY_CURVED":      ["swirly", "spiralled", "paisley", "frilly"],
    "FIBROUS_SOFT":       ["fibrous", "matted", "gauzy", "wrinkled", "cobwebbed"],
}

DESCRIPTIONS = {
    "banded": "parallel horizontal bands, stripe-like directional lines",
    "blotchy": "irregular soft blobs and patches of varying size",
    "braided": "interwoven strands plaited in a regular braid pattern",
    "bubbly": "rounded raised bubbles and blob-like protrusions",
    "bumpy": "an uneven surface covered in small rounded bumps",
    "chequered": "a regular checkerboard grid of alternating squares",
    "cobwebbed": "fine thin threads forming an irregular web-like mesh",
    "cracked": "a network of irregular cracks and fissures",
    "crosshatched": "many crossing straight lines forming a hatch pattern",
    "crystalline": "faceted angular crystal-like facets and grains",
    "dotted": "scattered small round dots",
    "fibrous": "many thin fibres or hairs running through the surface",
    "flecked": "tiny irregular flecks and specks scattered about",
    "freckled": "small irregular spots sprinkled like freckles",
    "frilly": "wavy ruffled frilled edges and folds",
    "gauzy": "thin sheer translucent gauze-like weave",
    "grid": "a regular grid of straight crossing lines",
    "grooved": "parallel carved grooves and channels",
    "honeycombed": "a regular hexagonal honeycomb cell pattern",
    "interlaced": "strands laced over and under one another",
    "knitted": "interlocking knitted loops of yarn",
    "lacelike": "delicate open lace-like openwork pattern",
    "lined": "many thin parallel straight lines",
    "marbled": "swirled veins of colour like polished marble",
    "matted": "tangled matted fibres pressed together",
    "meshed": "an open mesh net of crossing threads",
    "paisley": "curved teardrop paisley motifs",
    "perforated": "a surface punctured with regular holes",
    "pitted": "small pits and depressions across the surface",
    "pleated": "regular folded parallel pleats",
    "polka-dotted": "evenly spaced regular round polka dots",
    "porous": "many small pores and cavities like a sponge",
    "potholed": "irregular large holes and hollows",
    "scaly": "overlapping scales like fish or reptile skin",
    "smeared": "smeared blurred streaks of blended material",
    "spiralled": "coiled spiral swirls",
    "sprinkled": "small particles sprinkled randomly",
    "stained": "irregular discoloured stains and blotches",
    "stratified": "horizontal layered strata bands",
    "striped": "bold parallel stripes",
    "studded": "regular raised studs or bumps",
    "swirly": "flowing curved swirling patterns",
    "veined": "branching vein-like lines",
    "waffled": "a regular waffle grid of raised squares",
    "woven": "interwoven warp and weft threads of a woven fabric",
    "wrinkled": "creased wrinkled folds",
    "zigzagged": "repeated angular zigzag lines",
}

ATTR_TO_FAMILY = {a: f for f, attrs in FAMILIES.items() for a in attrs}
ALL_ATTRS = sorted(ATTR_TO_FAMILY)

def _check():
    assert len(ATTR_TO_FAMILY) == 47, f"expected 47, got {len(ATTR_TO_FAMILY)}"

def similarity_matrix(class_names=None, cross_family=0.0):
    """(47,47) similarity aligned to class_names (defaults alphabetical == DTD ids)."""
    names = list(class_names) if class_names is not None else ALL_ATTRS
    n = len(names)
    M = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a == b:
                M[i, j] = 1.0
            elif ATTR_TO_FAMILY[a] == ATTR_TO_FAMILY[b]:
                M[i, j] = 0.5
            else:
                M[i, j] = cross_family
    return M, names

def prompts(class_names=None, template="a {name} texture, {desc}"):
    names = list(class_names) if class_names is not None else ALL_ATTRS
    return [template.format(name=a, desc=DESCRIPTIONS[a]) for a in names]

_check()
