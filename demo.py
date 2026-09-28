"""Run the audit pipeline on the example images and print the results.

    python demo.py [--images examples] [--work DIR]
"""
import argparse
import glob
import os
import tempfile

import numpy as np

from leafaudit import corners, interventions, segment, shape, splits

CONDITIONS = ["rgb", "gray", "crop", "crop_mask", "crop_leafout", "crop_noise",
              "crop_swap", "crop_occ", "silhouette"]


def cosine(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="examples")
    ap.add_argument("--work", default=None)
    args = ap.parse_args()
    work = args.work or tempfile.mkdtemp(prefix="leafaudit-")
    srcs = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    names = [os.path.splitext(os.path.basename(s))[0] for s in srcs]
    print(f"{len(srcs)} images from {args.images} · work directory {work}\n")

    print("[1] Segmentation and crop")
    crops = []
    for s, n in zip(srcs, names):
        dst = os.path.join(work, "crops", f"{n}.jpg")
        frac = segment.crop_leaf(s, dst, dst[:-4] + "_m.png")
        crops.append(dst if frac > 0 else None)
        print(f"    {n}  leaf fraction of crop {frac:.3f}")
    ok = [i for i, c in enumerate(crops) if c]
    print(f"    segmented {len(ok)}/{len(srcs)}\n")

    print("[2] Leaf measurements (feature baseline)")
    cols = ["area_frac", "aspect", "solidity", "n_parts",
            "lightness", "greenness", "saturation"]
    print("    " + "image".ljust(10) + "".join(c.rjust(12) for c in cols))
    for s, n in zip(srcs, names):
        m = shape.measure(s)
        vals = "".join(str(m[c]).rjust(12) for c in cols) if m["ok"] else "  failed"
        print(f"    {n.ljust(10)}{vals}")
    print()

    print("[3] Corner condition: background corners free of leaf and coin")
    corner_sets = []
    for s, n in zip(srcs, names):
        ps, why = corners.patches(s)
        corner_sets.append(ps)
        print(f"    {n}  accepted {len(ps)}/4  rejected: {', '.join(why) or '-'}")
    print()

    sel = [i for i in ok if corner_sets[i]]
    paths = [crops[i] for i in sel]
    frames = [srcs[i] for i in sel]
    donors = paths[1:] + paths[:1]
    print(f"[4] Input conditions, ResNet-18 (ImageNet) features, {len(sel)} images")
    print("    cosine similarity of each condition's features to those of `crop`")
    feats = {}
    for mode in CONDITIONS:
        use = paths if mode in interventions.CROP_MODES else frames
        feats[mode] = interventions.embed(
            use, mode, donors=donors if mode == "crop_swap" else None)
    feats["corner"] = corners.embed_arrays([corner_sets[i] for i in sel])
    ref = feats["crop"]
    for mode in CONDITIONS + ["corner"]:
        c = cosine(feats[mode], ref)
        print(f"    {mode.ljust(13)} mean {c.mean():.3f}  min {c.min():.3f}  max {c.max():.3f}")
    print()

    print("[5] Between-image similarity within each condition")
    print("    mean cosine similarity between different images")
    for mode in ["crop", "crop_mask", "crop_leafout", "corner"]:
        X = feats[mode] / np.linalg.norm(feats[mode], axis=1, keepdims=True)
        S = X @ X.T
        off = S[~np.eye(len(S), dtype=bool)]
        print(f"    {mode.ljust(13)} {off.mean():.3f}")
    print()

    print("[6] Grouped five-fold split, one group per leaf")
    fold_of = splits.assign_folds({"examples": names})
    for f in range(5):
        members = [n for n in names if fold_of[n] == f]
        print(f"    fold {f}: {', '.join(members)}")
    print("\n    The classifier, the empirical null and the group bootstrap"
          " (leafaudit/evaluate.py) need labelled species and are not run here.")


if __name__ == "__main__":
    main()
