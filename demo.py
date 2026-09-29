"""Run the method of the article on the example images and print the results.

    python demo.py [--images examples] [--work DIR]

Section numbers refer to the article. The example images carry no species
labels, so the demo shows each step of the method but no classification score.
"""
import argparse
import glob
import os
import tempfile

import numpy as np

from leafaudit import corners, interventions, segment, splits

FRAME_CONDITIONS = ["rgb", "gray", "colornorm"]
CROP_CONDITIONS = ["crop", "crop_swap", "crop_occ", "crop_leafout", "silhouette"]


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
    print(f"{len(srcs)} images from {args.images} · crops and masks written to {work}\n")

    print("[1] Leaf segmentation and crops (Section 2.5)")
    print("    fraction of the crop that is leaf; Section 3.1 reports its median by protocol")
    crops = []
    for s, n in zip(srcs, names):
        dst = os.path.join(work, "crops", f"{n}.jpg")
        frac = segment.crop_leaf(s, dst, dst[:-4] + "_m.png")
        crops.append(dst if frac > 0 else None)
        print(f"    {n}  {frac:.3f}")
    ok = [i for i, c in enumerate(crops) if c]
    print(f"    segmented {len(ok)}/{len(srcs)}\n")

    print("[2] Corner condition: ground-sheet corners with no leaf and no other object (Section 2.6)")
    labels = {"leaf": "leaf", "object": "coin or other object"}
    corner_sets = []
    for s, n in zip(srcs, names):
        ps, why = corners.patches(s)
        corner_sets.append(ps)
        out = ", ".join(f"{w.split(':')[0]} ({labels[w.split(':')[1]]})" for w in why)
        print(f"    {n}  corners used {len(ps)}/4" + (f" · excluded {out}" if out else ""))
    used = [len(c) for c in corner_sets]
    print(f"    mean corners used per image {np.mean(used):.2f}\n")

    sel = [i for i in ok if corner_sets[i]]
    paths = [crops[i] for i in sel]
    frames = [srcs[i] for i in sel]
    donors = paths[1:] + paths[:1]
    print("[3] Input conditions and ResNet-18 (ImageNet) features (Sections 2.6, 2.8, 3.7)")
    print("    crop conditions of Section 2.6, corner, and the uncropped-frame colour conditions of Section 3.7")
    print("    cosine similarity of each condition's features to those of `crop`, a check for this demo only;")
    print("    the article reports macro-F1, which needs species labels")
    feats = {}
    for mode in CROP_CONDITIONS:
        feats[mode] = interventions.embed(
            paths, mode, donors=donors if mode == "crop_swap" else None)
    feats["corner"] = corners.embed_arrays([corner_sets[i] for i in sel])
    for mode in FRAME_CONDITIONS:
        feats[mode] = interventions.embed(frames, mode)
    ref = feats["crop"]
    for mode in CROP_CONDITIONS + ["corner"] + FRAME_CONDITIONS:
        c = cosine(feats[mode], ref)
        print(f"    {mode.ljust(13)} mean {c.mean():.3f}  min {c.min():.3f}  max {c.max():.3f}")
    print()

    print("[4] Grouped five-fold split, one group per leaf (Section 2.4)")
    fold_of = splits.assign_folds({"examples": names})
    for f in range(5):
        members = [n for n in names if fold_of[n] == f]
        print(f"    fold {f}: {', '.join(members)}")
    print("\n    Classification, the empirical null and the group bootstrap (Sections 2.7, 2.8, 2.10;"
          "\n    leafaudit/evaluate.py) need labelled species and are not run here.")


if __name__ == "__main__":
    main()
