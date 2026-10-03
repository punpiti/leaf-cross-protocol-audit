# Leaf-image species classification audit: code

Code for *Within-protocol accuracy does not transfer: an intervention-based
audit of leaf-image species classification in tropical tree seedlings*.

## Quick start

    pip install -r requirements.txt
    python demo.py

The demo runs the pipeline on the twelve images in `examples/` and prints every
result to the terminal in about twenty seconds on a CPU. The first run downloads
the ImageNet ResNet-18 weights through torchvision.

The example images are leaves of an unidentified species, photographed under
the same protocol as the study material but not part of it. They are reduced to
1600 pixels on the long side and carry no metadata. They have no species labels,
so the demo shows each step of the method but not a classification score.

## Contents

| Path | What it is |
|---|---|
| `leafaudit/segment.py` | leaf segmentation on the a* axis of CIELAB and the square crop around the leaf |
| `leafaudit/interventions.py` | the input conditions (background swap, occlusion, leaf-out, silhouette, and the grey and colour-normalised frames) and frozen ResNet-18 features |
| `leafaudit/corners.py` | the corner condition: features from image corners that contain neither leaf nor coin |
| `leafaudit/splits.py` | the grouped five-fold split, with seed 20260918 |
| `leafaudit/evaluate.py` | grouped cross-validated logistic regression, the empirical null and the paired group bootstrap |
| `demo.py` | runs the pipeline on `examples/` |
| `results/` | the aggregate result files behind the tables and figures of the article |
| `results/eval_counts/` | image and evaluation-unit counts of every evaluation set, by species and fold, and of the fine-tuning splits; no image names. `eval_sets_summary.csv` gives the totals reported in Section 2.4. P1 units are single images, because P1 has no leaf identity |

The functions in `leafaudit/` are those used for the article, with the same
parameters and order of operations. This record was built from commit `fc5e79d` of the working repository, where the same package sits at
`scripts/leafaudit/` and the result files at `labels-dataset2/`.

Species in the result files are given by three names: the accepted scientific
name, the English common name where one is in common use, and the Thai name
recorded when the plants were collected.

## What the demo prints

1. the leaf fraction of each crop after segmentation (Section 2.5);
2. which image corners are used by the corner condition (Section 2.6);
3. for each input condition of Sections 2.6 and 3.6, the cosine similarity of
   its ResNet-18 features to those of the unmodified crop, a check for this
   demo only;
4. the grouped fold assignment (Section 2.4).

## License

The code is released under the MIT License (`LICENSE`). The result files are
released under the Creative Commons Attribution 4.0 International License
(CC BY 4.0).
