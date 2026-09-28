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
| `leafaudit/shape.py` | size, shape and colour measurements for the feature baseline |
| `leafaudit/interventions.py` | the input conditions (mask, leaf-out, noise, background swap, occlusion, silhouette, grey) and frozen ResNet-18 features |
| `leafaudit/corners.py` | the corner condition: features from image corners that contain neither leaf nor coin |
| `leafaudit/splits.py` | the grouped five-fold split, with seed 20260918 |
| `leafaudit/evaluate.py` | grouped cross-validated logistic regression, the empirical null and the paired group bootstrap |
| `demo.py` | runs the pipeline on `examples/` |
| `results/` | the aggregate result files behind the tables and figures of the article |

The functions in `leafaudit/` are those used for the article, with the same
parameters and order of operations. This record was built from commit
`e81e969` of the working repository, where the same package sits at
`scripts/leafaudit/` and the result files at `labels-dataset2/`.

Species in the result files are given by three names: the accepted scientific
name, the English common name where one is in common use, and the Thai name
recorded when the plants were collected.

## What the demo prints

1. the leaf fraction of each crop after segmentation;
2. the leaf measurements used by the feature baseline;
3. which image corners pass the leaf-and-object test;
4. for each input condition, the cosine similarity of its ResNet-18 features to
   those of the unmodified crop;
5. the mean similarity between different images within a condition;
6. the grouped fold assignment.

## License

The code is released under the MIT License (`LICENSE`). The result files are
released under the Creative Commons Attribution 4.0 International License
(CC BY 4.0).
