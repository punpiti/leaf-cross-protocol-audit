"""Grouped five-fold split: the unit is a leaf group, so no group spans train and test."""
import random

SEED = 20260918


def assign_folds(groups_by_class, folds=5, seed=SEED):
    """{class: [group_key, ...]} -> {group_key: fold}, stratified by class."""
    rng = random.Random(seed)
    fold_of = {}
    for folder, groups in groups_by_class.items():
        g = sorted(groups)
        rng.shuffle(g)
        for i, gk in enumerate(g):
            fold_of[gk] = i % folds
    return fold_of
