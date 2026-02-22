#!/usr/bin/env python3
"""Explore BDD100K dataset structure and tag distributions.

Usage:
    python explore_bdd100k.py --bdd_root /path/to/bdd100k

This script prints counts per split, tag distributions (weather/scene/time),
and samples a few images showing their metadata.
"""
import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Optional

from bdd100k_dataset import BDD100KDataset


def find_tagging_json(bdd_root: Path, labels_root: Optional[Path] = None, labels_json: Optional[Path] = None):
    # If a specific labels JSON file is provided and exists, use it
    if labels_json:
        lj = Path(labels_json)
        if lj.exists():
            return lj

    # Search under the dataset root for tagging JSONs
    candidates = list(bdd_root.rglob("*tagging*.json"))
    if candidates:
        return candidates[0]

    # If a separate labels root is provided (e.g. bdd100k_labels_release), check common files
    if labels_root:
        lr = Path(labels_root)
        if lr.exists():
            specific = lr / "bdd100k_labels_images_train.json"
            if specific.exists():
                return specific
            lbl_candidates = list(lr.rglob("*tagging*.json"))
            if lbl_candidates:
                return lbl_candidates[0]
            common_lbl = lr / "tagging" / "tagging_images.json"
            if common_lbl.exists():
                return common_lbl

    # Fallback
    common = bdd_root / "labels" / "tagging" / "tagging_images.json"
    if common.exists():
        return common
    return None


def print_distributions(dataset: BDD100KDataset, n_sample=5):
    weather_c = Counter()
    scene_c = Counter()
    time_c = Counter()

    for _, meta in dataset:
        tags = meta.get("tags", {})
        weather_c.update([tags.get("weather", "unknown")])
        scene_c.update([tags.get("scene", "unknown")])
        time_c.update([tags.get("timeofday", tags.get("time", "unknown"))])

    print("Weather distribution:")
    for k, v in weather_c.most_common():
        print(f"  {k}: {v}")

    print("\nScene distribution:")
    for k, v in scene_c.most_common():
        print(f"  {k}: {v}")

    print("\nTime distribution:")
    for k, v in time_c.most_common():
        print(f"  {k}: {v}")

    print("\nSample images and metadata:")
    sampled = random.sample(list(dataset), min(n_sample, len(dataset)))
    for img_path, meta in sampled:
        print(f"- {img_path}")
        print(f"  tags: {meta.get('tags', {})}")
        ann = meta.get("annotations")
        if ann:
            print(f"  annotations: {len(ann)} entries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd_root", default=str(Path(__file__).resolve().parent / "dataset" / "bdd100k" / "bdd100k"))
    ap.add_argument("--labels_root", default=str(Path(__file__).resolve().parent / "dataset" / "bdd100k" / "bdd100k_labels_release" / "bdd100k" / "labels"))
    ap.add_argument("--labels_json", default=str(Path(__file__).resolve().parent / "dataset" / "bdd100k" / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_train.json"))
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n_sample", type=int, default=5)
    args = ap.parse_args()

    bdd_root = Path(args.bdd_root)
    labels_root = Path(args.labels_root) if args.labels_root else None
    labels_json = Path(args.labels_json) if args.labels_json else None
    print(f"Looking for tagging JSON under {bdd_root}, {labels_root}, or explicit file {labels_json}")
    tagging = find_tagging_json(bdd_root, labels_root=labels_root, labels_json=labels_json)
    if tagging:
        print(f"Found tagging json: {tagging}")
    else:
        print("No tagging JSON found — tags will be 'unknown'.")

    # Count images per split
    splits = {}
    for s in ["train", "val", "test"]:
        ds = BDD100KDataset(root_dir=str(bdd_root), split=s,
                             labels_root=str(labels_root) if labels_root else None,
                             labels_json=str(labels_json) if labels_json else None)
        splits[s] = len(ds)

    print("\nImage counts per split:")
    for s, c in splits.items():
        print(f"  {s}: {c}")

    ds = BDD100KDataset(root_dir=str(bdd_root), split=args.split,
                        labels_root=str(labels_root) if labels_root else None,
                        labels_json=str(labels_json) if labels_json else None)
    if len(ds) == 0:
        print(f"No images found for split {args.split}. Check dataset layout.")
        return

    print_distributions(ds, n_sample=args.n_sample)


if __name__ == "__main__":
    main()
