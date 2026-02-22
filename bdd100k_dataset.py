"""BDD100K lightweight dataset loader.

Provides BDD100KDataset which yields (image_path, metadata_dict).
Metadata contains 'tags' (weather/scene/timeofday) when available and
optional 'annotations' if detection JSONs are present.

The loader attempts several common dataset layouts and handles missing
files gracefully.
"""
from pathlib import Path
import json
import os
from typing import List, Optional, Dict, Iterator, Tuple


class BDD100KDataset:
    def __init__(self, root_dir: str = "/path/to/bdd100k", split: str = "val",
                 weather: Optional[str] = None, scene: Optional[str] = None,
                 timeofday: Optional[str] = None, labels_root: Optional[str] = None,
                 labels_json: Optional[str] = None):
        self.root = Path(root_dir)
        self.split = split
        self.weather = weather
        self.scene = scene
        self.timeofday = timeofday
        # Optional explicit labels/tagging locations (may be outside self.root)
        self.labels_root = Path(labels_root) if labels_root else None
        self.labels_json = Path(labels_json) if labels_json else None

        self.images_dir = self._find_images_dir()
        self.image_paths = self._collect_images()

        # Try to load tagging json if present
        self.tagging = self._load_tagging_json()

        # Try to load detection annotations (optional)
        self.detections = self._load_detection_json()

        # Filter by tags if requested
        if any([weather, scene, timeofday]):
            self.image_paths = [p for p in self.image_paths if self._matches_filters(p)]

    def _find_images_dir(self) -> Path:
        # Common layouts
        candidates = [
            self.root / "images" / "100k" / self.split,
            self.root / "images" / self.split,
            self.root / "bdd100k" / "images" / self.split,
            self.root / "images" / f"{self.split}"  # fallback
        ]
        for c in candidates:
            if c.exists():
                return c
        # If none found, fallback to root/images
        fallback = self.root / "images"
        return fallback

    def _collect_images(self) -> List[str]:
        if not self.images_dir.exists():
            return []
        imgs = [str(p) for p in sorted(self.images_dir.rglob("*.jpg"))]
        return imgs

    def _load_tagging_json(self) -> Dict[str, Dict]:
        # Merge all tagging-specific and label JSON files found under root or
        # optional external locations provided via constructor.
        mapping: Dict[str, Dict] = {}

        # Load any explicit tagging files under self.root
        tag_candidates = list(self.root.rglob("*tagging*.json"))
        for path in tag_candidates:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue
            if isinstance(data, dict):
                mapping.update(data)
            elif isinstance(data, list):
                for entry in data:
                    name = entry.get("name") or entry.get("image")
                    if name:
                        mapping[name] = entry.get("attributes", {})

        # If an explicit labels_json path was given, load it first
        det_candidates = []
        if self.labels_json and self.labels_json.exists():
            det_candidates.append(self.labels_json)

        # Also search under labels_root if provided
        if self.labels_root:
            det_candidates.extend(list(self.labels_root.rglob("*labels*.json")))

        # Finally search under self.root
        det_candidates.extend(list(self.root.rglob("*labels*.json")))
        for path in det_candidates:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue
            if isinstance(data, list):
                for entry in data:
                    name = entry.get("name") or entry.get("image")
                    if not name:
                        continue
                    # don't overwrite attributes loaded from explicit tagging files
                    if name not in mapping:
                        mapping[name] = entry.get("attributes", {})
            elif isinstance(data, dict):
                for name, info in data.items():
                    if name not in mapping:
                        mapping[name] = info.get("attributes", {})

        return mapping
    def _load_detection_json(self) -> Dict[str, list]:
        # Search for detection jsons
        candidates = list(self.root.rglob("*det*.json"))
        if not candidates:
            return {}
        path = candidates[0]
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            return {}

        mapping = {}
        if isinstance(data, list):
            for entry in data:
                name = entry.get("name") or entry.get("image")
                if not name:
                    continue
                mapping[name] = entry.get("labels") or entry.get("objects") or []
        elif isinstance(data, dict):
            mapping = data
        return mapping

    def _image_name(self, path: str) -> str:
        return Path(path).name

    def _matches_filters(self, path: str) -> bool:
        name = self._image_name(path)
        tags = self.tagging.get(name, {}) if self.tagging else {}

        def _matches(filter_val, actual_val) -> bool:
            if not filter_val:
                return True
            if actual_val is None:
                return False
            try:
                return filter_val.lower() in actual_val.lower()
            except Exception:
                return filter_val == actual_val

        if self.weather and not _matches(self.weather, tags.get("weather")):
            return False
        if self.scene and not _matches(self.scene, tags.get("scene")):
            return False
        td = tags.get("timeofday") or tags.get("time")
        if self.timeofday and not _matches(self.timeofday, td):
            return False
        return True

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self) -> Iterator[Tuple[str, Dict]]:
        for p in self.image_paths:
            yield p, self._build_metadata(p)

    def __getitem__(self, idx) -> Tuple[str, Dict]:
        p = self.image_paths[idx]
        return p, self._build_metadata(p)

    def _build_metadata(self, image_path: str) -> Dict:
        name = self._image_name(image_path)
        tags = self.tagging.get(name, {}) if self.tagging else {}
        anns = self.detections.get(name) if self.detections else None
        return {"tags": tags, "annotations": anns}


if __name__ == "__main__":
    # quick smoke test
    ds = BDD100KDataset(root_dir="/path/to/bdd100k", split="val")
    print(f"Found {len(ds)} images")
