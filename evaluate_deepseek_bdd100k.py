import argparse
import ast
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from bdd100k_dataset import BDD100KDataset


CORE_OBJECT_CATEGORIES = [
    "car",
    "person",
    "truck",
    "bus",
    "bike",
    "motor",
    "traffic light",
    "traffic sign",
]

SCENE_CHOICES = [
    "city street",
    "residential",
    "highway",
    "tunnel",
    "parking lot",
    "gas stations",
]

WEATHER_CHOICES = [
    "clear",
    "overcast",
    "partly cloudy",
    "rainy",
    "snowy",
    "foggy",
]

TIMEOFDAY_CHOICES = [
    "daytime",
    "night",
    "dawn/dusk",
]

TRAFFIC_LIGHT_CHOICES = ["red", "yellow", "green", "none", "unknown"]
YES_NO_CHOICES = ["yes", "no"]

TASK_DEFINITIONS = {
    "scene_classification": {
        "prompt": (
            "Classify the scene type using only visible evidence in the image. "
            "Do not guess from common driving defaults. "
            "Choose the single best option and answer with only the label. "
            "Options: "
            "1. city street "
            "2. residential "
            "3. highway "
            "4. tunnel "
            "5. parking lot "
            "6. gas stations"
        ),
        "max_new_tokens": 12,
    },
    "weather_classification": {
        "prompt": (
            "Classify the weather using only visible evidence in the image. "
            "Do not guess from common driving defaults. "
            "Choose the single best option and answer with only the label. "
            "Options: "
            "1. clear "
            "2. overcast "
            "3. partly cloudy "
            "4. rainy "
            "5. snowy "
            "6. foggy"
        ),
        "max_new_tokens": 8,
    },
    "timeofday_classification": {
        "prompt": (
            "Classify the time of day using only visible evidence in the image. "
            "Do not explain. Reply with exactly one label or one number only. "
            "Options: "
            "1. daytime "
            "2. night "
            "3. dawn/dusk"
        ),
        "max_new_tokens": 8,
    },
    "object_presence": {
        "prompt": (
            "Inspect the image carefully from left to right, center, near distance, and far distance. "
            "For each category, answer true if at least one instance is visible anywhere in the image, "
            "even if it is small, distant, partially occluded, blurry, or near the image edge. "
            "Do not default to all false. Use false only after checking that category and not seeing it. "
            "Cars, traffic lights, and traffic signs are common in driving scenes, so check those "
            "categories carefully before answering. "
            "Do not omit any key and do not add any extra text. "
            "Answer with one compact JSON object using only true or false values, exactly in this schema: "
            "{\"car\": false, \"person\": false, \"truck\": false, "
            "\"bus\": false, \"bike\": false, \"motor\": false, "
            "\"traffic light\": false, \"traffic sign\": false}"
        ),
        "max_new_tokens": 140,
    },
    "traffic_light_state": {
        "prompt": (
            "Determine the traffic light state relevant to the ego vehicle. "
            "First decide whether any traffic light is visible at all. "
            "If none is visible, answer none. "
            "If one or more traffic lights are visible but the illuminated color cannot be read clearly, "
            "answer unknown. "
            "If multiple traffic lights are visible, choose the one most likely controlling the ego lane; "
            "if that is unclear, choose the largest or most central visible traffic light. "
            "Do not default to red. Use only visible illuminated color. "
            "Do not explain. Reply with exactly one label or one number only. "
            "Options: 1. red 2. yellow 3. green 4. none 5. unknown"
        ),
        "max_new_tokens": 6,
    },
    "drivable_area_presence": {
        "prompt": (
            "Is the lane or path directly in front of the ego vehicle drivable right now? "
            "Use only what is visible in the image. "
            "Answer with exactly one word: yes or no."
        ),
        "max_new_tokens": 4,
    },
}

CANONICAL_VALUE_MAP = {
    "scene": {
        "citystreet": "city street",
        "city street": "city street",
        "street": "city street",
        "residential": "residential",
        "highway": "highway",
        "tunnel": "tunnel",
        "parkinglot": "parking lot",
        "parking lot": "parking lot",
        "parking": "parking lot",
        "gasstation": "gas stations",
        "gas station": "gas stations",
        "gas stations": "gas stations",
    },
    "weather": {
        "clear": "clear",
        "sunny": "clear",
        "overcast": "overcast",
        "cloudy": "overcast",
        "partlycloudy": "partly cloudy",
        "partly cloudy": "partly cloudy",
        "rainy": "rainy",
        "rain": "rainy",
        "snowy": "snowy",
        "snow": "snowy",
        "foggy": "foggy",
        "fog": "foggy",
    },
    "timeofday": {
        "day": "daytime",
        "daytime": "daytime",
        "night": "night",
        "dawn": "dawn/dusk",
        "dusk": "dawn/dusk",
        "sunrise": "dawn/dusk",
        "sunset": "dawn/dusk",
        "dawndusk": "dawn/dusk",
        "dawn/dusk": "dawn/dusk",
        "twilight": "dawn/dusk",
        "1": "daytime",
        "2": "night",
        "3": "dawn/dusk",
    },
    "traffic_light": {
        "red": "red",
        "yellow": "yellow",
        "amber": "yellow",
        "green": "green",
        "none": "none",
        "no light": "none",
        "not visible": "none",
        "unknown": "unknown",
        "1": "red",
        "2": "yellow",
        "3": "green",
        "4": "none",
        "5": "unknown",
    },
    "yes_no": {
        "yes": "yes",
        "true": "yes",
        "present": "yes",
        "available": "yes",
        "drivable": "yes",
        "no": "no",
        "false": "no",
        "absent": "no",
        "blocked": "no",
        "notdrivable": "no",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek-VL on BDD100K perception tasks."
    )
    parser.add_argument(
        "--bdd_root",
        default="dataset/bdd100k/bdd100k/bdd100k",
        help="Root directory for BDD100K images.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--labels_root",
        default="dataset/bdd100k/bdd100k_labels_release/bdd100k/labels",
        help="Optional labels root directory.",
    )
    parser.add_argument(
        "--labels_json",
        default=None,
        help="Optional explicit BDD100K labels JSON file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of images to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path for full evaluation JSON output.",
    )
    parser.add_argument(
        "--summary_out",
        default=None,
        help="Optional path for compact summary JSON output.",
    )
    parser.add_argument(
        "--model_path",
        default="deepseek-ai/deepseek-vl-1.3b-chat",
        help="Model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Optional global override for task max_new_tokens.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose model loading and progress messages.",
    )
    return parser.parse_args()


def default_labels_json(labels_root: str, split: str) -> str:
    return str(Path(labels_root) / f"bdd100k_labels_images_{split}.json")


def sanitize_text(value: str) -> str:
    lowered = value.strip().lower()
    lowered = lowered.replace("\n", " ").replace("_", " ")
    lowered = lowered.replace('"', " ").replace("'", " ")
    lowered = re.sub(r"[^a-z0-9/ ]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def compact_text(value: str) -> str:
    return sanitize_text(value).replace(" ", "")


def normalize_choice(raw_text: str, mapping: Dict[str, str], choices: Sequence[str]) -> str:
    text = sanitize_text(raw_text)
    if not text:
        return "invalid"

    if text in mapping:
        return mapping[text]

    compact = compact_text(raw_text)
    if compact in mapping:
        return mapping[compact]

    for choice in choices:
        if text == choice:
            return choice
        if choice in text:
            return choice

    for alias, canonical in mapping.items():
        if alias in text or alias == compact:
            return canonical

    return "invalid"


def normalize_scene(raw_text: str) -> str:
    return normalize_choice(raw_text, CANONICAL_VALUE_MAP["scene"], SCENE_CHOICES)


def normalize_weather(raw_text: str) -> str:
    return normalize_choice(raw_text, CANONICAL_VALUE_MAP["weather"], WEATHER_CHOICES)


def normalize_timeofday(raw_text: str) -> str:
    return normalize_choice(raw_text, CANONICAL_VALUE_MAP["timeofday"], TIMEOFDAY_CHOICES)


def normalize_traffic_light(raw_text: str) -> str:
    return normalize_choice(
        raw_text, CANONICAL_VALUE_MAP["traffic_light"], TRAFFIC_LIGHT_CHOICES
    )


def normalize_yes_no(raw_text: str) -> str:
    return normalize_choice(raw_text, CANONICAL_VALUE_MAP["yes_no"], YES_NO_CHOICES)


def try_extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = raw_text[start : end + 1]
    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            loaded = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            return None
    return loaded if isinstance(loaded, dict) else None


def parse_boolean_token(token: Any) -> Optional[bool]:
    if isinstance(token, bool):
        return token
    if token is None:
        return None
    normalized = sanitize_text(str(token))
    if normalized in {"true", "yes", "present", "1"}:
        return True
    if normalized in {"false", "no", "absent", "0"}:
        return False
    return None


def parse_object_presence(raw_text: str) -> Dict[str, Any]:
    parsed: Dict[str, Optional[bool]] = {}
    invalid_categories: List[str] = []

    obj = try_extract_json_object(raw_text)
    if obj is not None:
        for category in CORE_OBJECT_CATEGORIES:
            direct = obj.get(category)
            if direct is None:
                compact_matches = [
                    key for key in obj.keys()
                    if compact_text(str(key)) == compact_text(category)
                ]
                direct = obj.get(compact_matches[0]) if compact_matches else None
            parsed_value = parse_boolean_token(direct)
            parsed[category] = parsed_value
            if parsed_value is None:
                invalid_categories.append(category)
    else:
        text = sanitize_text(raw_text)
        for category in CORE_OBJECT_CATEGORIES:
            category_pattern = re.escape(category)
            match = re.search(
                rf"{category_pattern}\s*[:=-]?\s*(true|false|yes|no|present|absent)",
                text,
            )
            if match:
                parsed_value = parse_boolean_token(match.group(1))
            else:
                positive = re.search(rf"\b{category_pattern}\b", text)
                negative = re.search(rf"\b(no|not)\s+{category_pattern}\b", text)
                if negative:
                    parsed_value = False
                elif positive:
                    parsed_value = True
                else:
                    parsed_value = None
            parsed[category] = parsed_value
            if parsed_value is None:
                invalid_categories.append(category)

    # If the model returns a compact JSON with omitted keys, default only the
    # missing keys to False when at least half of the schema was provided.
    provided_keys = len(parsed) - len(invalid_categories)
    if obj is not None and provided_keys >= len(CORE_OBJECT_CATEGORIES) // 2:
        for category in CORE_OBJECT_CATEGORIES:
            if parsed.get(category) is None:
                parsed[category] = False
        invalid_categories = [c for c, v in parsed.items() if v is None]

    return {
        "parsed": {key: value for key, value in parsed.items() if value is not None},
        "invalid": len(invalid_categories) > 0,
        "invalid_categories": invalid_categories,
    }


def extract_category_presence(labels: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, bool]:
    present = {category: False for category in CORE_OBJECT_CATEGORIES}
    for label in labels or []:
        category = label.get("category")
        if category in present:
            present[category] = True
    return present


def dominant_traffic_light_color(labels: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, Any]:
    colors = []
    for label in labels or []:
        if label.get("category") != "traffic light":
            continue
        attributes = label.get("attributes") or {}
        color = sanitize_text(str(attributes.get("trafficLightColor", "")))
        if color in {"red", "yellow", "green"}:
            colors.append(color)
        elif color == "none":
            colors.append("none")

    if not colors:
        return {"eligible": False, "value": None}

    counts = Counter(colors)
    dominant = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return {"eligible": True, "value": dominant}


def has_direct_drivable_area(labels: Optional[Iterable[Dict[str, Any]]]) -> str:
    for label in labels or []:
        if label.get("category") != "drivable area":
            continue
        attributes = label.get("attributes") or {}
        if sanitize_text(str(attributes.get("areaType", ""))) == "direct":
            return "yes"
    return "no"


def build_ground_truth(metadata: Dict[str, Any]) -> Dict[str, Any]:
    tags = metadata.get("tags") or {}
    annotations = metadata.get("annotations") or []

    scene = normalize_scene(str(tags.get("scene", "")))
    weather = normalize_weather(str(tags.get("weather", "")))
    timeofday = normalize_timeofday(str(tags.get("timeofday") or tags.get("time") or ""))
    traffic_light = dominant_traffic_light_color(annotations)

    return {
        "scene_classification": scene,
        "weather_classification": weather,
        "timeofday_classification": timeofday,
        "object_presence": extract_category_presence(annotations),
        "traffic_light_state": traffic_light["value"],
        "traffic_light_state_eligible": traffic_light["eligible"],
        "drivable_area_presence": has_direct_drivable_area(annotations),
    }


def balanced_sample(
    dataset: BDD100KDataset, num_samples: int, seed: int
) -> List[Tuple[str, Dict[str, Any]]]:
    items = list(dataset)
    if num_samples >= len(items):
        return items

    rng = random.Random(seed)
    grouped: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for image_path, metadata in items:
        tags = metadata.get("tags") or {}
        scene = normalize_scene(str(tags.get("scene", "")))
        weather = normalize_weather(str(tags.get("weather", "")))
        timeofday = normalize_timeofday(str(tags.get("timeofday") or tags.get("time") or ""))
        key = f"{scene}|{weather}|{timeofday}"
        grouped[key].append((image_path, metadata))

    for samples in grouped.values():
        rng.shuffle(samples)

    ordered_groups = sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[0])
    )
    selected: List[Tuple[str, Dict[str, Any]]] = []
    selected_paths = set()

    while len(selected) < num_samples:
        added_this_round = False
        for _, group_items in ordered_groups:
            while group_items and group_items[0][0] in selected_paths:
                group_items.pop(0)
            if not group_items:
                continue
            image_path, metadata = group_items.pop(0)
            selected.append((image_path, metadata))
            selected_paths.add(image_path)
            added_this_round = True
            if len(selected) >= num_samples:
                break
        if not added_this_round:
            break

    if len(selected) < num_samples:
        leftovers = [item for item in items if item[0] not in selected_paths]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: num_samples - len(selected)])

    return selected[:num_samples]


def parse_prediction(task_name: str, raw_text: str) -> Dict[str, Any]:
    if task_name == "scene_classification":
        value = normalize_scene(raw_text)
        return {"value": value, "invalid": value == "invalid"}
    if task_name == "weather_classification":
        value = normalize_weather(raw_text)
        return {"value": value, "invalid": value == "invalid"}
    if task_name == "timeofday_classification":
        value = normalize_timeofday(raw_text)
        return {"value": value, "invalid": value == "invalid"}
    if task_name == "traffic_light_state":
        value = normalize_traffic_light(raw_text)
        return {"value": value, "invalid": value == "invalid"}
    if task_name == "drivable_area_presence":
        value = normalize_yes_no(raw_text)
        return {"value": value, "invalid": value == "invalid"}
    if task_name == "object_presence":
        return parse_object_presence(raw_text)
    raise ValueError(f"Unsupported task: {task_name}")


def score_prediction(
    task_name: str, prediction: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    if task_name == "object_presence":
        gt = ground_truth[task_name]
        pred = prediction.get("parsed", {})
        category_scores = {}
        for category in CORE_OBJECT_CATEGORIES:
            pred_value = pred.get(category)
            gt_value = gt[category]
            category_scores[category] = {
                "eligible": pred_value is not None,
                "correct": None if pred_value is None else pred_value == gt_value,
                "prediction": pred_value,
                "ground_truth": gt_value,
            }
        return {
            "eligible": True,
            "invalid": prediction.get("invalid", False),
            "category_scores": category_scores,
        }

    if task_name == "traffic_light_state":
        eligible = bool(ground_truth["traffic_light_state_eligible"])
        if not eligible:
            return {"eligible": False, "invalid": False, "correct": None}
        value = prediction.get("value")
        return {
            "eligible": True,
            "invalid": prediction.get("invalid", False),
            "correct": None if prediction.get("invalid") else value == ground_truth[task_name],
        }

    value = prediction.get("value")
    gt_value = ground_truth[task_name]
    eligible = gt_value != "invalid"
    if not eligible:
        return {"eligible": False, "invalid": False, "correct": None}

    return {
        "eligible": True,
        "invalid": prediction.get("invalid", False),
        "correct": None if prediction.get("invalid") else value == gt_value,
    }


def update_object_metrics(
    aggregate: Dict[str, Dict[str, float]], ground_truth: Dict[str, bool], prediction: Dict[str, Any]
) -> None:
    parsed = prediction.get("parsed", {})
    invalid_categories = set(prediction.get("invalid_categories", []))
    for category in CORE_OBJECT_CATEGORIES:
        stats = aggregate[category]
        gt_value = ground_truth[category]
        pred_value = parsed.get(category)
        if pred_value is None:
            stats["invalid"] += 1
            continue
        stats["eligible"] += 1
        if pred_value == gt_value:
            stats["correct"] += 1
        if pred_value and gt_value:
            stats["tp"] += 1
        elif pred_value and not gt_value:
            stats["fp"] += 1
        elif not pred_value and gt_value:
            stats["fn"] += 1
        else:
            stats["tn"] += 1

    for category in invalid_categories:
        aggregate[category]["invalid"] += 0


def compute_summary(
    sampled_items: List[Tuple[str, Dict[str, Any]]],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_images": len(results),
        "sampled_distributions": {
            "scene": Counter(),
            "weather": Counter(),
            "timeofday": Counter(),
        },
        "tasks": {},
    }

    for _, metadata in sampled_items:
        gt = build_ground_truth(metadata)
        summary["sampled_distributions"]["scene"][gt["scene_classification"]] += 1
        summary["sampled_distributions"]["weather"][gt["weather_classification"]] += 1
        summary["sampled_distributions"]["timeofday"][gt["timeofday_classification"]] += 1

    for key, counter in summary["sampled_distributions"].items():
        summary["sampled_distributions"][key] = dict(sorted(counter.items()))

    task_stats: Dict[str, Dict[str, Any]] = {}
    object_stats = {
        category: {
            "eligible": 0,
            "correct": 0,
            "invalid": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
        for category in CORE_OBJECT_CATEGORIES
    }

    for task_name in TASK_DEFINITIONS:
        if task_name == "object_presence":
            continue
        task_stats[task_name] = {
            "eligible": 0,
            "correct": 0,
            "invalid": 0,
        }

    for result in results:
        ground_truth = result["ground_truth"]
        predictions = result["predictions"]
        scores = result["scores"]

        for task_name in TASK_DEFINITIONS:
            if task_name == "object_presence":
                update_object_metrics(
                    object_stats, ground_truth["object_presence"], predictions["object_presence"]
                )
                continue

            task_score = scores[task_name]
            if not task_score["eligible"]:
                continue
            task_stats[task_name]["eligible"] += 1
            if task_score["invalid"]:
                task_stats[task_name]["invalid"] += 1
            if task_score["correct"] is True:
                task_stats[task_name]["correct"] += 1

    for task_name, stats in task_stats.items():
        eligible = stats["eligible"]
        invalid = stats["invalid"]
        correct = stats["correct"]
        task_stats[task_name] = {
            **stats,
            "accuracy": round(correct / eligible, 4) if eligible else None,
            "invalid_rate": round(invalid / eligible, 4) if eligible else None,
        }

    object_summary = {}
    for category, stats in object_stats.items():
        eligible = stats["eligible"]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall)
            else None
        )
        object_summary[category] = {
            **stats,
            "accuracy": round(stats["correct"] / eligible, 4) if eligible else None,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "invalid_rate": round(stats["invalid"] / len(results), 4) if results else None,
        }

    task_stats["object_presence"] = {
        "per_category": object_summary,
        "macro_accuracy": round(
            sum(v["accuracy"] for v in object_summary.values() if v["accuracy"] is not None)
            / max(1, sum(1 for v in object_summary.values() if v["accuracy"] is not None)),
            4,
        ),
        "macro_f1": round(
            sum(v["f1"] for v in object_summary.values() if v["f1"] is not None)
            / max(1, sum(1 for v in object_summary.values() if v["f1"] is not None)),
            4,
        ),
    }

    summary["tasks"] = task_stats
    return summary


def resolved_config(args: argparse.Namespace, dataset: BDD100KDataset) -> Dict[str, Any]:
    return {
        "bdd_root": str(Path(args.bdd_root).resolve()),
        "split": args.split,
        "labels_root": str(Path(args.labels_root).resolve()) if args.labels_root else None,
        "labels_json": str(Path(args.labels_json).resolve()) if args.labels_json else None,
        "resolved_images_dir": str(dataset.images_dir.resolve()),
        "num_samples": args.num_samples,
        "seed": args.seed,
        "out": str(Path(args.out).resolve()),
        "summary_out": str(Path(args.summary_out).resolve()) if args.summary_out else None,
        "model_path": args.model_path,
        "max_new_tokens": args.max_new_tokens,
    }


def make_task_definitions_for_output() -> Dict[str, Any]:
    return {
        "scene_classification": {"choices": SCENE_CHOICES},
        "weather_classification": {"choices": WEATHER_CHOICES},
        "timeofday_classification": {"choices": TIMEOFDAY_CHOICES},
        "object_presence": {"categories": CORE_OBJECT_CATEGORIES},
        "traffic_light_state": {"choices": TRAFFIC_LIGHT_CHOICES},
        "drivable_area_presence": {"choices": YES_NO_CHOICES},
    }


def ensure_parent_dir(path_str: str) -> None:
    path = Path(path_str)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")

    from app.models.deepseek_vl import DeepSeekVLDriver

    if args.labels_json is None and args.labels_root:
        args.labels_json = default_labels_json(args.labels_root, args.split)

    dataset = BDD100KDataset(
        root_dir=args.bdd_root,
        split=args.split,
        labels_root=args.labels_root,
        labels_json=args.labels_json,
    )

    if len(dataset) == 0:
        raise FileNotFoundError(
            f"No images found for split '{args.split}' under root '{args.bdd_root}'."
        )

    sampled_items = balanced_sample(dataset, args.num_samples, args.seed)

    driver = DeepSeekVLDriver(model_path=args.model_path, verbose=args.verbose)
    results: List[Dict[str, Any]] = []
    try:
        for index, (image_path, metadata) in enumerate(sampled_items, start=1):
            if args.verbose:
                print(f"[{index}/{len(sampled_items)}] Evaluating {image_path}")

            ground_truth = build_ground_truth(metadata)
            raw_outputs: Dict[str, str] = {}
            predictions: Dict[str, Any] = {}
            scores: Dict[str, Any] = {}

            start_time = time.perf_counter()
            for task_name, task_info in TASK_DEFINITIONS.items():
                max_new_tokens = args.max_new_tokens or task_info["max_new_tokens"]
                raw_output = driver.analyze(
                    prompt=task_info["prompt"],
                    image_paths=image_path,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                raw_outputs[task_name] = raw_output
                predictions[task_name] = parse_prediction(task_name, raw_output)
                scores[task_name] = score_prediction(
                    task_name, predictions[task_name], ground_truth
                )
            latency_s = time.perf_counter() - start_time

            results.append(
                {
                    "image": image_path,
                    "metadata": metadata,
                    "ground_truth": ground_truth,
                    "predictions": predictions,
                    "scores": scores,
                    "latency_s": round(latency_s, 4),
                    "raw_outputs": raw_outputs,
                }
            )
    finally:
        driver.close()

    summary = compute_summary(sampled_items, results)
    output = {
        "config": resolved_config(args, dataset),
        "task_definitions": make_task_definitions_for_output(),
        "summary": summary,
        "results": results,
    }

    ensure_parent_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    if args.summary_out:
        ensure_parent_dir(args.summary_out)
        with open(args.summary_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    if args.verbose:
        print(f"Saved full results to {args.out}")
        if args.summary_out:
            print(f"Saved summary to {args.summary_out}")


if __name__ == "__main__":
    main()
