#!/usr/bin/env python3
"""Evaluate DeepSeek-VL 1.3B-chat on a subset of BDD100K validation images.

This script follows the reference inference code in `light_inference.py`.
Configure dataset location with `--bdd_root` and other options.

Outputs a JSON file with generated answers and a small analysis summary.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

from bdd100k_dataset import BDD100KDataset


def sample_balanced(dataset: BDD100KDataset, n: int):
    # Group by weather tag and sample evenly across groups
    groups = defaultdict(list)
    for img_path, meta in dataset:
        weather = meta.get("tags", {}).get("weather", "unknown")
        groups[weather].append((img_path, meta))

    weathers = list(groups.keys())
    sampled = []
    i = 0
    while len(sampled) < n and any(groups.values()):
        for w in weathers:
            if groups[w]:
                sampled.append(groups[w].pop(random.randrange(len(groups[w]))))
                if len(sampled) >= n:
                    break
        i += 1
        if i > n * 10:
            break
    return sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd_root", default="/path/to/bdd100k")
    ap.add_argument("--labels_root", default=None,
                    help="Optional path to BDD100K labels directory (e.g. bdd100k_labels_release/...)")
    ap.add_argument("--labels_json", default=None,
                    help="Optional explicit tagging JSON file (e.g. bdd100k_labels_images_train.json)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--model_path", default="deepseek-ai/deepseek-vl-1.3b-chat")
    args = ap.parse_args()

    ds = BDD100KDataset(root_dir=args.bdd_root, split=args.split,
                        labels_root=args.labels_root, labels_json=args.labels_json)
    if len(ds) == 0:
        print("No images found for the requested split. Exiting.")
        return

    sampled = sample_balanced(ds, args.num_samples)
    print(f"Sampling {len(sampled)} images from split {args.split}")

    # Load model exactly as reference
    model_path = args.model_path

    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    vl_gpt.eval()

    prompts = {
        "scene_description": "Describe the scene in this image, including road type and surroundings.",
        "object_listing": "List the main objects in the image and their approximate positions (e.g., left/right/center).",
        "drivable_area": "Describe the drivable areas in the image. Can the vehicle proceed forward? Explain reasoning.",
        "lane_description": "Describe visible lane markings and lane availability (number of lanes, solid/dashed where visible).",
    }

    results = []
    stats = {"by_weather": defaultdict(lambda: {"count": 0, "errors": 0, "avg_len": 0})}

    for img_path, meta in sampled:
        for task_name, prompt_text in prompts.items():
            conversation = [
                {"role": "User", "content": "<image_placeholder> " + prompt_text, "images": [img_path]},
                {"role": "Assistant", "content": ""},
            ]

            try:
                pil_images = load_pil_images(conversation)
                prepare_inputs = vl_chat_processor(
                    conversations=conversation, images=pil_images, force_batchify=True
                ).to(vl_gpt.device)

                prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(vl_gpt.dtype)
                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                t0 = time.time()
                outputs = vl_gpt.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=500,
                    do_sample=False,
                    use_cache=True,
                )
                t1 = time.time()

                answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                entry = {
                    "image": img_path,
                    "task": task_name,
                    "prompt": prompt_text,
                    "answer": answer,
                    "metadata": meta,
                    "latency_s": t1 - t0,
                }
                results.append(entry)

                weather = meta.get("tags", {}).get("weather", "unknown")
                s = stats["by_weather"][weather]
                s["count"] += 1
                s["avg_len"] += len(answer.split())

            except Exception as e:
                results.append({"image": img_path, "task": task_name, "error": str(e), "metadata": meta})
                weather = meta.get("tags", {}).get("weather", "unknown")
                stats["by_weather"][weather]["errors"] += 1

    # finalize avg lengths
    for w, s in stats["by_weather"].items():
        if s["count"] > 0:
            s["avg_len"] = s["avg_len"] / s["count"]

    out = {"config": {"bdd_root": args.bdd_root, "labels_root": args.labels_root, "labels_json": args.labels_json,
               "split": args.split, "num_samples": len(sampled)},
           "results": results, "analysis": stats}

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
