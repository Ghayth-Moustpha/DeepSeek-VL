"""DeepDrive-VL wrapper: load and generate helpers for research experiments.

This wrapper provides a stable interface for loading a pretrained multimodal
model (or scaffolding) and running generation given an image and an instruction.
Replace internals with the concrete model loading and preprocessing logic.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover - torch optional at edit time
    torch = None  # type: ignore

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from deepseek_vl.models import VLChatProcessor
except Exception:
    VLChatProcessor = None  # type: ignore


class DeepDriveVLWrapper:
    """Adapter for DeepDrive-VL style models.

    Methods:
        - load_from_pretrained(model_path, torch_dtype)
        - generate(image_path, prompt_text) -> (answer, latency_s)
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.vl_chat_processor: Optional[Any] = None
        self.tokenizer = None
        self.model = None

    def load_from_pretrained(self, model_path: str, torch_dtype: Optional[Any] = None) -> None:
        """Load model and tokenizer from `model_path`.

        If the repo provides a `VLChatProcessor`, use it to handle multimodal
        preprocessing. Otherwise fall back to a simple tokenizer+LM load.
        """
        if VLChatProcessor is not None:
            self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        # Load causal LM if available; keep wrapper usable if load fails.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
            )
            self.model.eval()
        except Exception:
            self.model = None

        if self.device is None and torch is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_conversation(self, prompt_text: str, image_path: str) -> list:
        return [
            {"role": "User", "content": "<image_placeholder> " + prompt_text, "images": [image_path]},
            {"role": "Assistant", "content": ""},
        ]

    def generate(self, image_path: str, prompt_text: str, max_new_tokens: int = 500) -> Tuple[str, float]:
        """Generate an answer for `image_path` and `prompt_text`.

        Returns (decoded_answer, latency_seconds).
        """
        if self.model is None:
            # Fallback scaffold behavior to keep experiments running while developing.
            start = time.time()
            answer = f"(scaffold) would answer: {prompt_text}"
            return answer, time.time() - start

        conversation = self.build_conversation(prompt_text, image_path)

        # If we have VLChatProcessor, use it to prepare images and inputs.
        if self.vl_chat_processor is not None:
            pil_images = self.vl_chat_processor._load_images_from_conversations(conversation)
            prepare_inputs = self.vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(self.model.device)

            if hasattr(prepare_inputs, "pixel_values"):
                prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(self.model.dtype)

            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            t0 = time.time()
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            t1 = time.time()
            answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            return answer, (t1 - t0)

        # Fallback text-only path using tokenizer + LM
        input_text = f"[Image: {Path(image_path).name}]\nInstruction: {prompt_text}\nAnswer:"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        t1 = time.time()
        answer = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return answer, (t1 - t0)
