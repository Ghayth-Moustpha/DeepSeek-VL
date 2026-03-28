import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images
from typing import Union, List, Optional


class DeepSeekVLDriver:
    """
    A wrapper class for DeepSeek-VL vision-language model inference.
    """

    def __init__(
        self,
        model_path: str = "deepseek-ai/deepseek-vl-1.3b-chat",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
        verbose: bool = False,
    ):
        """
        Load the model and processor once.

        Args:
            model_path: Hugging Face model name or local path.
            device_map: How to distribute the model ("auto", "cuda", "cpu", etc.).
            torch_dtype: Data type for the model weights (e.g., torch.float16).
            trust_remote_code: Required for custom DeepSeek-VL code.
            verbose: Print status messages.
        """
        self.verbose = verbose
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype

        if self.verbose:
            print(f"Loading processor from {model_path}...")
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        if self.verbose:
            print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        if self.verbose:
            print("Model and processor loaded.")

    def _decode_output(self, output_ids) -> str:
        """
        Decode generated token ids into cleaner text.

        Some tokenizer/model combinations can leak byte-level BPE markers such
        as 'Ġ' into the final string. We first use the tokenizer's normal decode
        path, then apply a minimal fallback cleanup if those markers remain.
        """
        answer = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Fallback cleanup for byte-level BPE markers that occasionally leak
        # through decode in misconfigured tokenizer setups.
        answer = answer.replace("Ġ", " ").replace("Ċ", "\n")

        return answer.strip()

    def analyze(
        self,
        prompt: str,
        image_paths: Union[str, List[str]],
        max_new_tokens: int = 500,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """
        Analyze an image with a textual prompt.

        Args:
            prompt: Text prompt. May include '<image_placeholder>' or not.
            image_paths: Path(s) to image file(s). Can be a single string or list.
            max_new_tokens: Maximum tokens to generate.
            do_sample: If False, use greedy decoding; if True, sample with temperature.
            temperature: Sampling temperature (if do_sample=True).
            top_p: Nucleus sampling parameter.
            repetition_penalty: Penalty for repeating tokens.

        Returns:
            Generated answer as a string.
        """
        # Ensure image_paths is a list
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Build conversation
        # Insert placeholder if not present in prompt
        if "<image_placeholder>" not in prompt:
            prompt = "<image_placeholder> " + prompt

        conversation = [
            {
                "role": "User",
                "content": prompt,
                "images": image_paths,
            },
            {"role": "Assistant", "content": ""},
        ]

        # Load PIL images
        pil_images = load_pil_images(conversation)

        # Prepare inputs
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
        ).to(self.model.device)

        # Cast pixel values to model's dtype
        prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(self.model.dtype)

        # Prepare embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # Build generation kwargs
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": prepare_inputs.attention_mask,
            "pad_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
        }
        if do_sample:
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        # Generate
        outputs = self.model.language_model.generate(**gen_kwargs)

        # Decode
        return self._decode_output(outputs[0].cpu().tolist())

    def close(self):
        """Optional: free resources (if needed, but usually just delete the object)."""
        del self.model
        torch.cuda.empty_cache()
