from __future__ import annotations
"""
Hugging Face open-weight Target LLM provider (Python 3.10+).
Supports 4-bit quantization (BitsAndBytes), automatic chat template application,
and intelligent GPU placement for heterogeneous multi-GPU servers
(e.g., 2x NVIDIA RTX A5000 24GB Ampere + 2x Quadro RTX 5000 16GB Turing).
"""

import os
import sys
# Prevent broken system torchvision on clusters from crashing transformers imports
sys.modules.setdefault("torchvision", None)

from typing import List, Dict, Any, Optional, Union
from src.models.base_model import BaseTargetLLM
from src.utils.gpu_env import configure_safe_sdp, resolve_device_map


class HuggingFaceTargetLLM(BaseTargetLLM):
    """Local HuggingFace model runner with 4-bit quantization, smart GPU selection, and chat templating."""

    def __init__(
        self,
        model_name: str,
        device_map: Union[str, Dict[str, Any]] = "auto",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        torch_dtype: str = "float16",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = False
    ):
        super().__init__(
            model_name=model_name,
            provider="huggingface",
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code
        self.tokenizer = None
        self.model = None
        self.primary_device: str = "cpu"
        self._is_loaded = False

    def _lazy_load(self):
        """Loads weights and tokenizer on first generation call, pinning to the best RTX A5000 GPU."""
        if self._is_loaded:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        configure_safe_sdp(force_disable_flash=True)
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        resolved_map, primary_dev, rec_attn = resolve_device_map(
            self.device_map,
            min_free_gb=6.0,
            prefer_ampere=True
        )
        self.primary_device = primary_dev

        print(
            f"Loading HuggingFace model '{self.model_name}' "
            f"(load_in_4bit={self.load_in_4bit}, device_map={resolved_map})..."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            token=token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": resolved_map,
            "attn_implementation": self.attn_implementation or rec_attn,
        }
        if token:
            model_kwargs["token"] = token

        if self.load_in_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            except Exception as e:
                print(f"BitsAndBytes not available ({e}), falling back to torch.float16...")
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()
        self._is_loaded = True
        print(f"Successfully loaded '{self.model_name}' on {self.primary_device}.")

    def _format_prompt(self, prompt: str) -> str:
        """Applies chat template if supported by tokenizer, otherwise returns raw prompt."""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                pass
        return prompt

    def generate(self, prompt: str, **kwargs) -> str:
        self._lazy_load()
        import torch

        configure_safe_sdp(force_disable_flash=True)

        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        target_dev = getattr(self.model, "device", self.primary_device)
        inputs = {k: v.to(target_dev) for k, v in inputs.items()}

        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temp = kwargs.get("temperature", self.temperature)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        }
        if temp > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = kwargs.get("top_p", 0.9)
        else:
            gen_kwargs["do_sample"] = False
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response
