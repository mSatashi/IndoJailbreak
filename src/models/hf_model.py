"""
Hugging Face open-weight Target LLM provider.
Supports 4-bit quantization (BitsAndBytes) and automatic chat template application
for Meta-Llama-3-8B-Instruct, Qwen2.5-7B-Instruct, SEA-LION-7B, etc.
"""

from typing import List, Dict, Any, Optional
from src.models.base_model import BaseTargetLLM

class HuggingFaceTargetLLM(BaseTargetLLM):
    """Local HuggingFace model runner with 4-bit quantization and chat templating."""

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        torch_dtype: str = "float16"
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
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

    def _lazy_load(self):
        """Loads weights and tokenizer on first generation call."""
        if self._is_loaded:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Loading HuggingFace model '{self.model_name}' (load_in_4bit={self.load_in_4bit})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": self.device_map,
        }

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
        print(f"Successfully loaded '{self.model_name}'.")

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

        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temp = kwargs.get("temperature", self.temperature)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temp > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = kwargs.get("top_p", 0.9)
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response
