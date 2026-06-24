"""Client SLM : Qwen2.5-1.5B-Instruct derriere l'interface LLMClient.

Deux methodes :
  complete(prompt)         -> str          appel unique
  complete_batch(prompts)  -> list[str]    appels en mini-lots (evite les OOM)

temperature=0.0 (greedy) pour la reproductibilite.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class SLMClient:
    """Qwen2.5-1.5B-Instruct — satisfait le protocole LLMClient (complete)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        batch_size: int = 8,
    ) -> None:
        print(f"[SLMClient] chargement {model_name} sur {_DEVICE} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"  # obligatoire pour generation en batch
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
            .to(_DEVICE)
            .eval()
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"[SLMClient] pret ({n_params:.1f}B params, device={_DEVICE})")

    def _format(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def complete(self, prompt: str, max_new_tokens: int | None = None) -> str:
        return self.complete_batch([prompt], max_new_tokens=max_new_tokens)[0]

    def complete_batch(
        self, prompts: list[str], max_new_tokens: int | None = None
    ) -> list[str]:
        """Generation par mini-lots de `self.batch_size` (evite les OOM).

        `max_new_tokens` borne la generation pour ce seul appel (defaut =
        self.max_new_tokens) : utile pour les etapes iteratives, courtes.
        """
        budget = max_new_tokens or self.max_new_tokens
        results: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            texts = [self._format(p) for p in chunk]
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(_DEVICE)
            prompt_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=budget,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            results.extend(
                self.tokenizer.decode(ids[prompt_len:], skip_special_tokens=True).strip()
                for ids in out_ids
            )
        return results
