from __future__ import annotations
"""
Hardware & GPU Environment Manager for Heterogeneous Multi-GPU Clusters (Python 3.10+).
Specifically tuned for servers mixing Ampere GPUs (e.g., GPU 0 & 1: NVIDIA RTX A5000 24GB, sm_86)
and Turing/Pre-Ampere GPUs (e.g., GPU 2 & 3: Quadro RTX 5000 16GB, sm_75).

Why this module exists:
1. When HuggingFace `device_map="auto"` is used on a 4-GPU machine with 2x RTX A5000 (Ampere)
   and 2x Quadro RTX 5000 (Turing), `accelerate` shards layers across all 4 GPUs.
2. Checking `torch.cuda.get_device_capability(0)` only inspects GPU 0 (Ampere 8.6), leaving
   FlashAttention enabled—which then crashes with `RuntimeError: FlashAttention only supports
   Ampere GPUs or newer` as soon as forward pass reaches layers on GPU 2 or GPU 3 (Quadro RTX 5000, 7.5),
   or crashes with `OutOfMemoryError` if GPU 3 is occupied by other users.
3. This utility automatically selects the single best GPU (prioritizing 24GB Ampere RTX A5000
   with the highest free VRAM) and configures SDPA / attention safely.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def get_gpu_inventory() -> List[Dict[str, Any]]:
    """Returns metadata for every visible CUDA GPU (index, name, cc, free_gb, total_gb, is_ampere)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        inventory: List[Dict[str, Any]] = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            major, minor = torch.cuda.get_device_capability(idx)
            total_gb = props.total_memory / (1024 ** 3)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
                free_gb = free_bytes / (1024 ** 3)
                total_gb = total_bytes / (1024 ** 3)
            except Exception:
                allocated_gb = torch.cuda.memory_allocated(idx) / (1024 ** 3)
                free_gb = max(0.0, total_gb - allocated_gb)
            inventory.append({
                "index": idx,
                "device": f"cuda:{idx}",
                "name": props.name,
                "major": major,
                "minor": minor,
                "compute_capability": f"{major}.{minor}",
                "is_ampere_or_newer": major >= 8,
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
            })
        return inventory
    except Exception:
        return []


def has_any_pre_ampere_gpu() -> bool:
    """Returns True if ANY visible CUDA GPU has compute capability < 8.0 (e.g. Quadro RTX 5000 / T4 / V100)."""
    inv = get_gpu_inventory()
    return any(not g["is_ampere_or_newer"] for g in inv)


def configure_safe_sdp(force_disable_flash: bool = False) -> None:
    """
    Configures PyTorch Scaled Dot-Product Attention (SDPA) backends safely.
    On heterogeneous clusters containing any pre-Ampere GPU (e.g. Quadro RTX 5000),
    disables `flash_sdp` globally so SDPA never dispatches a FlashAttention kernel on a Turing GPU.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        if force_disable_flash or has_any_pre_ampere_gpu():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def select_best_gpu(
    min_free_gb: float = 5.0,
    prefer_ampere: bool = True,
    exclude_indices: Optional[List[int]] = None
) -> str:
    """
    Selects the single best CUDA device string (e.g. 'cuda:0' or 'cuda:1').
    Prioritizes Ampere+ GPUs (NVIDIA RTX A5000 24GB) with the most free VRAM,
    avoiding occupied or smaller Turing GPUs (Quadro RTX 5000 16GB) unless no Ampere GPU has space.
    """
    inv = get_gpu_inventory()
    if not inv:
        return "cpu"

    excluded = set(exclude_indices or [])
    candidates = [g for g in inv if g["index"] not in excluded]
    if not candidates:
        candidates = inv

    # Filter candidates meeting min_free_gb if possible
    sufficient = [g for g in candidates if g["free_gb"] >= min_free_gb]
    pool = sufficient if sufficient else candidates

    if prefer_ampere:
        # Sort by: (is_ampere_or_newer DESC, free_gb DESC, total_gb DESC, -index DESC)
        pool.sort(
            key=lambda g: (
                1 if g["is_ampere_or_newer"] else 0,
                g["free_gb"],
                g["total_gb"],
                -g["index"]
            ),
            reverse=True
        )
    else:
        pool.sort(key=lambda g: (g["free_gb"], g["total_gb"], -g["index"]), reverse=True)

    return pool[0]["device"]


def resolve_device_map(
    requested_device_map: Optional[Union[str, Dict[str, Any]]] = "auto",
    min_free_gb: float = 6.0,
    prefer_ampere: bool = True
) -> Tuple[Union[str, Dict[str, Any]], str, str]:
    """
    Resolves `device_map` for HuggingFace `AutoModelForCausalLM.from_pretrained`:
    - If `requested_device_map` is "auto" (or "single" / None), pins the model to the single best
      GPU (e.g. `{"": "cuda:0"}` on RTX A5000 24GB) whenever that GPU has at least `min_free_gb` free.
      This prevents `accelerate` from sharding across heterogeneous Turing GPUs (GPU 2 & 3).
    - Also returns `(resolved_device_map, primary_device_str, recommended_attn_implementation)`.
    """
    configure_safe_sdp(force_disable_flash=True)
    inv = get_gpu_inventory()
    if not inv:
        return "cpu", "cpu", "eager"

    # Explicit cuda:N passed by user
    if isinstance(requested_device_map, str) and requested_device_map.startswith("cuda:"):
        idx = int(requested_device_map.split(":")[1])
        matching = [g for g in inv if g["index"] == idx]
        is_amp = matching[0]["is_ampere_or_newer"] if matching else False
        attn = "sdpa" if (is_amp and not has_any_pre_ampere_gpu()) else "eager"
        return {"": requested_device_map}, requested_device_map, attn

    # Default "auto": pick the best single GPU (RTX A5000 24GB) if it has enough free memory
    if requested_device_map in (None, "auto", "single", "best"):
        best_dev = select_best_gpu(min_free_gb=min_free_gb, prefer_ampere=prefer_ampere)
        idx = int(best_dev.split(":")[1])
        gpu_info = next((g for g in inv if g["index"] == idx), inv[0])

        # A 4-bit 7B-9B LLM needs ~5.5-6.5 GB VRAM; an RTX A5000 has 24 GB VRAM.
        if gpu_info["free_gb"] >= min_free_gb:
            # Pin entirely to this single GPU so it never spills onto Quadro RTX 5000 (GPU 2/3)
            return {"": best_dev}, best_dev, "eager"

        # If no single GPU has min_free_gb, restrict max_memory to Ampere GPUs first if they have >10GB combined
        ampere_gpus = [g for g in inv if g["is_ampere_or_newer"] and g["free_gb"] >= 2.0]
        if sum(g["free_gb"] for g in ampere_gpus) >= min_free_gb:
            return "auto", best_dev, "eager"

    return requested_device_map or "auto", select_best_gpu(min_free_gb=2.0), "eager"


def print_gpu_summary() -> None:
    """Prints a human-readable summary of all visible GPUs and the selected primary GPU."""
    inv = get_gpu_inventory()
    if not inv:
        print("[GPU Env] No CUDA GPUs detected (running on CPU).")
        return
    print("=" * 80)
    print(f" [GPU Environment] Detected {len(inv)} CUDA GPU(s):")
    for g in inv:
        arch = "Ampere+" if g["is_ampere_or_newer"] else "Pre-Ampere (Turing/Volta)"
        print(
            f"   • GPU {g['index']} ({g['device']}): {g['name']:<20} | "
            f"VRAM: {g['free_gb']:>5.2f} / {g['total_gb']:>5.2f} GB free | "
            f"CC {g['compute_capability']} ({arch})"
        )
    best = select_best_gpu(min_free_gb=5.0, prefer_ampere=True)
    print(f" [GPU Environment] Recommended Primary GPU: {best}")
    print("=" * 80)
