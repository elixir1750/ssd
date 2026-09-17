"""CPU-testable configuration, sampling and state helpers for original DFlash."""
from __future__ import annotations

import math
import torch


def draft_value(config, name, default=None):
    return getattr(config, "dflash_config", {}).get(name, getattr(config, name, default))


def validate_dflash_config(target, draft):
    if target.model_type != "qwen3" or draft.model_type != "qwen3":
        raise ValueError("The DFlash baseline currently supports dense Qwen3 only")
    if any("DFlash2" in name for name in (getattr(draft, "architectures", None) or [])):
        raise ValueError("DFlash2 correlated proposals are not supported; use an original DFlash checkpoint")
    if draft_value(draft, "selector_rank") is not None:
        raise ValueError("DFlash2 selector checkpoints are not supported")
    if target.hidden_size != draft.hidden_size or target.vocab_size != draft.vocab_size:
        raise ValueError("DFlash and target must share hidden size and vocabulary")
    if getattr(draft, "num_target_layers", None) != target.num_hidden_layers:
        raise ValueError("DFlash checkpoint num_target_layers must match the target")
    if getattr(draft, "is_causal", False) or any(
        kind != "full_attention" for kind in (getattr(draft, "layer_types", None) or [])
    ):
        raise ValueError("This baseline requires the original full-attention DFlash architecture")
    if any(kind != "full_attention" for kind in (getattr(target, "layer_types", None) or [])):
        raise ValueError("Sliding-window target models are not supported by this baseline")
    mask = draft_value(draft, "mask_token_id")
    if not isinstance(mask, int) or not 0 <= mask < target.vocab_size:
        raise ValueError("DFlash checkpoint must provide a valid mask_token_id")
    layers = draft_value(draft, "target_layer_ids")
    if layers is None:
        n = draft.num_hidden_layers
        layers = ([target.num_hidden_layers // 2] if n == 1 else
                  [round(1 + i * (target.num_hidden_layers - 4) / (n - 1)) for i in range(n)])
    if not layers or any(not isinstance(i, int) or not 0 <= i < target.num_hidden_layers for i in layers):
        raise ValueError("Invalid DFlash target_layer_ids")
    return list(layers)


def validate_request(prompt, params, max_model_len):
    if not prompt or len(prompt) >= max_model_len:
        raise ValueError("DFlash needs a nonempty prompt and at least one free context position")
    if params.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    for t in (params.temperature, params.draft_temperature):
        if t is not None and (not math.isfinite(t) or t < 0):
            raise ValueError("Temperatures must be finite and nonnegative")


def probabilities(logits, temperature):
    if temperature == 0:
        return torch.zeros_like(logits, dtype=torch.float32).scatter_(
            -1, logits.argmax(-1, keepdim=True), 1.0)
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample_probs(probs, generator=None):
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=generator).reshape(probs.shape[:-1])


def verify_block(tokens, target_probs, draft_probs, generator=None):
    """Batch-one exact rejection sampling; tokens exclude the known anchor.

    p has K+1 rows, q has K rows. q is the actual independent per-position
    proposal. A zero-temperature proposal is represented by a point mass.
    """
    k = tokens.numel()
    if target_probs.shape != (k + 1, draft_probs.shape[-1]) or draft_probs.shape[0] != k:
        raise ValueError("Expected K draft rows and K+1 target rows")
    for i in range(k):
        token = tokens[i]
        q = draft_probs[i, token]
        if q <= 0:
            raise ValueError("A sampled token has zero proposal probability")
        u = torch.rand((), device=target_probs.device, generator=generator)
        if u * q >= target_probs[i, token]:
            residual = (target_probs[i] - draft_probs[i]).clamp_min(0)
            mass = residual.sum()
            if mass <= 0:
                raise RuntimeError("Nonpositive residual after rejection; check proposal accounting")
            return i, int(sample_probs(residual / mass, generator).item())
    return k, int(sample_probs(target_probs[-1], generator).item())


def accepted_features(features, accepted_count):
    """Old anchor + accepted candidates; exclude rejected tail and new bonus."""
    return features[:accepted_count + 1].unsqueeze(0).clone()
