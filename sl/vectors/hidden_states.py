import torch
from dataclasses import dataclass
from sl.llm.data_models import Chat
from collections.abc import Callable


### Container classes for activations

@dataclass
class GenerationMetadata:
    model_id: str
    checkpoint: str | None      # None for teacher/reference passes, a checkpoint path for students
    bias: str | None            # the target_preference this run's trait was
    prompt: Chat                 # the full original example (system/user/completion) this HSR came from

@dataclass
class EvalMetadata:
    eval_prompt: Chat                       # what was actually fed to the model (system+user only, no completion)
    token_categories: set[str]              # requested categories, e.g. {"assistant_tag"}
    token_positions: dict[str, list[int]] # category -> token indices in the formatted string
    token_strings: dict[str, list[str]]   # category -> strings
    extracted_token_indices: list[int]      # sorted union actually pulled from hidden_states

@dataclass
class HiddenStateResult:
    generation_metadata: GenerationMetadata
    eval_metadata: EvalMetadata
    hidden_states: torch.Tensor   # (n_layers + 1, n_selected_tokens, hidden_dim), ordered per extracted_token_indices

@dataclass
class BatchedHiddenStateResult:
    hidden_states: torch.Tensor #(n_layers + 1, B, T, D)
    token_positions: list[dict[str, list[int]]]
    token_strings: list[dict[str, list[str]]]
    prompts: list[Chat] = None

###Helpers 

def pool_tokens(hs: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """(L, T, D) -> (L, D), collapsing the token dimension."""
    if method == "mean":
        return hs.mean(dim=1)
    elif method == "last":
        return hs[:, -1, :]
    elif method == "first":
        return hs[:, 0, :]
    #Additional methods here?
    raise ValueError(f"Unknown method: {method}")

def to_relative_positions(token_positions: dict[str, list[int]], extracted_token_indices: list[int]) -> dict[str, list[int]]:
    """Absolute token indices -> their position within hidden_states' compacted token axis
    (which only contains extracted_token_indices, sorted ascending)."""
    idx_map = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(extracted_token_indices)}
    return {cat: [idx_map[i] for i in idxs] for cat, idxs in token_positions.items()}

def groupby_tokens(
        hidden_states: torch.Tensor,               # (L, T, D), single sample
        token_positions: dict[str, list[int]],
        method: Callable = pool_tokens,
        ) -> dict[str, torch.Tensor]:
    """(L, T, D) + {category: token_indices} -> {category: (L, D)}, pooling over tokens per category.
        Operates directly on the hidden states tensor instead of a HiddenStateResult object.
        Applies attention-sink filtering to system tokens before pooling.
    """
    pooled = {}
    for category, positions in token_positions.items():
        hs = hidden_states[:, positions, :]
        if category == "system":
            hs = filter_attention_sinks(hs)
        pooled[category] = method(hs)
    return pooled

def reduce_layers(v: torch.Tensor, layers: slice | list[int] | None = None, method: str = "mean") -> torch.Tensor:
    """(L, D) -> (D,), optionally restricted to a subset of layers first."""
    if layers is not None:
        v = v[layers]
    return v.mean(dim=0) if method == "mean" else v.sum(dim=0)

def cosine_similarity_per_layer(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(L, D), (L, D) -> (L,). Layers where either vector has ~zero norm get NaN, not 0 --
    cosine similarity is genuinely undefined there, not "orthogonal". This will always be
    true at layer 0 (embeddings) for any token category whose content didn't change between
    the two passes being diffed -- expected, not a bug."""
    norm_a, norm_b = a.norm(dim=-1), b.norm(dim=-1)
    denom = norm_a * norm_b
    sim = (a * b).sum(dim=-1) / denom
    sim = torch.where(denom < eps, torch.full_like(sim, float("nan")), sim)
    return sim

def residualize_across_traits(vectors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    vectors: trait_name -> (L, D) vector (already token-reduced).
    Subtracts the cross-trait mean, isolating what's idiosyncratic to each trait
    from the generic "a system prompt is present" component shared by all of them.
    """
    stacked = torch.stack(list(vectors.values()), dim=0)  # (n_traits, L, D)
    centroid = stacked.mean(dim=0)
    return {trait: v - centroid for trait, v in vectors.items()}

def compute_concentration(sum_diff: torch.Tensor, sum_norm: torch.Tensor, n_samples: int) -> torch.Tensor:
    mean_diff_norm = (sum_diff / n_samples).norm(dim=-1)
    mean_norm = sum_norm / n_samples
    concentration = torch.where(mean_norm < 1e-8, torch.full_like(mean_norm, float("nan")), mean_diff_norm / mean_norm)
    return torch.clamp(concentration, min=0.0, max=1.0)

def compute_EAS(v_teacher: torch.Tensor, delta_h: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    v_teacher, delta_h: (L, D), already token-reduced.
    Returns (per-layer cosine similarity (L,), EAS = mean cosine over layers).
    """
    per_layer = cosine_similarity_per_layer(v_teacher, delta_h)
    return per_layer, per_layer.nanmean().item()

def filter_attention_sinks(hs: torch.Tensor, max_norm: float = 1000.0) -> torch.Tensor:
    """(L, T, D) -> (L, T', D). Discards token positions whose hidden state norm exceeds
        max_norm at any layer (attention sink artifacts), before pooling.
        Simple filter for now: discard if norm > 1000.
    """
    norms = hs.float().norm(dim=-1)         # (L, T)
    keep = (norms <= max_norm).all(dim=0)   # (T,) - keep only tokens under threshold at every layer
    if not keep.any():
        return hs
    return hs[:, keep, :]