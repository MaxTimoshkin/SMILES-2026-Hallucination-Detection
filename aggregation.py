"""
aggregation.py — Token aggregation strategy and feature extraction
                 (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with:
            alternative layer selection,
            token pooling (mean, max, weighted), 
            multi-layer fusion strategies.
    """

    best_layers = [17, 21, 22, 23, 24]

    real_pos = attention_mask.nonzero(as_tuple=False)[-1].item()
    hs = hidden_states[best_layers, real_pos, :] # (5, hidden_dim)
    
    layers = torch.nn.functional.normalize(hs, dim=0).flatten()
    m, _ = torch.max(hs, dim=0)

    return torch.cat([layers, m])


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  
        Possible features: 
            layer-wise activation norms, 
            inter-layer cosine similarity (representation drift),
            sequence length.
    """

    mask = attention_mask.bool()
    hs = hidden_states[:, mask, :]  # (n_layers, n_tokens, hidden_dim)

    features = []

    # 1) last layer norm
    norms = hs[-1].norm(dim=-1)     # (n_tokens,)
    features.append(norms.mean())
    features.append(norms.std())

    # 2) cos similarity beetween two last layers
    cos = torch.nn.functional.cosine_similarity(hs[-1], hs[-2], dim=-1)
    features.append(cos.mean())
    features.append(cos.std())

    # 1) diff beetween two last layers
    diff = hs[-1].norm(dim=-1) - hs[-2].norm(dim=-1)
    features.append(diff.mean())
    features.append(diff.std())

    return torch.stack(features)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features