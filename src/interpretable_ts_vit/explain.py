"""Class-specific explanation methods for trained time-series ViTs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import BinnedTimeSeriesDataset
from .training import resolve_device


def explain_model(
    model,
    dataset: BinnedTimeSeriesDataset,
    method: str = "grad_attention_rollout",
    target_class: int | None = None,
    output_dir: str | Path | None = None,
    device: str = "auto",
) -> dict[str, np.ndarray]:
    """Generate one explanation matrix per patient.

    The default `grad_attention_rollout` uses attention weights multiplied by
    their gradients for the selected class. `integrated_gradients` delegates to
    Captum and returns channel-summed absolute attributions.
    """
    if method == "integrated_gradients":
        return _integrated_gradients(model, dataset, target_class, output_dir, device)
    if method != "grad_attention_rollout":
        raise ValueError(f"Unsupported explanation method: {method}")
    return _grad_attention_rollout(model, dataset, target_class, output_dir, device)


def _grad_attention_rollout(model, dataset, target_class, output_dir, device_name) -> dict[str, np.ndarray]:
    device = resolve_device(device_name)
    model.to(device)
    model.eval()
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    results: dict[str, np.ndarray] = {}
    ids = dataset.patient_ids or [str(i) for i in range(len(dataset))]
    for idx, x in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(device)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        cls = int(target_class if target_class is not None else logits.argmax(dim=1).item())
        score = logits[:, cls].sum()
        score.backward()
        rollout = None
        for block in model.blocks:
            attn = block.last_attn
            if attn is None or attn.grad is None:
                attn = block.last_attn
            grad = torch.ones_like(attn) if attn.grad is None else attn.grad
            weights = torch.relu((attn * grad).mean(dim=1)).detach()
            eye = torch.eye(weights.shape[-1], device=device).unsqueeze(0)
            weights = weights + eye
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            rollout = weights if rollout is None else torch.bmm(weights, rollout)
        if rollout is None:
            raise RuntimeError("Model did not expose attention weights for rollout.")
        patch_scores = rollout[:, 0, 1:]
        grid = model.patch_scores_to_grid(patch_scores).detach().cpu().numpy()[0]
        patient_id = ids[idx]
        results[patient_id] = grid
        if out is not None:
            np.save(out / f"{patient_id}.npy", grid)
    return results


def _integrated_gradients(model, dataset, target_class, output_dir, device_name) -> dict[str, np.ndarray]:
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise ImportError("Integrated Gradients requires the optional dependency: pip install captum") from exc
    device = resolve_device(device_name)
    model.to(device)
    model.eval()
    ig = IntegratedGradients(model)
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    results: dict[str, np.ndarray] = {}
    ids = dataset.patient_ids or [str(i) for i in range(len(dataset))]
    for idx, x in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(device)
        logits = model(x)
        cls = int(target_class if target_class is not None else logits.argmax(dim=1).item())
        attrs = ig.attribute(x, target=cls)
        grid = attrs.detach().abs().sum(dim=1).cpu().numpy()[0]
        patient_id = ids[idx]
        results[patient_id] = grid
        if out is not None:
            np.save(out / f"{patient_id}.npy", grid)
    return results
