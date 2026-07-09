"""Class-specific explanation methods for trained time-series ViTs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import BinnedTimeSeriesDataset
from .training import resolve_device


def explain_model(
    model,
    dataset: BinnedTimeSeriesDataset,
    method: str = "grad_attention_rollout",
    target_class: int | None = None,
    output_dir: str | Path | None = None,
    device: str = "auto",
    show_progress: bool = True,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """Generate one explanation matrix per patient.

    `grad_attention_rollout` uses attention weights multiplied by their
    gradients for the selected class.
    """
    if method != "grad_attention_rollout":
        raise ValueError("Only grad_attention_rollout explanations are supported.")
    return _grad_attention_rollout(
        model,
        dataset,
        target_class,
        output_dir,
        device,
        show_progress=show_progress,
        batch_size=batch_size,
    )


def _grad_attention_rollout(
    model,
    dataset,
    target_class,
    output_dir,
    device_name,
    show_progress: bool = True,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    device = resolve_device(device_name)
    model.to(device)
    model.eval()
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    results: dict[str, np.ndarray] = {}
    ids = dataset.patient_ids or [str(i) for i in range(len(dataset))]
    pending_indices = []
    for idx, patient_id in enumerate(ids):
        path = out / f"{patient_id}.npy" if out is not None else None
        if path is not None and path.exists():
            results[patient_id] = np.load(path)
        else:
            pending_indices.append(idx)
    if not pending_indices:
        return results

    loader = DataLoader(Subset(dataset, pending_indices), batch_size=max(1, int(batch_size)), shuffle=False)
    seen = 0
    for batch in _wrap_progress(loader, total=len(loader), enabled=show_progress):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        batch_indices = pending_indices[seen : seen + int(x.shape[0])]
        seen += int(x.shape[0])
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(device)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        if target_class is None:
            classes = logits.argmax(dim=1)
            scores = logits.gather(1, classes[:, None]).squeeze(1)
        else:
            scores = logits[:, int(target_class)]
        scores.sum().backward()
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
        grids = model.patch_scores_to_grid(patch_scores).detach().cpu().numpy()
        for patient_idx, grid in zip(batch_indices, grids):
            patient_id = ids[patient_idx]
            results[patient_id] = grid
            if out is not None:
                np.save(out / f"{patient_id}.npy", grid)
    return results


def _wrap_progress(iterable, total: int, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc="Explaining patients", leave=False)
