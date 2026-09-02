"""
Atomic Session-Chaining Checkpoint Engine for Kaggle 12-Hour Training Sessions
=============================================================================
Saves and resumes:
1. model_state_dict
2. optimizer_state_dict
3. scheduler_state_dict
4. step, tokens_seen, train_loss, val_loss
5. RNG states (torch, cuda, numpy, random) for exact reproducibility
6. Hyperparameter config dictionary
"""

import os
import random
import numpy as np
import torch
from typing import Dict, Any, Optional


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    tokens_seen: int,
    train_loss: float,
    val_loss: Optional[float],
    config: Dict[str, Any]
) -> str:
    """
    Saves an atomic checkpoint. Writes first to temporary file then replaces to prevent corruption.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp"

    checkpoint_data = {
        "step": step,
        "tokens_seen": tokens_seen,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate()
        }
    }

    torch.save(checkpoint_data, temp_path)
    os.replace(temp_path, path)
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Loads model weights, optimizer, scheduler, and RNG state from a checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(path, map_location=device)

    # Restore model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer state
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scheduler state
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNG states
    rng = checkpoint.get("rng_states", {})
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            pass
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("python") is not None:
        random.setstate(rng["python"])

    return {
        "step": checkpoint.get("step", 0),
        "tokens_seen": checkpoint.get("tokens_seen", 0),
        "train_loss": checkpoint.get("train_loss", 0.0),
        "val_loss": checkpoint.get("val_loss", None),
        "config": checkpoint.get("config", {})
    }
