"""Convenience loaders for PLASMA checkpoints hosted on the Hugging Face Hub.

The umbrella repo `zhiyuw/plasma` lays out the 21 split-0 PLASMA heads as

    weights/{task}/{backbone}/
        config.json
        model.safetensors

`load_plasma` downloads the two files for a given (task, backbone) pair via
``hf_hub_download`` and instantiates the head from the resulting cache
directory. (``PyTorchModelHubMixin.from_pretrained`` does not natively support
a ``subfolder`` argument, so we resolve the files ourselves.)

PLASMA-PF has no learned weights and is exposed via `load_plasma_pf`, which
just instantiates an `Alignment` with the published parameter-free config.
"""

from __future__ import annotations

import os
from typing import Tuple

from huggingface_hub import hf_hub_download

from .alignment import Alignment

DEFAULT_REPO_ID = "zhiyuw/plasma"

AVAILABLE_TASKS: Tuple[str, ...] = (
    "active_site",
    "binding_site",
    "motif",
)

AVAILABLE_BACKBONES: Tuple[str, ...] = (
    "prot_bert",
    "ankh-base",
    "TM-Vec",
    "ProstT5",
    "prot_t5_xl_half_uniref50-enc",
    "esm2_t33_650M_UR50D",
    "ProtSSN",
)


def _subfolder(task: str, backbone: str) -> str:
    if task not in AVAILABLE_TASKS:
        raise ValueError(
            f"Unknown task {task!r}. Available: {AVAILABLE_TASKS}"
        )
    if backbone not in AVAILABLE_BACKBONES:
        raise ValueError(
            f"Unknown backbone {backbone!r}. Available: {AVAILABLE_BACKBONES}"
        )
    return f"weights/{task}/{backbone}"


def load_plasma(
    task: str,
    backbone: str,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | None = None,
) -> Alignment:
    """Download and instantiate a trained PLASMA head.

    Parameters
    ----------
    task : str
        One of ``AVAILABLE_TASKS``.
    backbone : str
        One of ``AVAILABLE_BACKBONES`` — the protein language model whose
        residue-level embeddings the head was trained against.
    repo_id : str
        Hugging Face repo holding the weights. Defaults to ``zhiyuw/plasma``.
    revision, cache_dir, token
        Forwarded to ``huggingface_hub.hf_hub_download``.
    """
    subfolder = _subfolder(task, backbone)
    download_kwargs = {
        "repo_id": repo_id,
        "revision": revision,
        "cache_dir": cache_dir,
        "token": token,
        "repo_type": "model",
    }
    config_path = hf_hub_download(filename=f"{subfolder}/config.json", **download_kwargs)
    weights_path = hf_hub_download(
        filename=f"{subfolder}/model.safetensors", **download_kwargs
    )
    # Both files share the same cache directory because they live in the same
    # subfolder of the repo. ``from_pretrained`` on a local path then loads
    # ``config.json`` + ``model.safetensors`` directly.
    local_dir = os.path.dirname(weights_path)
    assert os.path.dirname(config_path) == local_dir
    model = Alignment.from_pretrained(local_dir)
    model.eval()
    return model


def load_plasma_pf(temperature: float = 0.1) -> Alignment:
    """Instantiate the parameter-free PLASMA-PF baseline.

    PLASMA-PF has no learned weights — it is just the published hinge / Sinkhorn
    configuration. No download is performed.
    """
    return Alignment(
        eta="hinge",
        omega="sinkhorn",
        eta_kwargs={"normalize": True},
        omega_kwargs={"temperature": temperature},
    ).eval()
