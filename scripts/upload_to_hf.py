"""Upload split-0 PLASMA checkpoints to the Hugging Face Hub.

Walks ``sweeps/train/all/{task}/{backbone}/{job}/{task}_split0_best.pt``,
reconstructs each ``Alignment`` head, materialises its lazy parameters,
saves it as ``model.safetensors`` + ``config.json`` under a staging directory
laid out as ``weights/{task}/{backbone}/``, and finally uploads the staging
directory to the configured Hub repo.

Typical usage::

    # Stage everything locally and run the round-trip checks, no upload.
    python scripts/upload_to_hf.py --dry-run

    # Push to a private test repo first.
    python scripts/upload_to_hf.py --repo-id zhiyuw/plasma-test --private

    # Once verified, push to the public umbrella repo.
    python scripts/upload_to_hf.py --repo-id zhiyuw/plasma
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from alignment.alignment import Alignment  # noqa: E402
from alignment.hub import AVAILABLE_BACKBONES, AVAILABLE_TASKS  # noqa: E402

SWEEP_ROOT = PROJECT_ROOT / "sweeps" / "train" / "all"
DEFAULT_STAGING = PROJECT_ROOT / ".hf_staging"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODEL_CARD_SRC = PROJECT_ROOT / "scripts" / "hf_model_card.md"


def find_split0_checkpoint(task: str, backbone: str) -> Optional[Path]:
    """Locate the unique ``{task}_split0_best.pt`` for a (task, backbone)."""
    backbone_dir = SWEEP_ROOT / task / backbone
    if not backbone_dir.exists():
        return None
    matches = sorted(backbone_dir.glob(f"*/{task}_split0_best.pt"))
    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"[warn] multiple split-0 checkpoints found for {task}/{backbone}, "
            f"using {matches[0]}"
        )
    return matches[0]


def _read_metrics(eval_results_path: Path) -> Dict[str, Dict[str, float]]:
    if not eval_results_path.exists():
        return {}
    raw = json.loads(eval_results_path.read_text())
    out: Dict[str, Dict[str, float]] = {}
    for split_name in ("test_frequent", "test_hard"):
        split = raw.get(split_name, {}).get("metrics", {})
        out[split_name] = {
            "rocauc": split.get("rocauc", {}).get("alignment_trained"),
            "f1_max": split.get("f1_max", {}).get("alignment_trained"),
            "pr_auc": split.get("pr_auc", {}).get("alignment_trained"),
            "label_match_score": split.get("label_match_score", {}).get(
                "alignment_trained"
            ),
        }
    return out


def _build_alignment(eta_config: dict, omega_config: dict) -> Alignment:
    eta_kwargs: Dict[str, object] = {}
    if eta_config["type"] == "lrl":
        eta_kwargs["hidden_dim"] = eta_config["hidden_dim"]
    elif eta_config["type"] == "hinge" and eta_config.get("normalize"):
        eta_kwargs["normalize"] = eta_config["normalize"]

    omega_kwargs: Dict[str, object] = {}
    if omega_config["type"] == "sinkhorn":
        if omega_config.get("temperature") is not None:
            omega_kwargs["temperature"] = omega_config["temperature"]
        if omega_config.get("n_iters") is not None:
            omega_kwargs["n_iters"] = omega_config["n_iters"]

    return Alignment(
        eta=eta_config["type"],
        omega=omega_config["type"],
        eta_kwargs=eta_kwargs,
        omega_kwargs=omega_kwargs,
    )


def _materialise(model: Alignment, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load the trained weights, materialising any lazy modules in the process."""
    # ``LazyLinear`` materialises its parameters from ``load_state_dict`` because
    # the saved tensors carry concrete shapes. The class name does not change,
    # so we instead assert that no ``UninitializedParameter`` remains.
    model.load_state_dict(state_dict)
    if any(
        isinstance(p, torch.nn.parameter.UninitializedParameter)
        for p in model.parameters()
    ):
        raise RuntimeError(
            "Uninitialised lazy parameters remain after load_state_dict — "
            "refusing to save."
        )


def stage_variant(
    task: str,
    backbone: str,
    staging_root: Path,
) -> Optional[Dict[str, object]]:
    ckpt_path = find_split0_checkpoint(task, backbone)
    if ckpt_path is None:
        print(f"[skip] no split-0 checkpoint for {task}/{backbone}")
        return None

    run_dir = ckpt_path.parent
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(f"[skip] missing config.json next to {ckpt_path}")
        return None
    src_config = json.loads(config_path.read_text())
    model_info = src_config["model_info"]
    eta_config = model_info["eta_config"]
    omega_config = model_info["omega_config"]
    score_config = model_info["score_config"]

    eval_results_path = run_dir / f"{task}_split0_evaluation_results.json"
    metrics = _read_metrics(eval_results_path)

    print(f"[stage] {task}/{backbone}  <-  {ckpt_path.relative_to(PROJECT_ROOT)}")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = _build_alignment(eta_config, omega_config)
    _materialise(model, state_dict)
    model.eval()

    out_dir = staging_root / "weights" / task / backbone
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PyTorchModelHubMixin.save_pretrained writes both `model.safetensors`
    # (or pytorch_model.bin) and `config.json` derived from __init__ kwargs.
    # It also drops a stub README.md per subfolder, which we delete in favour
    # of the single top-level model card written by `stage_repo_assets`.
    model.save_pretrained(out_dir)
    auto_readme = out_dir / "README.md"
    if auto_readme.exists():
        auto_readme.unlink()

    # Sidecar metadata: hyperparameters, eval metrics, source pointer.
    sidecar = {
        "task": task,
        "backbone": backbone,
        "split": 0,
        "eta_config": eta_config,
        "omega_config": omega_config,
        "score_config": score_config,
        "metrics": metrics,
        "source_checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
    }
    (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2))

    return {
        "task": task,
        "backbone": backbone,
        "out_dir": out_dir,
        "metrics": metrics,
    }


def stage_repo_assets(staging_root: Path) -> None:
    """Copy the model card + visual abstract into the staging root."""
    if MODEL_CARD_SRC.exists():
        shutil.copyfile(MODEL_CARD_SRC, staging_root / "README.md")
    else:
        print(f"[warn] model card {MODEL_CARD_SRC} missing — repo will have no README")

    visual = ASSETS_DIR / "visual_abstract.png"
    if visual.exists():
        target_assets = staging_root / "assets"
        target_assets.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(visual, target_assets / "visual_abstract.png")


def _infer_input_dim(out_dir: Path) -> int:
    """Read the saved safetensors file to derive the LRL input dimension."""
    from safetensors import safe_open

    sf_path = out_dir / "model.safetensors"
    with safe_open(sf_path, framework="pt") as f:
        shape = f.get_slice("eta.lrl.0.weight").get_shape()
    return int(shape[1])


def round_trip_check(staged: List[Dict[str, object]]) -> None:
    """Reload each staged variant via ``Alignment.from_pretrained``."""
    print("\n[verify] reloading staged variants via from_pretrained ...")
    for entry in staged:
        out_dir: Path = entry["out_dir"]  # type: ignore[assignment]
        backbone: str = entry["backbone"]  # type: ignore[assignment]
        dim = _infer_input_dim(out_dir)
        model = Alignment.from_pretrained(str(out_dir))
        model.eval()
        n_q, n_c = 7, 9
        H_q = torch.randn(n_q, dim)
        H_c = torch.randn(n_c, dim)
        batch_q = torch.zeros(n_q, dtype=torch.long)
        batch_c = torch.zeros(n_c, dtype=torch.long)
        with torch.no_grad():
            M = model(H_q, H_c, batch_q, batch_c)
        assert M.shape == (n_q, n_c), f"unexpected output shape {M.shape}"
        print(
            f"  [ok]   {entry['task']}/{backbone}  in_dim={dim}  out={tuple(M.shape)}"
        )


def upload(staging_root: Path, repo_id: str, private: bool, token: Optional[str]) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(staging_root),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload PLASMA split-0 checkpoints",
    )
    print(f"\n[done] uploaded {staging_root} to https://huggingface.co/{repo_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging",
        type=Path,
        default=DEFAULT_STAGING,
        help="Local staging directory (default: .hf_staging at the repo root).",
    )
    parser.add_argument(
        "--repo-id",
        default="zhiyuw/plasma",
        help="Target Hub repo id (default: zhiyuw/plasma).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the target repo as private (useful for the test repo).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage and verify locally without uploading to the Hub.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. Defaults to whatever huggingface_hub finds (env / login).",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=AVAILABLE_TASKS,
        help="Restrict to one or more tasks (repeatable). Defaults to all.",
    )
    parser.add_argument(
        "--backbone",
        action="append",
        choices=AVAILABLE_BACKBONES,
        help="Restrict to one or more backbones (repeatable). Defaults to all.",
    )
    args = parser.parse_args()

    tasks = args.task or list(AVAILABLE_TASKS)
    backbones = args.backbone or list(AVAILABLE_BACKBONES)

    staging_root: Path = args.staging
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    staged: List[Dict[str, object]] = []
    for task in tasks:
        for backbone in backbones:
            entry = stage_variant(task, backbone, staging_root)
            if entry is not None:
                staged.append(entry)

    if not staged:
        print("[error] no variants staged — nothing to upload.")
        return 1

    stage_repo_assets(staging_root)
    round_trip_check(staged)

    print(f"\n[summary] staged {len(staged)} variants in {staging_root}")
    if args.dry_run:
        print("[dry-run] skipping upload.")
        return 0

    upload(staging_root, args.repo_id, args.private, args.token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
