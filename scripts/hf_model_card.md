---
license: mit
library_name: plasma-protein-local-alignment
pipeline_tag: feature-extraction
tags:
  - protein
  - protein-language-model
  - alignment
  - optimal-transport
  - sinkhorn
  - bioinformatics
  - biology
---

# PLASMA: Pluggable Local Alignment via Sinkhorn MAtrix

[![arXiv](https://img.shields.io/badge/arXiv-2510.11752-b31b1b.svg)](https://arxiv.org/abs/2510.11752)
[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://arxiv.org/abs/2510.11752)
[![GitHub stars](https://img.shields.io/github/stars/ZW471/PLASMA-Protein-Local-Alignment?style=social)](https://github.com/ZW471/PLASMA-Protein-Local-Alignment)

![Visual abstract](assets/visual_abstract.png)

**PLASMA** is a tiny, pluggable head that turns any frozen protein-language-model
(PLM) into a residue-level *local* aligner. It reformulates protein substructure
alignment as a regularised optimal transport problem and runs ~50× faster than
structure-based aligners (TM-Align, Foldseek) by operating on pre-computed
embeddings.

This repository hosts the trained **PLASMA** heads for every (task, backbone)
combination from the paper, plus instructions for the parameter-free
**PLASMA-PF** baseline (which has no learned weights).

- **Paper:** <https://arxiv.org/abs/2510.11752>
- **Code:** <https://github.com/ZW471/PLASMA-Protein-Local-Alignment>
- **License:** MIT

## What's in this repo

Each variant lives in its own subfolder and is loaded by the `load_plasma`
helper from the GitHub package:

```
weights/
  active_site/
    prot_bert/                 # config.json + model.safetensors + metadata.json
    ankh-base/
    TM-Vec/
    ProstT5/
    prot_t5_xl_half_uniref50-enc/
    esm2_t33_650M_UR50D/
    ProtSSN/
  binding_site/
    ...
  motif/
    ...
```

All heads share the same architecture: a small `LRL` non-linearity
(`LazyLinear → ReLU → Linear → LayerNorm`, hidden dim 512) followed by a
parameter-free Sinkhorn iteration (`temperature=0.1`, `n_iters=20`). The
checkpoint files are ~3 MB each.

## How to use

PLASMA is a *head*: it consumes per-residue embeddings from a frozen protein
language model and returns a soft alignment matrix between two
sub-structures. The end-to-end pipeline is therefore three steps:

1. Embed each protein with the backbone the head was trained on (one of the
   seven listed above).
2. Run the PLASMA head on the (residue × residue) embeddings to get a soft
   alignment matrix `M ∈ [0, 1]^{n_q × n_c}`.
3. Optionally reduce `M` to a scalar similarity score with
   `utils.alignment_score`.

### 1. Install

```bash
git clone https://github.com/ZW471/PLASMA-Protein-Local-Alignment
cd PLASMA-Protein-Local-Alignment
uv sync
```

The `Alignment` class and the `load_plasma` helper live in the `alignment`
package shipped by that repo.

### 2. Load a trained head

```python
from alignment import load_plasma

# task ∈ {"active_site", "binding_site", "motif"}
# backbone is the PLM whose embeddings the head was trained on
model = load_plasma(task="active_site", backbone="esm2_t33_650M_UR50D")
model.eval()
```

`load_plasma` downloads the matching `config.json` + `model.safetensors` from
this repo via `huggingface_hub` and rebuilds the `Alignment` module.

### 3. Compute embeddings with the matching backbone

PLASMA does not embed sequences itself. The example below shows how to do it
with **ESM-2** via `transformers`; the same pattern works for any other
backbone (`Ankh`, `ProstT5`, `ProtBERT`, `ProtT5`, `TM-Vec`, `ProtSSN` —
their loaders are documented in `embed.py` in the GitHub repo).

```python
import torch
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
backbone = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device).eval()

@torch.no_grad()
def embed(sequence: str) -> torch.Tensor:
    """Return per-residue embeddings of shape (L, 1280) — no special tokens."""
    tokens = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    h = backbone(**tokens).last_hidden_state[0]   # (L+2, 1280): <cls> ... <eos>
    return h[1:-1].cpu()                          # drop <cls> and <eos>

seq_q = "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"
seq_c = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNI"

H_q = embed(seq_q)            # (n_q, 1280)
H_c = embed(seq_c)            # (n_c, 1280)
```

### 4. Run PLASMA and read the alignment matrix

```python
# `batch_q` / `batch_c` assign each residue to a sample. Use zeros for a
# single pair; use [0, 0, ..., 1, 1, ...] to score multiple pairs in one batch.
batch_q = torch.zeros(H_q.size(0), dtype=torch.long)
batch_c = torch.zeros(H_c.size(0), dtype=torch.long)

with torch.no_grad():
    M = model(H_q, H_c, batch_q, batch_c)        # (n_q, n_c) in [0, 1]

# Hard residue-residue assignment (top of column / row in the transport plan)
q_to_c = M.argmax(dim=1)        # for each query residue, the best candidate residue
c_to_q = M.argmax(dim=0)        # for each candidate residue, the best query residue
```

`M` is a (near-)doubly-stochastic transport plan: rows and columns each sum
to ~1, so `M[i, j]` is the soft probability that query residue `i` aligns to
candidate residue `j`. Thresholding at `0.5` gives a sparse local alignment;
plotting `M` as a heatmap gives the canonical PLASMA visualisation (the
diagonal stripe in the visual abstract above).

### 5. Reduce to a similarity score

To collapse the alignment matrix into a single number per protein pair (the
quantity used to compute ROC-AUC / F1-Max in the tables above), use
`utils.alignment_score` from the GitHub repo. It applies the diagonal
convolution + thresholding described in the paper:

```python
from utils.alignment_utils import alignment_score

score = alignment_score(
    H_q, H_c, M, batch_c,
    threshold=0.5,           # gating on max-row / max-col residues
    K=10,                    # diagonal-convolution window
)                            # -> shape (num_pairs_in_batch,), here (1,)
print(float(score))
```

## PLASMA-PF (parameter-free)

PLASMA-PF is a hinge / Sinkhorn baseline with **no learned weights**. Use it
when you want a strong zero-training baseline on top of any backbone — there
is nothing to download:

```python
from alignment import load_plasma_pf

model = load_plasma_pf()         # Alignment(eta='hinge', omega='sinkhorn', ...)

with torch.no_grad():
    M_pf = model(H_q, H_c, batch_q, batch_c)
```

It accepts the same forward signature as the trained heads above and pairs
with any of the seven supported backbones.

## Available variants & evaluation results

Numbers below are 3-seed averages reported in the paper. The seven backbone
columns correspond to the seven subfolders under each task. **Bold** marks the
best backbone for each row.

### Interpolation (in-distribution test split)

| Task | Metric | Ankh | ESM-2 | ProstT5 | ProtBERT | ProtSSN | ProtT5 | TM-Vec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Motif** | ROC-AUC | .925 | .933 | .954 | .854 | .922 | **.972** | .910 |
|  | F1-Max | .885 | .877 | .885 | .784 | .866 | **.918** | .853 |
|  | PR-AUC | .921 | .931 | .953 | .872 | .920 | **.971** | .914 |
|  | Label Match Score | .921 | .890 | .929 | .746 | .767 | **.937** | .792 |
| **Binding Site** | ROC-AUC | **.995** | .992 | .993 | .981 | .992 | .993 | .980 |
|  | F1-Max | .987 | .986 | .983 | .948 | .982 | **.988** | .970 |
|  | PR-AUC | **.996** | .994 | .995 | .985 | .993 | .995 | .984 |
|  | Label Match Score | **.951** | .950 | **.951** | .880 | .872 | **.951** | .900 |
| **Active Site** | ROC-AUC | **.994** | .991 | .993 | .986 | .992 | **.994** | .991 |
|  | F1-Max | **.989** | .985 | .987 | .967 | .987 | .987 | .982 |
|  | PR-AUC | **.994** | .992 | **.994** | .988 | **.994** | **.994** | .992 |
|  | Label Match Score | **.975** | .969 | **.975** | .904 | .885 | .972 | .938 |

### Extrapolation (held-out hard test split)

| Task | Metric | Ankh | ESM-2 | ProstT5 | ProtBERT | ProtSSN | ProtT5 | TM-Vec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Motif** | ROC-AUC | .960 | .972 | **.975** | .870 | .949 | .968 | .954 |
|  | F1-Max | .915 | **.931** | .926 | .799 | .896 | .922 | .903 |
|  | PR-AUC | .948 | **.970** | .969 | .873 | .940 | .962 | .944 |
|  | Label Match Score | **.842** | .786 | .801 | .541 | .537 | .738 | .704 |
| **Binding Site** | ROC-AUC | .995 | **.999** | .993 | .951 | **.999** | **.999** | .990 |
|  | F1-Max | .992 | .991 | .985 | .896 | .988 | **.996** | .983 |
|  | PR-AUC | .997 | **.999** | .995 | .958 | .998 | **.999** | .992 |
|  | Label Match Score | .894 | .851 | .891 | .603 | .753 | **.902** | .824 |
| **Active Site** | ROC-AUC | .995 | .996 | .996 | .980 | .997 | **.999** | .995 |
|  | F1-Max | **.992** | .986 | .991 | .950 | .991 | .991 | .985 |
|  | PR-AUC | .995 | .997 | .997 | .984 | .998 | **.999** | .996 |
|  | Label Match Score | **.938** | .882 | .931 | .697 | .737 | .893 | .880 |

Each subfolder also contains a `metadata.json` with the full hyperparameter
config in machine-readable form.

## Training details

- **Architecture:** `Alignment(eta='lrl', omega='sinkhorn',
  eta_kwargs={'hidden_dim': 512},
  omega_kwargs={'temperature': 0.1, 'n_iters': 20})`.
- **Score head:** `K=10`, `threshold=0.5` (used by
  `utils.alignment_score` to reduce the transport plan to a scalar).
- **Optimiser / loss:** Adam (`lr=1e-4`), `BCEWithLogitsLoss` on the alignment
  score plus a label-match auxiliary loss (`target_loss_weight=1.0`).
- **Data:** the InterPro-derived motif / binding-site / active-site datasets
  shipped under `data/raw/` in the GitHub repo, split into train / validation /
  test / test-hard with `dataset_fraction=0.1` (default sweep) and
  `dataset_fraction=1.0` (full sweep — checkpoints here are from the full
  sweep).
- **Selection metric:** validation loss (early stopping, `patience=3`).

## Citation

If you use these weights, please cite the PLASMA paper:

```bibtex
@inproceedings{wang2026plasma,
  title     = {Fast and Interpretable Protein Substructure Alignment via Optimal Transport},
  author    = {Wang, Zhiyu and Zhou, Bingxin and Wang, Jing and Tan, Yang and Zhao, Weishu and Li{\`o}, Pietro and Hong, Liang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2510.11752},
}
```
