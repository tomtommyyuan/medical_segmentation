# CHROMA-Net — nuclei instance segmentation on PanNuke

Pan-cancer nuclei **instance** segmentation and classification on
[PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke), evaluated on
the official three-fold protocol and reported in mPQ / bPQ so the numbers sit
directly against the published table.

CHROMA-Net pairs a frozen pathology foundation-model encoder, adapted with
LoRA, with a HoVer-style multi-task decoder, and adds two things aimed at the
metric and at the dataset's known weak spot:

- **Stain-consistency self-distillation** — an EMA teacher–student objective
  over Macenko-resampled optical-density views, targeting cross-tissue
  robustness.
- **Tissue-conditioned FiLM + long-tail logit adjustment** — targeting mPQ,
  which averages over five classes and so is dragged down by the ~1% "Dead"
  class that most methods score near zero on.

---

## Status

The pipeline is implemented, unit-tested and smoke-tested end to end. **The
results tables below are empty on purpose** — they get filled by running the
commands in [Reproducing](#reproducing) on the real dataset. Nothing in this
README reports a number that has not been measured.

Published baselines are quoted from
[LKCell](https://arxiv.org/html/2407.18054v1) (Table 2), which uses the same
protocol.

| Method | mPQ | bPQ |
|---|---|---|
| HoVer-Net (2019) | 0.4629 | 0.6596 |
| StarDist | 0.4796 | 0.6692 |
| CPP-Net | 0.4815 | 0.6767 |
| CellViT-256 | 0.4846 | 0.6696 |
| CellViT-SAM-H | 0.4980 | 0.6793 |
| LKCell-L (2024) | **0.5080** | **0.6851** |
| | | |
| Classical (threshold + watershed) | — | — |
| CNN baseline | — | — |
| U-Net | — | — |
| Attention U-Net | — | — |
| CHROMA-Net | — | — |
| CHROMA-Net + TTA | — | — |

Target: **mPQ ≥ 0.505, bPQ ≥ 0.685.**

The four baselines have no instance or class output, so they are scored by
taking connected components of their binary masks. They merge every touching
nucleus into one, which is the gap the distance maps exist to close, and it is
visible directly in `figures/qualitative_instances.png`.

---

## Protocol

PanNuke ships three folds. The benchmark trains on one, validates on a second
and tests on the third, rotating over all three and averaging:

| Split | Train | Val | Test |
|---|---|---|---|
| 1 | fold 1 | fold 2 | fold 3 |
| 2 | fold 2 | fold 3 | fold 1 |
| 3 | fold 3 | fold 1 | fold 2 |

**The folds must not be pooled and re-split at random.** Patches within a fold
can come from the same tissue section, so a pooled split leaks between train
and test and produces numbers comparable to nothing. An earlier version of this
repo pooled all three folds and split 70/15/15 at random; that is what the
current `preprocess.py` replaces.

---

## Method

```
H&E patch ──┬─→ ViT-L/16 encoder (Phikon-v2 or UNI, LoRA-adapted)
            │        └─ tokens from 4 blocks, all stride 16
            │
            └─→ conv stem, full resolution ──┐
                                             ▼
                          shared decoder, 4 upsampling stages
                          (FiLM-modulated by tissue embedding)
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    ▼                        ▼                        ▼
              nuclei (NP)            distance maps (HV)          class (NT)
                    │                        │                        │
                    └──────→ marker-controlled watershed ←────────────┘
                                             │
                                    labelled instances + class
```

**Why the conv stem.** A ViT's tokens are stride 16, so a 256×256 patch becomes
a 16×16 grid while a nucleus is about 20px across — the boundary between two
touching nuclei is well below one token. The full-resolution stem feeds the
last decoder stage the detail the transformer never represents. Without it,
boundaries are lost and PQ collapses.

**Why distance maps.** Each nucleus's pixels store their signed offset from its
centre, rescaled to [-1, 1] per side. Two touching nuclei have maps running in
opposite directions across the join, so the gradient spikes exactly at the
boundary — which is what the watershed turns into a split.

**Why logit adjustment.** mPQ averages PQ over the five classes, so Dead nuclei
(~1% of instances) count as much as Neoplastic. Adding `τ·log(prior)` to the
logits during training shifts the boundary toward rare classes with no
architectural change; inference uses the plain logits.

---

## Reproducing

```bash
pip install -r requirements.txt

python src/download_data.py            # PanNuke folds 1-3 from Warwick
python src/preprocess.py               # -> fold{1,2,3}_{images,insts,types,masks,tissues}.npy

# CHROMA-Net, one job per split
for s in 1 2 3; do python src/train_chroma.py --split $s --encoder phikon; done

# Baseline ladder, same folds and same recipe
for s in 1 2 3; do
  for m in cnn unet attention_unet; do python src/train.py --model $m --split $s; done
done

# Numbers
python src/evaluate_instance.py                     # mPQ / bPQ, all three splits
python src/evaluate_instance.py --tta               # + dihedral TTA
python src/evaluate_instance.py --model unet        # a baseline, same metric
python src/evaluate.py                              # pixel-level Dice / IoU

# Figures
python src/evaluate_by_tissue.py
python src/visualize_qualitative.py
python src/plot_training.py
```

On Slurm:

```bash
# ON A LOGIN NODE, once. Compute nodes have no outbound internet, so the first
# from_pretrained() inside a job stalls until it times out rather than failing.
bash scripts/prefetch_encoder.sh

sbatch scripts/train_chroma.sbatch        # job array over the three splits
sbatch scripts/train_baselines.sbatch
bash   scripts/run_ablations.sh
```

The job scripts set `PYTHONNOUSERSITE=1` so a stray package in `~/.local`
cannot shadow the environment, `HF_HOME` on `$SCRATCH` because `$HOME` quotas
fail a 1.2 GB download, and `HF_HUB_OFFLINE=1` so a cold cache errors
immediately instead of hanging.

Before queueing anything, smoke-test the whole pipeline in about a minute:

```bash
python src/train_chroma.py --split 1 --encoder conv --epochs 1 --limit-batches 4
pytest tests/ -q
```

### Encoders

| `--encoder` | Model | Access |
|---|---|---|
| `phikon` (default) | `owkin/phikon-v2`, ViT-L/16 | ungated |
| `uni` | `MahmoodLab/UNI`, ViT-L/16 | gated, free for academic use |
| `conv` | learned from PanNuke only | none — offline fallback and ablation row |

---

## Ablations

Each row is one flag, so the contribution of every component is measurable:

| Run | Command |
|---|---|
| Full | `--encoder phikon` |
| No foundation encoder | `--encoder conv` |
| No stain consistency | `--consistency none` |
| Self-distillation, no EMA teacher | `--consistency self` |
| No tissue conditioning | `--no-tissue-film` |
| No logit adjustment | `--tau 0` |
| Frozen encoder, no LoRA | `--lora-rank 0` |
| No augmentation | `--no-augment` |

`scripts/run_ablations.sh` runs the set and tags each so
`evaluate_instance.py --tag <name>` scores it.

---

## Layout

```
src/
  download_data.py        PanNuke download
  preprocess.py           raw folds -> instance, type, binary maps
  dataset.py              official SPLITS, datasets, augmentation
  stain_augment.py        Macenko optical-density stain jitter
  hover_targets.py        distance-map targets
  postprocess.py          watershed decoding, per-instance class vote
  pq_metrics.py           official mPQ / bPQ
  tta.py                  dihedral TTA with distance-map sign correction
  chroma_net.py           encoder, decoder, FiLM, LoRA
  losses.py               multi-task loss, logit adjustment, consistency
  train_chroma.py         CHROMA-Net training
  train.py                binary baseline training
  evaluate_instance.py    mPQ / bPQ over the three splits
  evaluate.py             pixel-level Dice / IoU
  evaluate_by_tissue.py   per-tissue bPQ + figure
  visualize_qualitative.py per-instance comparison figure
  plot_training.py        curves and benchmark figure
  unet.py, attention_unet.py, cnn_baseline.py, classical.py
tests/                    102 tests, no dataset required
scripts/                  Slurm job scripts, ablation runner
```

---

## Correctness notes

Instance segmentation has several places where a mistake produces plausible
numbers rather than a crash, so those are pinned by tests (`pytest tests/ -q`,
runs in ~5s on CPU with no dataset).

- **PQ aggregation.** A class absent from both prediction and ground truth
  contributes NaN, not zero — scoring it zero costs several mPQ points, since
  most patches hold only two or three of the five classes. Scores average per
  image, then within a tissue, then over the 19 tissues, so Breast cannot
  dominate.
- **TTA sign correction.** Mirroring a patch negates the horizontal distance
  map; rotating swaps both maps and negates one. Getting this wrong makes the
  maps cancel when averaged, flattens the watershed ridges and *lowers* PQ
  while appearing to work. A test averages eight wrongly-inverted views and
  asserts the result is measurably worse.
- **Target equivariance.** HoVer-Net rounds each instance centroid to the
  nearest pixel, which makes the targets only approximately equivariant to
  reflection — mirroring a half-integer centroid rounds the other way. Keeping
  the centroid as a float makes all eight dihedral transforms exact, so TTA
  averages views that agree.
- **Empty-patch Dice.** The previous `evaluate.py` averaged a per-patch Dice
  with a `1e-5` smoothing term, giving every patch with no nuclei a free 1.0.
  Scores now accumulate over a fold and divide once.
- **Instance mask lookup.** PQ indexes masks by instance id, not list
  position, so a prediction covering every pixel (no background id) does not
  crash validation hours into a run.
- **Per-worker RNG.** DataLoader workers are forked copies; a generator built
  in `__init__` hands every worker the same stream, so each batch repeats one
  augmentation across its workers.

---

## References

- Gamper et al., *PanNuke Dataset Extension, Insights and Baselines*, 2020 — [arXiv:2003.10778](https://arxiv.org/pdf/2003.10778)
- Graham et al., *HoVer-Net*, Medical Image Analysis 2019 — [arXiv:1812.06499](https://arxiv.org/abs/1812.06499)
- Hörst et al., *CellViT*, Medical Image Analysis 2024 — [arXiv:2306.15350](https://arxiv.org/pdf/2306.15350)
- Cui et al., *LKCell*, 2024 — [arXiv:2407.18054](https://arxiv.org/html/2407.18054v1)
- Zhu et al., *CellVTA*, 2025 — [arXiv:2504.00784](https://arxiv.org/pdf/2504.00784)
- Chen et al., *UNI*, Nature Medicine 2024 — [github.com/mahmoodlab/UNI](https://github.com/mahmoodlab/UNI)
- Filiot et al., *Phikon-v2*, 2024 — [huggingface.co/owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2)
- Macenko et al., *A method for normalizing histology slides*, ISBI 2009
- Tellez et al., *Quantifying the effects of data augmentation and stain color normalization*, MIA 2019
- Menon et al., *Long-tail learning via logit adjustment*, ICLR 2021
- Tarvainen & Valpola, *Mean teachers*, NeurIPS 2017
- Hu et al., *LoRA*, ICLR 2022
