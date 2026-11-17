# ThermalFlow-MI

Code for **ThermalFlow-MI**, a model inversion attack on thermal face recognition. Given a leaked, degraded copy
of a private thermal face (blurred or partly masked), the attack rebuilds the face well enough that the target
classifier recognises the person again. It needs only the top-1 labels of the deployed classifier: no gradients,
no confidence scores.

The attack has two stages:

1. **Prior.** The attacker labels public auxiliary thermal faces with the target classifier and trains a
   flow matching prior on them. The prior is class-conditional for the targeted attack and unconditional for the
   untargeted one.
2. **Reconstruction.** PnP-Flow alternates a gradient step on the data term with a denoising step by the prior,
   starting from the degraded copy. Only forward passes of the prior are needed.

```
auxiliary images ──► target classifier (top-1 labels) ──► flow matching prior v_θ(x, t, c)
                                                                    │
leaked copy y = Hx + n ──► PnP-Flow: z = x − γ∇ℓ(y; x),  x = D_t(t z + (1−t) ε) ──► reconstruction x̂
```

## Repository layout

```
thermalflow_mi/                 the Python package
├── attack/                     ThermalFlow-MI
│   ├── unet.py                 UNet velocity field and its class-conditional version
│   ├── labeling.py             label the auxiliary data with the target classifier (label-only access)
│   ├── flow_matching.py        train the prior: conditional flow matching with minibatch OT coupling
│   ├── pnp_flow.py             PnP-Flow reconstruction and the observation model y = Hx + σn
│   └── run.py                  attack a set of images; save images and a manifest
├── degradations/               one operator per file (H and its adjoint)
│   ├── gaussian_blur.py        σ = 1, 61 × 61 kernel, noise 0.05
│   ├── motion_blur.py          15 px, 0°, noise 0.05
│   ├── box_inpainting.py       40 × 40 centre box, noise 0.05
│   └── random_inpainting.py    70% of pixels missing, noise 0.01
├── defenses/                   one defence per file
│   ├── nls.py                  negative label smoothing
│   ├── bido.py                 BiDO-HSIC
│   ├── dp_sgd.py               differentially private training
│   └── immunization.py         perturbation of the released image against the attacker's solver
├── metrics/                    one metric per file
│   ├── accuracy.py             attack accuracy (top-1 of a classifier on the reconstructions)
│   ├── psnr.py  ssim.py  lpips.py
│   ├── fid.py  kid.py          distribution metrics on InceptionV3 features
│   └── inception.py            the FID InceptionV3
├── target_models/              target classifiers: architectures and training
├── data/                       datasets, the 70/10/20 split, evaluation subsets
├── utils/                      seeding, image I/O, manifests
└── config.py                   dataset configs and default paths
scripts/                        command-line entry points
├── train_classifier.py         target classifiers, with or without a model-side defence
├── train_prior.py              stage 1 of the attack
├── attack.py                   stage 2 of the attack
├── immunize.py                 craft immunization perturbations
├── evaluate.py                 accuracy, PSNR, SSIM, LPIPS, FID, KID of a run
└── make_splits.py              regenerate a split file
experiments/                    one shell script per part of the paper
configs/datasets/               cftd.yaml, charlotte.yaml, thermvision.yaml
splits/                         the train / test / auxiliary split of every dataset used in the paper
```

## Installation

```bash
conda create -n thermalflow python=3.10 -y
conda activate thermalflow
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended. The experiments ran with PyTorch 2.13 and torchvision 0.28 on one GPU.
LPIPS, the ImageNet classifier weights and the FID Inception weights are downloaded on first use.

## Data

The datasets are public but are not redistributed here. Download them from their authors and place them under
`data/` (or edit `data_root` in `configs/datasets/*.yaml`):

| Dataset | Used as | Folder | Expected layout |
|---|---|---|---|
| Comprehensive Facial Thermal Dataset (CFTD) | private data; OOD prior data for the other two | `data/CFTD` | `<identity 1..10>/<palette>/*.bmp` |
| Charlotte-ThermalFace | OOD prior data for CFTD; private data in its own experiments | `data/Charlotte-ThermalFace` | `S<k>/*.jpg` |
| ThermVision-DB | private data | `data/ThermVision-DB` | `<Female\|Male> Subjects/Set 2 Cropped Facial Expressions/Subject-<k>/…/*.png` |

`splits/<dataset>.csv` lists every image with its identity and split, with paths relative to the dataset folder:

- `train` (70%): the private data. It trains the target classifiers, and the attack reconstructs these images.
- `test` (10%): held out, used for the classifiers' test accuracy and to choose the PnP step sizes.
- `auxiliary` (20%): the attacker's public data for the in-distribution prior.

The out-of-distribution prior is trained on another dataset whose people do not appear in the private data
(Charlotte-ThermalFace for CFTD; CFTD for the other two).

## Quick start

All commands run from the repository root. Defaults are the paper's settings.

```bash
# 1. target classifier
python scripts/train_classifier.py --dataset cftd --model resnet18

# 2. stage 1: priors (200 epochs each)
python scripts/train_prior.py --dataset cftd --distribution in_distribution --attack untargeted
python scripts/train_prior.py --dataset cftd --distribution in_distribution --attack targeted --target-model resnet18

# 3. stage 2: reconstruct the private images from their blurred copies
python scripts/attack.py --dataset cftd --distribution in_distribution --attack targeted \
    --target-model resnet18 --degradation gaussian_blur

# 4. score the run on the paper's 200 images (first 20 per identity)
python scripts/evaluate.py results/cftd/in_distribution/targeted/resnet18/gaussian_blur/manifest.csv \
    --dataset cftd --classifiers resnet18 --per-class 20
```

An attack run writes `original/`, `observed/` and `reconstructed/` PNGs, `manifest.csv` with one row per image, and
`config.json`. `evaluate.py` writes `metrics.json` next to the manifest.

## Reproducing the paper

| Part of the paper | Script |
|---|---|
| Target classifiers | `experiments/01_train_classifiers.sh` |
| Main results: untargeted and targeted, in- and out-of-distribution, four degradations | `experiments/02_main_results.sh [cftd\|charlotte\|thermvision]` |
| Sensitivity: step size, mis-specified blur kernel | `experiments/03_sensitivity.sh` |
| Ablation: label source, class conditioning, number of denoiser draws M | `experiments/04_ablation.sh` |
| Defences: NLS, BiDO, DP-SGD, image immunization | `experiments/05_defenses.sh` |

### Settings

| | Value |
|---|---|
| Image size | 1 × 128 × 128, values in [−1, 1] |
| Prior | UNet (32 channels, multipliers 1/2/4/8, 6 residual blocks, attention at 16 and 8); Adam, lr 1e-4, batch 16, 200 epochs |
| PnP-Flow | N = 200 steps, M = 5 denoiser draws, γ_n = η σ² (1 − t_n)^α |
| η, α on CFTD | Gaussian 1.0, 0.1; motion 1.5, 0.01; box 0.5, 0.1; random 2.0, 0.01 |
| η, α on Charlotte, ThermVision | Gaussian and motion 1.5, 0.01; box 0.5, 0.1; random 2.0, 0.01 |
| Classifiers | ResNet18, ResNet50, VGG16, ViT-B/16 (ImageNet initialisation, one-channel input, 224 px); AdamW, lr 1e-4, 40 epochs |
| Evaluation subset | main tables: first 20 images per identity; ablation, sensitivity, defences: 20 per identity drawn with seed 0 (`--per-class 20`) |
| NLS | smoothing factor 0 for the first 20 epochs, then linearly to −0.05 |
| BiDO | λ_x = 5, λ_y = 50, HSIC on the four ResNet stages |
| DP-SGD | ε ∈ {1, 8, 50}, δ = 1e-5, clip 1, expected batch 256, lr 1e-2, head trained on the frozen ImageNet backbone |
| Immunization | ‖δ‖∞ ≤ 8/255 and 16/255, 60 PGD steps of size ε/8, surrogate solver with K = 20 steps |

### Reproducibility

- The observation noise is seeded per batch of 4 images, and the PnP-Flow draws continue from that seed. The same
  prior and images therefore always give the same reconstruction.
- This code was checked against the original research code with the paper's trained priors and classifiers. On
  the same images, it gave pixel-identical observations and reconstructions under all four degradations, for both
  the targeted and the untargeted attack.
- Scoring a paper run with `scripts/evaluate.py` reproduces its table entry. For example, targeted ThermalFlow-MI
  on VGG16 under Gaussian blur gives 97.5% accuracy, PSNR 34.12, SSIM 0.954, LPIPS 0.026 and FID 60.0.
- Training the priors and classifiers follows the original procedure and settings. Retraining can still give
  slightly different weights, because GPU kernels and data-loading order are not bit-for-bit deterministic.
- The split files in `splits/` are the ones used in the paper. `scripts/make_splits.py` regenerates them from the
  raw datasets. With the file listing of the original machine, it reproduced all three files exactly.

## Threat model

The attacker:

- has one degraded copy of a private training image, such as a blurred or partly occluded frame;
- can query the deployed classifier for top-1 labels only;
- has public thermal faces of the same kind (in-distribution) or of other people (out-of-distribution).

The attack never uses the private images themselves. The private images and the clean test images are used only
to measure how well the attack works.

## Citation

If you use this code, please cite the paper (BibTeX to be added on publication).

## Acknowledgements

- The PnP-Flow solver follows Martin et al., "PnP-Flow: Plug-and-play image restoration with flow matching",
  ICLR 2025.
- The flow matching training follows Tong et al., "Improving and generalizing flow-based generative models with
  minibatch optimal transport", TMLR 2024.
- The UNet comes from Huang et al., NeurIPS 2021 (sdeflow-light), and the FID network from pytorch-fid.
