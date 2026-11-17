#!/bin/bash
# Sensitivity analysis on CFTD.
#  (a) Step size: eta x alpha grid of gamma_n = eta sigma^2 (1 - t_n)^alpha, untargeted attack on the held-out
#      test split (the step sizes were chosen there, never on the private training images).
#  (b) Mis-specified blur kernel: the attacker's kernel differs from the true one (sigma 1.0, length 15, 0 deg);
#      targeted ViT-B/16, 20 images per identity. residual_l2 in each manifest is ||H_k x_hat - y|| for the blind
#      attack (kernel selection by data residual or by the discrepancy principle).
source "$(dirname "$0")/common.sh"
MODELS=(resnet18 vgg16 vit_b_16)
for dist in "${DISTRIBUTIONS[@]}"; do
  for deg in "${DEGRADATIONS[@]}"; do
    if [[ $deg == random_inpainting ]]; then ETAS=(0.5 1.0 2.0 3.0); else ETAS=(0.5 1.0 1.5); fi
    for eta in "${ETAS[@]}"; do
      for alpha in 0.01 0.1 0.5 1.0; do
        out="results/sensitivity/step_size/$dist/$deg/eta=${eta}_alpha=$alpha"
        $PYTHON scripts/attack.py --dataset cftd --distribution "$dist" --attack untargeted --degradation "$deg" \
            --split test --lr-pnp "$eta" --alpha "$alpha" --out "$out"
        $PYTHON scripts/evaluate.py "$out/manifest.csv" --dataset cftd --classifiers "${MODELS[@]}"
      done
    done
  done

  A=(--dataset cftd --distribution "$dist" --attack targeted --target-model vit_b_16 --per-class 20)
  for s in 0.5 0.75 1.0 1.25 1.5 2.0; do
    out="results/sensitivity/kernel/$dist/gaussian_blur/sigma=$s"
    $PYTHON scripts/attack.py "${A[@]}" --degradation gaussian_blur --attack-sigma-blur "$s" --out "$out"
    $PYTHON scripts/evaluate.py "$out/manifest.csv" --dataset cftd --classifiers vit_b_16 resnet50 --kid
  done
  for l in 9 11 13 15 17 19 21; do
    out="results/sensitivity/kernel/$dist/motion_blur/length=${l}_angle=0"
    $PYTHON scripts/attack.py "${A[@]}" --degradation motion_blur --attack-motion-length "$l" --out "$out"
    $PYTHON scripts/evaluate.py "$out/manifest.csv" --dataset cftd --classifiers vit_b_16 resnet50 --kid
  done
  for ang in -15 -5 5 15; do
    out="results/sensitivity/kernel/$dist/motion_blur/length=15_angle=$ang"
    $PYTHON scripts/attack.py "${A[@]}" --degradation motion_blur --attack-motion-angle "$ang" --out "$out"
    $PYTHON scripts/evaluate.py "$out/manifest.csv" --dataset cftd --classifiers vit_b_16 resnet50 --kid
  done
done
