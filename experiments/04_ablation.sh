#!/bin/bash
# Component ablation on CFTD (ResNet50 target, 20 images per identity, Gaussian and motion blur):
#  label source of the prior, class conditioning, number of denoiser draws M.
source "$(dirname "$0")/common.sh"
$PYTHON scripts/train_prior.py --dataset cftd --distribution in_distribution --attack targeted --target-model resnet50
$PYTHON scripts/train_prior.py --dataset cftd --distribution in_distribution --attack targeted --target-model resnet50 \
    --label-source ground_truth --tag gt_labels
A=(--dataset cftd --attack targeted --target-model resnet50 --per-class 20)
for dist in "${DISTRIBUTIONS[@]}"; do
  [[ $dist == out_of_distribution ]] && $PYTHON scripts/train_prior.py --dataset cftd --distribution "$dist" \
      --attack targeted --target-model resnet50
  for deg in gaussian_blur motion_blur; do
    base="results/ablation/$dist/$deg"
    $PYTHON scripts/attack.py "${A[@]}" --distribution "$dist" --degradation "$deg" --out "$base/reference"
    $PYTHON scripts/attack.py "${A[@]}" --distribution "$dist" --degradation "$deg" --unconditional-prior --out "$base/unconditional"
    for M in 1 3 10; do
      $PYTHON scripts/attack.py "${A[@]}" --distribution "$dist" --degradation "$deg" --num-samples $M --out "$base/M=$M"
    done
    [[ $dist == in_distribution ]] && $PYTHON scripts/attack.py "${A[@]}" --distribution "$dist" --degradation "$deg" \
        --prior checkpoints/cftd/priors/in_distribution/targeted/resnet50_gt_labels --out "$base/gt_labels"
    for run in "$base"/*/; do
      $PYTHON scripts/evaluate.py "$run/manifest.csv" --dataset cftd --classifiers resnet50 --kid
    done
  done
done
