#!/bin/bash
# Main results (Tables 2-3 and Figures of the paper): untargeted and targeted ThermalFlow-MI, in-distribution and
# out-of-distribution priors, four degradations. Usage: experiments/02_main_results.sh [dataset]  (default cftd)
source "$(dirname "$0")/common.sh"
DS="${1:-cftd}"
MODELS=(resnet18 vgg16 vit_b_16)
for dist in "${DISTRIBUTIONS[@]}"; do
  $PYTHON scripts/train_prior.py --dataset "$DS" --distribution "$dist" --attack untargeted
  for m in "${MODELS[@]}"; do
    $PYTHON scripts/train_prior.py --dataset "$DS" --distribution "$dist" --attack targeted --target-model "$m"
  done
  for deg in "${DEGRADATIONS[@]}"; do
    out="results/$DS/$dist/untargeted/$deg"
    $PYTHON scripts/attack.py --dataset "$DS" --distribution "$dist" --attack untargeted --degradation "$deg" --out "$out"
    $PYTHON scripts/evaluate.py "$out/manifest.csv" --dataset "$DS" --classifiers "${MODELS[@]}" --per-class 20
    for m in "${MODELS[@]}"; do
      out="results/$DS/$dist/targeted/$m/$deg"
      $PYTHON scripts/attack.py --dataset "$DS" --distribution "$dist" --attack targeted --target-model "$m" \
          --degradation "$deg" --out "$out"
      $PYTHON scripts/evaluate.py "$out/manifest.csv" --dataset "$DS" --classifiers "$m" --per-class 20
    done
  done
done
# "No attack" rows: the degraded observations themselves (any manifest of the same degradation).
for deg in "${DEGRADATIONS[@]}"; do
  $PYTHON scripts/evaluate.py "results/$DS/in_distribution/untargeted/$deg/manifest.csv" --dataset "$DS" \
      --classifiers "${MODELS[@]}" --per-class 20 --observed
done
