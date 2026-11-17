#!/bin/bash
# Defences on CFTD (ResNet50 target, in-distribution prior, 20 images per identity, four degradations).
# Model-side: NLS, BiDO, DP-SGD (epsilon 1, 8, 50). The attacker retrains its prior on the labels of the deployed
# (defended) model. Every reconstruction is scored with the undefended ResNet50 (--resize tensor).
# Data-side: image immunization at 8/255 and 16/255 against the attacker's solver.
source "$(dirname "$0")/common.sh"
A=(--dataset cftd --distribution in_distribution --attack targeted --target-model resnet50 --per-class 20)
UNDEF=checkpoints/cftd/classifiers/resnet50/final_model.pt
declare -A TRAIN=([nls]="--defense nls" [bido]="--defense bido" [dp_eps1]="--defense dp --epsilon 1"
                  [dp_eps8]="--defense dp --epsilon 8" [dp_eps50]="--defense dp --epsilon 50")
for d in nls bido dp_eps1 dp_eps8 dp_eps50; do
  $PYTHON scripts/train_classifier.py --dataset cftd --model resnet50 ${TRAIN[$d]}
  ck=checkpoints/cftd/classifiers_$d/resnet50/final_model.pt
  $PYTHON scripts/train_prior.py --dataset cftd --distribution in_distribution --attack targeted \
      --target-model resnet50 --classifier-checkpoint "$ck" --tag "$d"
  for deg in "${DEGRADATIONS[@]}"; do
    out="results/defenses/$d/$deg"
    $PYTHON scripts/attack.py "${A[@]}" --degradation "$deg" \
        --prior "checkpoints/cftd/priors/in_distribution/targeted/resnet50_$d" --out "$out"
    $PYTHON scripts/evaluate.py "$out/manifest.csv" --dataset cftd --classifiers resnet50 --checkpoint "$UNDEF" \
        --resize tensor --no-quality
  done
done
for deg in "${DEGRADATIONS[@]}"; do
  out="results/defenses/none/$deg"
  $PYTHON scripts/attack.py "${A[@]}" --degradation "$deg" --out "$out"
  $PYTHON scripts/evaluate.py "$out/manifest.csv" --dataset cftd --classifiers resnet50 --resize tensor --no-quality
  for eps in 8 16; do
    imm="results/defenses/immunization/eps=$eps/$deg"
    $PYTHON scripts/immunize.py "${A[@]}" --degradation "$deg" --eps $eps --out "$imm/delta"
    $PYTHON scripts/attack.py "${A[@]}" --degradation "$deg" --delta "$imm/delta/delta.pt" --out "$imm"
    $PYTHON scripts/evaluate.py "$imm/manifest.csv" --dataset cftd --classifiers resnet50 --resize tensor --no-quality
    $PYTHON scripts/evaluate.py "$imm/manifest.csv" --dataset cftd --classifiers resnet50 --resize tensor --observed
  done
done
