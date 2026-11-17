#!/bin/bash
# Target classifiers of every dataset (ResNet18, VGG16, ViT-B/16; ResNet50 is the defence study's target).
source "$(dirname "$0")/common.sh"
for ds in cftd charlotte thermvision; do
  for m in resnet18 resnet50 vgg16 vit_b_16; do
    $PYTHON scripts/train_classifier.py --dataset "$ds" --model "$m"
  done
done
