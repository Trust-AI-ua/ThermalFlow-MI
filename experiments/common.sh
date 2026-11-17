# Shared settings of the experiment scripts. Run them from the repository root.
set -euo pipefail
PYTHON="${PYTHON:-python}"
DEGRADATIONS=(gaussian_blur motion_blur box_inpainting random_inpainting)
DISTRIBUTIONS=(in_distribution out_of_distribution)
