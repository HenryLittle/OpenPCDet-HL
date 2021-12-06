#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

# python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}


# Laucher is a custom parameter to select method to init work group
# common_utils init_dist_pytorch
torchrun --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}
