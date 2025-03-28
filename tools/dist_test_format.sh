#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
Out_dir=$3
GPUS=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29100}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test_format.py \
    $CONFIG \
    $CHECKPOINT \
    $Out_dir \
    --launcher pytorch \
    ${@:5}
