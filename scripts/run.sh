#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# Default values
NODES=1
GPUS_PER_NODE=8

export WORK_DIR=$HOME/generative-recommenders
export TRAIN_SCRIPT="generative_recommenders/research/trainer/train.py"
export LOG_DIR=$HOME/logs

echo "Submitting job $JOB_NAME to SLURM cluster..."
sbatch --nodes=$NODES \
  --gpus-per-node=$GPUS_PER_NODE \
  --ntasks-per-node=1 \
  --output="$LOG_DIR/sbatch_init.out" \
  --requeue \
  --job-name=$JOB_NAME \
  scripts/sbatch.slurm
