#!/usr/bin/bash

export LD_LIBRARY_PATH=/usr/local/gib/lib64:$LD_LIBRARY_PATH

export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID
export NODE_RANK=$SLURM_NODEID

set -evx
cd $WORK_DIR
torchrun \
  --rdzv-backend=static \
  --rdzv_id=job_${SLURM_JOB_ID} \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --nnodes=$SLURM_NNODES \
  --nproc-per-node=$SLURM_GPUS_PER_NODE \
  --node-rank=${NODE_RANK} \
  ${TORCHRUN_TRAINER_SCRIPT}  --gin_config_file=${CONFIG_PATH}