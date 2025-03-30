#!/bin/bash

OUT_FILE=$1

NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python training.py accumulate_grad_batches=2 &> $OUT_FILE &

NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 nohup python training.py accumulate_grad_batches=3 &> $OUT_FILE &
