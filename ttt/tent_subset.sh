#!/bin/bash

OUT_FILE=$1

NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 nohup python ttt_subset.py --config-name tent &> $OUT_FILE &