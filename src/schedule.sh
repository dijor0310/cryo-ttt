#!/bin/bash

NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ttt.py method.learning_rate=1e-4 method.max_epochs=20 exp_name=memseg-dyn-f2fd-cts-vpp-003-lr1e-4

NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ttt.py method.learning_rate=1e-5 method.max_epochs=20 exp_name=memseg-dyn-f2fd-cts-vpp-003-lr1e-5

NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ttt.py method.learning_rate=1e-6 method.max_epochs=20 exp_name=memseg-dyn-f2fd-cts-vpp-003-lr1e-6

# NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ttt.py method.learning_rate=1e-4 method.max_epochs=20
