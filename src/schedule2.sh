#!/bin/bash

NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ttt.py method.max_epochs=40 exp_name=memseg-dyn-f2fd-spinach-0.125-def-034 'ckpt_path=/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-dynunet-f2fd-spinach-0.125-bd1f8/epoch\=826-val/dice_loss\=0.18.ckpt'

NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ttt.py method.max_epochs=40 exp_name=memseg-dyn-f2fd-spinach-0.25-def-034 'ckpt_path=/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-dynunet-f2fd-spinach-0.25-3e112/epoch\=978-val/dice_loss\=0.20.ckpt'

NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ttt.py method.max_epochs=40 exp_name=memseg-dyn-f2fd-spinach-0.5-def-034 'ckpt_path=/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-dynunet-f2fd-spinach-0.5-df71c/epoch\=976-val/dice_loss\=0.20.ckpt'

NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ttt.py method.max_epochs=40 exp_name=memseg-dyn-f2fd-spinach-0.75-def-034 'ckpt_path=/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-dynunet-f2fd-spinach-0.75-9714f/epoch\=959-val/dice_loss\=0.20.ckpt'
