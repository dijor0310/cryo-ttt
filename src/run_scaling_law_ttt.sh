#!/bin/bash

# # Define your list of strings
checkpoints=(
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-53-1-a8298/epoch=1184-val/dice_loss=0.23.ckpt"
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-55-2-567e1/epoch=1239-val/dice_loss=0.25.ckpt"
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-57-3-035e1/epoch=1144-val/dice_loss=0.23.ckpt"
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-59-4-9ffbd/epoch=1084-val/dice_loss=0.23.ckpt"
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-61-0-ed7b3/epoch=1184-val/dice_loss=0.24.ckpt"
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-63-0-deff4/epoch=904-val/dice_loss=0.23.ckpt"
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-65-1-77c85/epoch=969-val/dice_loss=0.24.ckpt"
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-67-2-598d6/epoch=969-val/dice_loss=0.24.ckpt"
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-69-3-f148b/epoch=904-val/dice_loss=0.24.ckpt"
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-71-4-d5262/epoch=904-val/dice_loss=0.23.ckpt"
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-73-5-295b1/epoch=969-val/dice_loss=0.24.ckpt"
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-75-6-34b55/epoch=944-val/dice_loss=0.24.ckpt"
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-77-7-f40c9/epoch=969-val/dice_loss=0.23.ckpt"
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-79-8-86614/epoch=819-val/dice_loss=0.24.ckpt"
"/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-f2fd-scaling-law-spinach-81-9-e87f7/epoch=789-val/dice_loss=0.24.ckpt"
)

# Loop through the list
for item in "${checkpoints[@]}"; do
    # Run your command with each string
    # echo "Processing: $item"
    # name= "$(basename "$(dirname "$(dirname $item)")")"
    name="$(dirname "$(dirname $item)")"
    name="$(basename $name)"
    # echo "fdsfs $name"
    # NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 nohup python ttt_subset.py "ckpt_path=${item}" "exp_name=${name}-ttt" &> "nohup/${name}-ttt.out" &
    NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python ttt_subset.py ckpt_path=\"${item}\" exp_name=\"${name}-ttt\"
done
