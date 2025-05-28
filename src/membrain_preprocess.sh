#!/bin/bash

# RAW_TOMO=/mnt/hdd_pool_zion/userdata/diyor/data/in_situ_reinhardtii/tomograms/01122021_BrnoKrios_arctis_lam1_pos4.mrc
# OUTPUT_TOMO=/mnt/hdd_pool_zion/userdata/diyor/data/in_situ_reinhardtii/tomograms_pixel_10/01122021_BrnoKrios_arctis_lam1_pos4.mrc

# RAW_LABEL=/mnt/hdd_pool_zion/userdata/diyor/data/in_situ_reinhardtii/labels/01122021_BrnoKrios_arctis_lam1_pos4_memb.mrc
# OUTPUT_LABEL=/mnt/hdd_pool_zion/userdata/diyor/data/in_situ_reinhardtii/labels_pixel_10/01122021_BrnoKrios_arctis_lam1_pos4_memb.mrc

RAW_TOMO=/mnt/hdd_pool_zion/userdata/diyor/data/deepict/DEF/tomograms_normalized/TS_041_trimmed.rec
OUTPUT_TOMO=/mnt/hdd_pool_zion/userdata/diyor/data/deepict/DEF/tomograms_normalized_pixel_10/TS_041_trimmed.rec

RAW_LABEL=/mnt/hdd_pool_zion/userdata/diyor/data/deepict/DEF/labels/TS_041_membranes_trimmed.mrc
OUTPUT_LABEL=/mnt/hdd_pool_zion/userdata/diyor/data/deepict/DEF/labels_pixel_10/TS_041_membranes_trimmed.mrc

INPUT_PX_SIZE=13.48
OUTPUT_PX_SIZE=10.0

echo $RAW_TOMO

tomo_preprocessing match_pixel_size \
    --input-tomogram $RAW_TOMO \
    --output-path $OUTPUT_TOMO \
    --pixel-size-in $INPUT_PX_SIZE \
    --pixel-size-out $OUTPUT_PX_SIZE

echo $OUTPUT_TOMO

tomo_preprocessing match_seg_to_tomo \
    --seg-path $RAW_LABEL \
    --orig-tomo-path $OUTPUT_TOMO \
    --output-path $OUTPUT_LABEL