#!/bin/bash

CKPT_PATH=/workspaces/cryo/membrain/ckpts/MemBrain_seg_v10_alpha.ckpt
OUT_FOLDER=/mnt/hdd_pool_zion/userdata/diyor/data/deepict/DEF/membrain_out_no_aug
IN_FOLDER=/mnt/hdd_pool_zion/userdata/diyor/data/deepict/DEF/tomograms

# membrain segment --ckpt-path $CKPT_PATH --out-folder $OUT_FOLDER --tomogram-path "${IN_FOLDER}/TS_0001_trimmed.rec" --no-test-time-augmentation
# membrain segment --ckpt-path $CKPT_PATH --out-folder $OUT_FOLDER --tomogram-path "${IN_FOLDER}/TS_0002_trimmed.rec" --no-test-time-augmentation
# membrain segment --ckpt-path $CKPT_PATH --out-folder $OUT_FOLDER --tomogram-path "${IN_FOLDER}/TS_0010_trimmed.rec" --no-test-time-augmentation
# membrain segment --ckpt-path $CKPT_PATH --out-folder $OUT_FOLDER --tomogram-path "${IN_FOLDER}/TS_0004_trimmed.rec" --no-test-time-augmentation
# membrain segment --ckpt-path $CKPT_PATH --out-folder $OUT_FOLDER --tomogram-path "${IN_FOLDER}/TS_0005_trimmed.rec" --no-test-time-augmentation
# membrain segment --ckpt-path $CKPT_PATH --out-folder $OUT_FOLDER --tomogram-path "${IN_FOLDER}/TS_0006_trimmed.rec" --no-test-time-augmentation
# membrain segment --ckpt-path $CKPT_PATH --out-folder $OUT_FOLDER --tomogram-path "${IN_FOLDER}/TS_0008_trimmed.rec" --no-test-time-augmentation
# membrain segment --ckpt-path $CKPT_PATH --out-folder $OUT_FOLDER --tomogram-path "${IN_FOLDER}/TS_0009_trimmed.rec" --no-test-time-augmentation
# membrain segment --ckpt-path $CKPT_PATH --out-folder $OUT_FOLDER --tomogram-path "${IN_FOLDER}/TS_0003_trimmed.rec" --no-test-time-augmentation
membrain segment --ckpt-path $CKPT_PATH --out-folder $OUT_FOLDER --tomogram-path "${IN_FOLDER}/TS_026_trimmed.rec" --no-test-time-augmentation
membrain segment --ckpt-path $CKPT_PATH --out-folder $OUT_FOLDER --tomogram-path "${IN_FOLDER}/TS_030_trimmed.rec" --no-test-time-augmentation
membrain segment --ckpt-path $CKPT_PATH --out-folder $OUT_FOLDER --tomogram-path "${IN_FOLDER}/TS_034_trimmed.rec" --no-test-time-augmentation
membrain segment --ckpt-path $CKPT_PATH --out-folder $OUT_FOLDER --tomogram-path "${IN_FOLDER}/TS_037_trimmed.rec" --no-test-time-augmentation
membrain segment --ckpt-path $CKPT_PATH --out-folder $OUT_FOLDER --tomogram-path "${IN_FOLDER}/TS_041_trimmed.rec" --no-test-time-augmentation
