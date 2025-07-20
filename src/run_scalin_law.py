import os
import csv
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Final

# === CONFIGURATION ===

ALL_FILENAMES: Final = [
    "spinach_tomo02_patch000_raw_split0_0000.nii.gz",
    "spinach_tomo02_patch000_split0_0000.nii.gz",
    "spinach_tomo02_patch001_raw_split0_0000.nii.gz",
    "spinach_tomo02_patch001_split0_0000.nii.gz",
    "spinach_tomo02_patch002_raw_split0_0000.nii.gz",
    "spinach_tomo02_patch002_split0_0000.nii.gz",
    "spinach_tomo02_patch003_raw_split0_0000.nii.gz",
    "spinach_tomo02_patch003_split0_0000.nii.gz",
    "spinach_tomo02_patch004_raw_split0_0000.nii.gz",
    "spinach_tomo02_patch004_split0_0000.nii.gz",
    "spinach_tomo02_patch005_raw_split0_0000.nii.gz",
    "spinach_tomo02_patch005_split0_0000.nii.gz",
    "spinach_tomo02_patch006_raw_split0_0000.nii.gz",
    "spinach_tomo02_patch006_split0_0000.nii.gz",
    "spinach_tomo02_patch020_raw_split0_0000.nii.gz",
    "spinach_tomo02_patch020_split0_0000.nii.gz",
    "spinach_tomo02_patch021_raw_split0_0000.nii.gz",
    "spinach_tomo02_patch021_split0_0000.nii.gz",
    "spinach_tomo02_patch030_raw_split0_0000.nii.gz",
    "spinach_tomo02_patch030_split0_0000.nii.gz",
    "spinach_tomo03_patch000_raw_split0_0000.nii.gz",
    "spinach_tomo03_patch000_split0_0000.nii.gz",
    "spinach_tomo03_patch002_raw_split0_0000.nii.gz",
    "spinach_tomo03_patch002_split0_0000.nii.gz",
    "spinach_tomo03_patch004_raw_split0_0000.nii.gz",
    "spinach_tomo03_patch004_split0_0000.nii.gz",
    "spinach_tomo03_patch006_raw_split0_0000.nii.gz",
    "spinach_tomo03_patch006_split0_0000.nii.gz",
    "spinach_tomo03_patch008_raw_split0_0000.nii.gz",
    "spinach_tomo03_patch008_split0_0000.nii.gz",
    "spinach_tomo03_patch010_raw_split0_0000.nii.gz",
    "spinach_tomo03_patch010_split0_0000.nii.gz",
    "spinach_tomo03_patch020_raw_split0_0000.nii.gz",
    "spinach_tomo03_patch020_split0_0000.nii.gz",
    "spinach_tomo03_patch021_raw_split0_0000.nii.gz",
    "spinach_tomo03_patch021_split0_0000.nii.gz",
    "spinach_tomo03_patch022_raw_split0_0000.nii.gz",
    "spinach_tomo03_patch022_split0_0000.nii.gz",
    "spinach_tomo03_patch030_raw_split0_0000.nii.gz",
    "spinach_tomo03_patch030_split0_0000.nii.gz",
    "spinach_tomo03_patch031_raw_split0_0000.nii.gz",
    "spinach_tomo03_patch031_split0_0000.nii.gz",
    "spinach_tomo03_patch032_raw_split0_0000.nii.gz",
    "spinach_tomo03_patch032_split0_0000.nii.gz",
    "spinach_tomo04_patch000_split0_0000.nii.gz",
    "spinach_tomo04_patch002_split0_0000.nii.gz",
    "spinach_tomo04_patch004_split0_0000.nii.gz",
    "spinach_tomo04_patch006_split0_0000.nii.gz",
    "spinach_tomo04_patch008_split0_0000.nii.gz",
    "spinach_tomo04_patch009_split0_0000.nii.gz",
    "spinach_tomo04_patch020_split0_0000.nii.gz",
    "spinach_tomo04_patch021_split0_0000.nii.gz",
    "spinach_tomo04_patch030_split0_0000.nii.gz",
    "spinach_tomo10_patch001_split0_0000.nii.gz",
    "spinach_tomo10_patch002_split0_0000.nii.gz",
    "spinach_tomo10_patch003_split0_0000.nii.gz",
    "spinach_tomo10_patch009_split0_0000.nii.gz",
    "spinach_tomo10_patch010_split0_0000.nii.gz",
    "spinach_tomo10_patch011_split0_0000.nii.gz",
    "spinach_tomo10_patch020_split0_0000.nii.gz",
    "spinach_tomo10_patch030_split0_0000.nii.gz",
    "spinach_tomo17_patch000_split0_0000.nii.gz",
    "spinach_tomo17_patch001_split0_0000.nii.gz",
    "spinach_tomo17_patch002_split0_0000.nii.gz",
    "spinach_tomo17_patch006_split0_0000.nii.gz",
    "spinach_tomo17_patch008_split0_0000.nii.gz",
    "spinach_tomo17_patch009_split0_0000.nii.gz",
    "spinach_tomo17_patch010_split0_0000.nii.gz",
    "spinach_tomo17_patch014_split0_0000.nii.gz",
    "spinach_tomo17_patch015_split0_0000.nii.gz",
    "spinach_tomo17_patch030_split0_0000.nii.gz",
    "spinach_tomo32_patch000_split0_0000.nii.gz",
    "spinach_tomo32_patch002_split0_0000.nii.gz",
    "spinach_tomo32_patch004_split0_0000.nii.gz",
    "spinach_tomo32_patch006_split0_0000.nii.gz",
    "spinach_tomo32_patch008_split0_0000.nii.gz",
    "spinach_tomo32_patch010_split0_0000.nii.gz",
    "spinach_tomo32_patch012_split0_0000.nii.gz",
    "spinach_tomo32_patch014_split0_0000.nii.gz",
    "spinach_tomo32_patch020_split0_0000.nii.gz",
    "spinach_tomo32_patch030_split0_0000.nii.gz",
    "spinach_tomo32_patch031_split0_0000.nii.gz",
    "spinach_tomo38_patch000_split0_0000.nii.gz",
    "spinach_tomo38_patch002_split0_0000.nii.gz",
    "spinach_tomo38_patch004_split0_0000.nii.gz",
    "spinach_tomo38_patch006_split0_0000.nii.gz",
    "spinach_tomo38_patch008_split0_0000.nii.gz",
    "spinach_tomo38_patch010_split0_0000.nii.gz",
    "spinach_tomo38_patch020_split0_0000.nii.gz",
    "spinach_tomo38_patch021_split0_0000.nii.gz",
    "spinach_tomo38_patch030_split0_0000.nii.gz",
]

# VALIDATION_FILENAMES: Final = ["tomo9", "tomo10"]
VALIDATION_FILENAMES: Final = [
    "spinach_tomo38_patch000_split0_0000.nii.gz",
    "spinach_tomo38_patch002_split0_0000.nii.gz",
    "spinach_tomo38_patch004_split0_0000.nii.gz",
    "spinach_tomo38_patch006_split0_0000.nii.gz",
    "spinach_tomo38_patch008_split0_0000.nii.gz",
    "spinach_tomo38_patch010_split0_0000.nii.gz",
    "spinach_tomo38_patch020_split0_0000.nii.gz",
    "spinach_tomo38_patch021_split0_0000.nii.gz",
    "spinach_tomo38_patch030_split0_0000.nii.gz",
]
TRAINING_SET_SIZES: Final = range(63, 91, 2)  # Number of training samples for scaling law
REPEATS_PER_SIZE = 1

LOG_CSV_PATH = Path("scaling_law/scaling_law_results.csv")
TRAINING_SCRIPT = Path("training.py")

# === HELPER FUNCTIONS ===

def get_random_subset(full_list, size, exclude=[]):
    candidates = list(set(full_list) - set(exclude))
    return random.sample(candidates, size)

def run_training_job(train_subset, valid_subset, id):
    train_arg = ",".join(train_subset)
    valid_arg = ",".join(valid_subset)

    cmd = [
        "python", str(TRAINING_SCRIPT),
        f"exp_name=memseg-f2fd-scaling-law-spinach-{len(train_subset)}-{id}",
        f"'method.train_tomo_names=[{train_arg}]'",
        f"'method.val_tomo_names=[{valid_arg}]'",
    ]

    # print(' '.join(cmd)) 
    cmd = ' '.join(cmd)
    my_env = os.environ.copy()
    my_env["NCCL_P2P_DISABLE"] = "1"
    my_env["OMP_NUM_THREADS"] = "16"
    my_env["MKL_NUM_THREADS"] = "16"

    print(f"\nLaunching training job with size={len(train_subset)}")
    process = subprocess.Popen(cmd, env=my_env, shell=True)
    return process

def log_job(train_subset, process_id, id):
    with LOG_CSV_PATH.open(mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().isoformat(),
            ",".join(train_subset),
            ",".join(VALIDATION_FILENAMES),
            process_id,
            id,
            len(train_subset),
        ])

def init_csv():
    if not LOG_CSV_PATH.exists():
        with LOG_CSV_PATH.open(mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "train_tomo_names", "valid_tomo_names", "process_id"])

# === MAIN SCRIPT ===

def main():
    init_csv()

    id = 0
    for size in TRAINING_SET_SIZES:
        for repeat in range(REPEATS_PER_SIZE):
            train_subset = get_random_subset(ALL_FILENAMES, size, exclude=VALIDATION_FILENAMES)
            process = run_training_job(train_subset, VALIDATION_FILENAMES, id)
            log_job(train_subset, process.pid, id)
            id += 1
            process.wait()
            time.sleep(1)  # Optional pause between jobs

if __name__ == "__main__":
    main()
