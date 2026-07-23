#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=launcher
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

set -euo pipefail

DATASET="fakecovid"
CONDITION="structural" # raw | structural | third-party | source_attr
STRUCTURAL_HOPS="2"

SEEDS=(1 2 3 4 5)

for seed in "${SEEDS[@]}"; do
  job_name="${DATASET}-${CONDITION}-h${STRUCTURAL_HOPS}-seed${seed}"
  echo "[INFO] Submitting seed ${seed} as ${job_name}"
  sbatch -J "${job_name}" \
    --export=ALL,INIT_SEED="${seed}",DATASET="${DATASET}",CONDITION="${CONDITION}",STRUCTURAL_HOPS="${STRUCTURAL_HOPS}" \
    run.sh
done
