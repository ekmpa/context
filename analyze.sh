#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=benjamin-cc
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


set -euo pipefail
mkdir -p logs

module load python/3.10
source $SCRATCH/ctxt-env/bin/activate

ENV=".env"

DATASET="rumors"
JUDGE="gpt-5-mini"
# JUDGE="Qwen/Qwen2.5-7B-Instruct"
CONDITION="conflict" # none | conflict | conflict_compare | compare | stale | opinion | unverif | ambig
COMPARE_CONDITIONS="raw,structural,third-party,source_attr"
ANCHOR_CONDITION="raw"
THIRD_PARTY_RATINGS_FILE="data/domain_ratings.csv"

python scripts/analyze_data.py \
  --dataset "$DATASET" \
  --condition "$CONDITION" \
  --compare-conditions "$COMPARE_CONDITIONS" \
  --anchor-condition "$ANCHOR_CONDITION" \
  --third-party-ratings-file "$THIRD_PARTY_RATINGS_FILE" \
  --judge "$JUDGE" \
  --output-dir "analysis/${DATASET}_${CONDITION}"