#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=test-gemini
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs

module load python/3.10 


ROOT_DIR="."
ENV_FILE="${ROOT_DIR}/.env"

DATASET="${DATASET:-fakecovid}"
HF_DATASET="rabuahmad/climatecheck"
HF_CONFIG="default"
HF_SPLIT="test"
HF_LABEL_AGGREGATION="climatecheck-narrative"
PIPELINE="${PIPELINE:-facttool}" # rarr | facttool
CONDITION="${CONDITION:-raw}" # raw | structural | third-party | source_attr
RARR_MODEL="Qwen/Qwen3-1.7B"
# RARR_MODEL="gpt-5-mini"
# RARR_MODEL="gemini-3.1-flash-lite"
# FACTTOOL_MODEL="gpt-5-mini"
FACTTOOL_MODEL="Qwen/Qwen3-1.7B" # gpt-5-mini
MAX_ROWS=""

STRUCTURAL_SHARDS_DIR="$SCRATCH/credibench-neighbors_serving_shards"
STRUCTURAL_HOPS="${STRUCTURAL_HOPS:-2}"
STRUCTURAL_MAX_DOMAINS_PER_HOP="5"
STRUCTURAL_HOOK_MODE="temporal" # temporal | latest
STRUCTURAL_MONTHS_BACK="3"
STRUCTURAL_TEMPORAL_FALLBACK_TO_1HOP="1"
STRUCTURAL_HOOK_TIMEOUT="40"
STRUCTURAL_CACHE_FILE="${ROOT_DIR}/data/structural_neighbors_cache.json"
THIRD_PARTY_RATINGS_FILE="${ROOT_DIR}/data/domain_ratings.csv"

INIT_SEED="${INIT_SEED:-5}"
MIN_SAMPLES_PER_SPLIT="20"
MAX_SAMPLES_PER_LABEL="1000"

source "${SCRATCH}/ctxt-env/bin/activate"

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

export RARR_CONDITION="${CONDITION}"
export RARR_MODEL="${RARR_MODEL}"
export FACTCHECK_PIPELINE="${PIPELINE}"
export FACTTOOL_MODEL="${FACTTOOL_MODEL}"
export RARR_THIRD_PARTY_RATINGS_FILE="${THIRD_PARTY_RATINGS_FILE}"
export RARR_SEED="${INIT_SEED}"
export RARR_NUM_ROUNDS_QGEN="3"
export RARR_MAX_EVIDENCES_PER_QUESTION="5"
export RARR_SAMPLE_WORKERS="${SLURM_CPUS_PER_TASK:-4}"
export RARR_QUERY_WORKERS="2"
export RARR_GATE_WORKERS="2"
export RARR_USE_COMPACT_GATE_PROMPT="1"
export RARR_SEARCH_PROVIDER="auto"
export RARR_SEARCH_TIMEOUT="8"
export RARR_STRUCTURAL_HOPS="${STRUCTURAL_HOPS}"
export RARR_STRUCTURAL_MAX_DOMAINS_PER_HOP="${STRUCTURAL_MAX_DOMAINS_PER_HOP}"
export RARR_STRUCTURAL_HOOK_MODE="${STRUCTURAL_HOOK_MODE}"
export RARR_STRUCTURAL_MONTHS_BACK="${STRUCTURAL_MONTHS_BACK}"
export RARR_STRUCTURAL_TEMPORAL_FALLBACK_TO_1HOP="${STRUCTURAL_TEMPORAL_FALLBACK_TO_1HOP}"
export RARR_STRUCTURAL_HOOK_TIMEOUT="${STRUCTURAL_HOOK_TIMEOUT}"
export RARR_STRUCTURAL_CACHE_FILE="${STRUCTURAL_CACHE_FILE}"

echo "[INFO] DATASET=${DATASET}"
echo "[INFO] PIPELINE=${PIPELINE}"
echo "[INFO] HF selector=${HF_DATASET}/${HF_CONFIG}:${HF_SPLIT}"
echo "[INFO] CONDITION=${RARR_CONDITION}"
echo "[INFO] MODEL=${RARR_MODEL}"
echo "[INFO] FACTTOOL_MODEL=${FACTTOOL_MODEL}"
echo "[INFO] MAX_ROWS=${MAX_ROWS:-all}"

declare -a RUN_ARGS=()
EXTRA_ARGS=()

if [[ -n "${MAX_ROWS}" ]]; then
  EXTRA_ARGS+=(--max-rows "${MAX_ROWS}")
fi
if [[ -n "${RARR_MODEL}" ]]; then
  EXTRA_ARGS+=(--rarr-model "${RARR_MODEL}")
fi
if [[ -n "${FACTTOOL_MODEL}" ]]; then
  EXTRA_ARGS+=(--facttool-model "${FACTTOOL_MODEL}")
fi
EXTRA_ARGS+=(--pipeline "${PIPELINE}")

if [[ "${DATASET,,}" == "climatecheck" ]]; then
  RUN_ARGS=(
    --hf-dataset "${HF_DATASET}"
    --hf-config "${HF_CONFIG}"
    --hf-split "${HF_SPLIT}"
    --hf-label-aggregation "${HF_LABEL_AGGREGATION}"
    --condition "${RARR_CONDITION}"
    "${EXTRA_ARGS[@]}"
  )
else
  TEST_DATASET_PATH="${ROOT_DIR}/data/${DATASET}/test/data.jsonl"
  if [[ ! -f "${TEST_DATASET_PATH}" ]]; then
    echo "[INFO] Missing ${TEST_DATASET_PATH}; bootstrapping split assets"
    python "${ROOT_DIR}/scripts/initialize_run_assets.py" \
      --hf-dataset "ComplexDataLab/Misinfo_Datasets" \
      --hf-config "default" \
      --origin-col "dataset" \
      --label-col "label" \
      --seed "${INIT_SEED}" \
      --data-dir "${ROOT_DIR}/data" \
      --metadata-path "${ROOT_DIR}/data/metadata.json" \
      --structural-shards-dir "${STRUCTURAL_SHARDS_DIR}" \
      --min-samples-per-split "${MIN_SAMPLES_PER_SPLIT}" \
      --max-samples-per-label "${MAX_SAMPLES_PER_LABEL}"
  fi

  RUN_ARGS=(
    "${TEST_DATASET_PATH}"
    --condition "${RARR_CONDITION}"
    "${EXTRA_ARGS[@]}"
  )
fi

echo "[INFO] Running: python ${ROOT_DIR}/scripts/run_rarr_eval.py ${RUN_ARGS[*]}"
python "${ROOT_DIR}/scripts/run_rarr_eval.py" "${RUN_ARGS[@]}"
