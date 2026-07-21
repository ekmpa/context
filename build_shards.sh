#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=build-shards
#SBATCH --output=logs/build-shards-%j.out
#SBATCH --error=logs/build-shards-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
mkdir -p logs

module load python/3.10
source "$SCRATCH/ctxt-env/bin/activate"

SHARDS_DIR="$SCRATCH/credibench-neighbors_serving_shards"
TARGET_MONTH="${1:-all}"

month_to_folder() {
    case "${1}" in
        oct2024) echo "oct2024" ;;
        nov2024) echo "nov2024" ;;
        dec2024) echo "dec2024" ;;
        jan2025) echo "jan2025" ;;
        feb2025) echo "feb2025" ;;
        mar2025) echo "march2025" ;;
        apr2025) echo "april2025" ;;
        *) return 1 ;;
    esac
}

if [[ "$TARGET_MONTH" == "all" ]]; then
    echo "Rebuilding serving shards with all months -> $SHARDS_DIR"
    python scripts/hook.py \
        --credibench \
        --build-serving-shards "$SHARDS_DIR" \
        --all-months \
        --force-rebuild-serving-shards \
        --shard-workers 8 \
        -v
else
    MONTH_FOLDER="$(month_to_folder "$TARGET_MONTH")" || {
        echo "Unsupported month '$TARGET_MONTH'. Use one of: oct2024 nov2024 dec2024 jan2025 feb2025 mar2025 apr2025 or all" >&2
        exit 2
    }
    MONTH_URL="https://huggingface.co/datasets/credi-net/CrediBench/resolve/main/${MONTH_FOLDER}/edges.csv.gz"
    MONTH_SHARDS_DIR="${SHARDS_DIR}_${TARGET_MONTH}"
    echo "Building month $TARGET_MONTH only -> $MONTH_SHARDS_DIR"
    python scripts/hook.py \
        --month-file "${TARGET_MONTH}=${MONTH_URL}" \
        --build-serving-shards "$MONTH_SHARDS_DIR" \
        --force-rebuild-serving-shards \
        --shard-workers 8 \
        -v
fi

echo "Done. Meta:"
if [[ "$TARGET_MONTH" == "all" ]]; then
    cat "$SHARDS_DIR/_meta.json"
else
    cat "${SHARDS_DIR}_${TARGET_MONTH}/_meta.json"
fi
