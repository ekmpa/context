#!/usr/bin/env bash
#SBATCH --job-name=qwen08b-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/mila/k/kondrupe/context

set -euo pipefail

module load python/3.10
source "$SCRATCH/ctxt-env/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - <<'PY'
import torch
from transformers import pipeline

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

pipe = pipeline(
    "text-generation",
    model=MODEL,
    model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},
)

if hasattr(pipe.model, "generation_config"):
    pipe.model.generation_config.max_length = None

prompt = "Reply with exactly one word: pong"
output = pipe(prompt, max_new_tokens=128, do_sample=False, return_full_text=False)
print(output[0]["generated_text"])
PY
