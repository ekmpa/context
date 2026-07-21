# context

Automated fact-checking pipeline with different conditions:
- Raw fact-checking, following the RARR pipeline from [OpenFactCheck](https://openfactcheck.com)
- With third-party reliability signals, from [DQR](https://academic.oup.com/pnasnexus/article/2/9/pgad286/7258994?guestAccessKey=)
- With 1- or 2-hop structural information, from [CrediBench](https://huggingface.co/datasets/credi-net/CrediBench) 

On: 
- [CDL-Misinfo-Datasets](https://dl.acm.org/doi/abs/10.1145/3711896.3737437)

## Set-up

This repo uses uv, set it up with `uv sync` and install additional dependencies with `uv add [...]`.

## Fact-checking pipeline

### Raw

Within `run.sh`, set the arguments to the desired dataset, experiment condition, and model. `HF_*` vars are used only when `DATASET=climatecheck`; other datasets run from `data/<dataset>/test/data.jsonl` (bootstrapped automatically if missing).

```bash
cd context
sbatch run.sh
```

For example, for `fakecovid`,
```bash
cd context
uv run python scripts/run_rarr_eval.py \
	data/fakecovid/test/data.jsonl \
	--condition raw \
	--rarr-model gpt-5-mini
```



### With 3rd-party reliability signals

The default scores used are from [DQR] and can be found under the `data` folder. 

### With structural information 

The structural information retrieved displays the domain's 1- and 2-hop neighbors from [CrediBench](https://huggingface.co/datasets/credi-net/CrediBench)  using the hook in `scripts/hook.py`. The first run may take longer because shards must be available.

```bash
cd context
uv run python scripts/run_rarr_eval.py \
	data/fakecovid/test/data.jsonl \
	--condition structural \
	--structural-shards-dir "$SCRATCH/credibench-neighbors_serving_shards" \
	--rarr-model gpt-5-mini
```

Use `--hf-dataset/--hf-config/--hf-split` only for climatecheck-style Hugging Face runs.

For smaller runs, set `max_rows`

The default `run.sh` launcher now targets `Qwen/Qwen2.5-7B-Instruct` on GPU, so `sbatch run.sh` runs the hf-local path by default. Swap `RARR_MODEL` back to `gpt-5-mini` if you want the OpenAI backend instead.

## Data Analysis

To run benchmark analyses, set these parameters in `analyze.sh`:

```bash
DATASET="liar"
JUDGE="Qwen/Qwen2.5-7B-Instruct"
# JUDGE="gpt-5-mini"
CONDITION="ambig" # none | conflict | conflict_compare | stale | opinion | unverif | ambig
THIRD_PARTY_RATINGS_FILE="data/domain_ratings.csv"
```

For local open-source judges, request a GPU and use `--workers` as the local batch size. The backend is inferred from the model name in the Python entry points, so Qwen-style models such as `Qwen/Qwen2.5-7B-Instruct` automatically use the hf-local path.

Supported analysis conditions are:
- `none | conflict | conflict_compare | stale | opinion | unverif | ambig`

For `stale`, dataset date metadata is in `scripts/dataset_info.json` (for example LIAR=2017, FakeCovid=2020/date column).

and `bash/sbatch analyze.sh`
