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

```bash
cd context

uv run python scripts/run_rarr_eval.py \
	--hf-dataset ComplexDataLab/Misinfo_Datasets \
	--hf-config default \
	--hf-split train \
	--condition raw \
	--max-rows 100
```


### With 3rd-party reliability signals

The default scores used are from [DQR] and can be found under the `data` folder. 

### With structural information 

The structural information retrieved displays the domain's 1- and 2-hop neighbors from [CrediBench](https://huggingface.co/datasets/credi-net/CrediBench)  using the hook in `scripts/hook.py`. The first run may take longer because shards must be available.

```bash
cd context
uv run python scripts/run_rarr_eval.py \
	--condition structural \
	--structural-shards-dir credibench-neighbors_serving_shards \
	--hf-dataset ComplexDataLab/Misinfo_Datasets \
	--hf-config default \
	--hf-split train \
```

For smaller runs, set `max_rows`

## Data Analysis

To run benchmark analyses, set these parameters in `analyze.sh`:

```bash
DATASET="liar"
JUDGE="gpt-5-mini"
# JUDGE="Qwen/Qwen2.5-7B-Instruct"
CONDITION="ambig" # none | conflict | conflict_compare | stale | opinion | unverif | ambig
THIRD_PARTY_RATINGS_FILE="data/domain_ratings.csv"
```

Supported analysis conditions are:
- `none | conflict | conflict_compare | stale | opinion | unverif | ambig`

For `stale`, dataset date metadata is in `scripts/dataset_info.json` (for example LIAR=2017, FakeCovid=2020/date column).

and `bash/sbatch analyze.sh`
