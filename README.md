# context

Automated act-checking pipeline with different conditions: 
- Raw fact-checking, following the RARR pipeline from [OpenFactCheck]
- With third-party reliability signals, from [DQR]
- With 1- or 2-hop structural information, from [CrediBench] 

On: 
- [CDL-Misinfo-Datasets]
- [...]

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
	my-run 100
```


### With 3rd-party reliability signals

The default scores used are from [DQR] and can be found under the `data` folder. 

### With structural information 

The structural information retrieved displays the domain's 1- and 2-hop neighbors from [CrediBench] using the hook in `scripts/hook.py`. The first run will need to first build the webgraph shards so may take longer. 

```bash
cd context

uv run python scripts/run_rarr_eval.py \
	--structural \
	--structural-shards-dir credibench-neighbors_serving_shards \
	--hf-dataset ComplexDataLab/Misinfo_Datasets \
	--hf-config default \
	--hf-split train \
	my-struct-run 100
```

## Data Analysis

To run benchmark analyses, set these parameters in `analyze.sh`:

```bash
DATASET="liar"
JUDGE="gpt-5-mini"
# JUDGE="Qwen/Qwen2.5-7B-Instruct"
CONDITION="ambig" # none | conflict | stale | opinion | unverif | ambig
THIRD_PARTY_RATINGS_FILE="data/domain_ratings.csv"
```

and `bash/sbatch analyze.sh`