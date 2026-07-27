from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CANONICAL_LABELS = ("ambiguous", "unverifiable", "opinion", "none_of_three")

HUMAN_COLUMN_TO_LABEL = {
	"Multiple justifiable answers? (0/1)": "ambiguous",
	"No justifiable answer? (0/1)": "unverifiable",
	"Opinion? (0/1)": "opinion",
}

LLM_LABEL_MAP = {
	"multiple_justifiable_answers": "ambiguous",
	"no_justifiable_answer": "unverifiable",
	"opinion": "opinion",
	"subjective": "opinion",
}


@dataclass
class AnnotationRecord:
	annotator: str
	labels_by_idx: Dict[str, str]
	total_rows: int
	indexed_rows: int
	usable_rows: int
	missing_rows: int
	multi_label_rows: int
	duplicate_idx_rows: int
	duplicate_idx_conflicts: int


def _clean_annotator_name(path: Path) -> str:
	name = path.stem
	name = re.sub(r"\s*-\s*(ClaimsKG|LIAR)\s*$", "", name, flags=re.IGNORECASE)
	name = re.sub(r"\s+", " ", name).strip()
	return name


def _normalize_idx(value: object) -> Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	if text == "":
		return None
	try:
		return str(int(float(text)))
	except ValueError:
		return text


def _parse_binary_flag(value: object) -> Optional[int]:
	if value is None:
		return None
	text = str(value).strip().lower()
	if text in {"", "na", "n/a", "null", "none", "nan"}:
		return None
	if text in {"1", "1.0", "true", "t", "yes", "y"}:
		return 1
	if text in {"0", "0.0", "false", "f", "no", "n"}:
		return 0
	return None


def _row_to_label(row: Dict[str, object]) -> Tuple[Optional[str], str]:
	positives: List[str] = []
	zeros = 0
	blanks = 0
	for col, label in HUMAN_COLUMN_TO_LABEL.items():
		flag = _parse_binary_flag(row.get(col))
		if flag == 1:
			positives.append(label)
		elif flag == 0:
			zeros += 1
		else:
			blanks += 1

	if len(positives) == 1:
		return positives[0], "ok"
	if len(positives) == 0 and zeros == len(HUMAN_COLUMN_TO_LABEL):
		# Annotator explicitly marked none of the three categories.
		return "none_of_three", "ok"
	if len(positives) == 0:
		return None, "missing"
	return None, "multi_label"


def _row_is_blank_padding(row: Dict[str, object]) -> bool:
	idx = _normalize_idx(row.get("idx") or row.get("index") or row.get("id"))
	claim = str(row.get("claim") or "").strip()
	if idx is not None or claim != "":
		return False
	for col in HUMAN_COLUMN_TO_LABEL:
		if str(row.get(col) or "").strip() != "":
			return False
	return True


def load_human_annotations(path: Path) -> AnnotationRecord:
	labels_by_idx: Dict[str, str] = {}
	total_rows = 0
	indexed_rows = 0
	usable_rows = 0
	missing_rows = 0
	multi_label_rows = 0
	duplicate_idx_rows = 0
	duplicate_idx_conflicts = 0

	with path.open("r", encoding="utf-8-sig", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			total_rows += 1
			idx = _normalize_idx(row.get("idx") or row.get("index") or row.get("id"))
			if idx is None:
				# Ignore physically empty/non-index rows from CSV exports.
				if not _row_is_blank_padding(row):
					missing_rows += 1
				continue
			indexed_rows += 1

			label, status = _row_to_label(row)
			if status == "missing":
				missing_rows += 1
				continue
			if status == "multi_label":
				multi_label_rows += 1
				continue

			prev = labels_by_idx.get(idx)
			if prev is not None:
				duplicate_idx_rows += 1
				if prev != label:
					duplicate_idx_conflicts += 1
					continue
				continue

			labels_by_idx[idx] = label
			usable_rows += 1

	return AnnotationRecord(
		annotator=_clean_annotator_name(path),
		labels_by_idx=labels_by_idx,
		total_rows=total_rows,
		indexed_rows=indexed_rows,
		usable_rows=usable_rows,
		missing_rows=missing_rows,
		multi_label_rows=multi_label_rows,
		duplicate_idx_rows=duplicate_idx_rows,
		duplicate_idx_conflicts=duplicate_idx_conflicts,
	)


def load_llm_labels(path: Path) -> Dict[str, str]:
	labels_by_idx: Dict[str, str] = {}
	with path.open("r", encoding="utf-8-sig", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			idx = _normalize_idx(row.get("index") or row.get("idx") or row.get("id"))
			if idx is None:
				continue
			raw = str(row.get("judge_label") or "").strip().lower()
			label = LLM_LABEL_MAP.get(raw)
			if label is None:
				continue
			labels_by_idx[idx] = label
	return labels_by_idx


def _cohen_kappa(a: List[str], b: List[str], labels: Iterable[str]) -> float:
	n = len(a)
	if n == 0:
		return float("nan")
	po = sum(1 for x, y in zip(a, b) if x == y) / n
	cnt_a = Counter(a)
	cnt_b = Counter(b)
	pe = 0.0
	for label in labels:
		pe += (cnt_a[label] / n) * (cnt_b[label] / n)
	if math.isclose(1.0 - pe, 0.0):
		return float("nan")
	return (po - pe) / (1.0 - pe)


def _pearson_r(x: List[float], y: List[float]) -> float:
	n = len(x)
	if n != len(y) or n < 2:
		return float("nan")
	mean_x = sum(x) / n
	mean_y = sum(y) / n
	cov = sum((xv - mean_x) * (yv - mean_y) for xv, yv in zip(x, y))
	var_x = sum((xv - mean_x) ** 2 for xv in x)
	var_y = sum((yv - mean_y) ** 2 for yv in y)
	if math.isclose(var_x, 0.0) or math.isclose(var_y, 0.0):
		return float("nan")
	return cov / math.sqrt(var_x * var_y)


def _average_ranks(values: List[float]) -> List[float]:
	indexed = sorted(enumerate(values), key=lambda item: item[1])
	ranks = [0.0] * len(values)
	i = 0
	while i < len(indexed):
		j = i
		while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
			j += 1
		avg_rank = (i + j + 2) / 2.0  # 1-based average rank for tied groups.
		for k in range(i, j + 1):
			ranks[indexed[k][0]] = avg_rank
		i = j + 1
	return ranks


def _spearman_rho(x: List[float], y: List[float]) -> float:
	n = len(x)
	if n != len(y) or n < 2:
		return float("nan")
	rank_x = _average_ranks(x)
	rank_y = _average_ranks(y)
	return _pearson_r(rank_x, rank_y)


def _macro_f1(y_true: List[str], y_pred: List[str], labels: Iterable[str]) -> float:
	scores: List[float] = []
	for label in labels:
		tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
		fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
		fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
		if tp == 0 and fp == 0 and fn == 0:
			scores.append(0.0)
			continue
		precision = tp / (tp + fp) if (tp + fp) else 0.0
		recall = tp / (tp + fn) if (tp + fn) else 0.0
		if precision + recall == 0:
			scores.append(0.0)
		else:
			scores.append(2 * precision * recall / (precision + recall))
	if not scores:
		return float("nan")
	return sum(scores) / len(scores)


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
	if n == 0:
		return (float("nan"), float("nan"))
	phat = successes / n
	denom = 1 + (z**2) / n
	center = (phat + (z**2) / (2 * n)) / denom
	radius = z * math.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n) / denom
	return (center - radius, center + radius)


def _pairwise_rows(records: List[AnnotationRecord]) -> List[Dict[str, object]]:
	rows: List[Dict[str, object]] = []
	label_to_num = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}
	for i in range(len(records)):
		for j in range(i + 1, len(records)):
			a = records[i]
			b = records[j]
			overlap = sorted(set(a.labels_by_idx) & set(b.labels_by_idx))
			y_a = [a.labels_by_idx[idx] for idx in overlap]
			y_b = [b.labels_by_idx[idx] for idx in overlap]
			y_a_num = [float(label_to_num[label]) for label in y_a]
			y_b_num = [float(label_to_num[label]) for label in y_b]
			n = len(overlap)
			matches = sum(1 for x, y in zip(y_a, y_b) if x == y)
			acc = (matches / n) if n else float("nan")
			kappa = _cohen_kappa(y_a, y_b, CANONICAL_LABELS)
			pearson = _pearson_r(y_a_num, y_b_num)
			spearman = _spearman_rho(y_a_num, y_b_num)
			lo, hi = _wilson_interval(matches, n)
			rows.append(
				{
					"annotator_a": a.annotator,
					"annotator_b": b.annotator,
					"n_overlap": n,
					"percent_agreement": acc,
					"agreement_ci95_low": lo,
					"agreement_ci95_high": hi,
					"cohen_kappa": kappa,
					"pearson_r": pearson,
					"spearman_rho": spearman,
				}
			)
	return rows


def _fleiss_kappa_complete(records: List[AnnotationRecord]) -> Tuple[float, int]:
	if not records:
		return float("nan"), 0

	common = set(records[0].labels_by_idx)
	for rec in records[1:]:
		common &= set(rec.labels_by_idx)
	common_items = sorted(common)
	if not common_items:
		return float("nan"), 0

	n_raters = len(records)
	if n_raters < 2:
		return float("nan"), len(common_items)

	p_i_values: List[float] = []
	label_totals = Counter()
	for idx in common_items:
		counts = Counter(rec.labels_by_idx[idx] for rec in records)
		for label in CANONICAL_LABELS:
			label_totals[label] += counts[label]
		numerator = sum(v * v for v in counts.values()) - n_raters
		denominator = n_raters * (n_raters - 1)
		p_i_values.append(numerator / denominator)

	p_bar = sum(p_i_values) / len(p_i_values)
	total_assignments = len(common_items) * n_raters
	p_e = sum((label_totals[label] / total_assignments) ** 2 for label in CANONICAL_LABELS)
	if math.isclose(1.0 - p_e, 0.0):
		return float("nan"), len(common_items)
	return (p_bar - p_e) / (1.0 - p_e), len(common_items)


def _krippendorff_alpha_nominal(records: List[AnnotationRecord]) -> Tuple[float, int]:
	if not records:
		return float("nan"), 0

	all_ids = sorted(set().union(*(set(rec.labels_by_idx) for rec in records)))
	coincidence = {c: {k: 0.0 for k in CANONICAL_LABELS} for c in CANONICAL_LABELS}
	used_items = 0

	for idx in all_ids:
		ratings = [rec.labels_by_idx[idx] for rec in records if idx in rec.labels_by_idx]
		m = len(ratings)
		if m < 2:
			continue
		used_items += 1
		counts = Counter(ratings)
		denom = m - 1
		for c in CANONICAL_LABELS:
			for k in CANONICAL_LABELS:
				if c == k:
					val = counts[c] * max(counts[c] - 1, 0)
				else:
					val = counts[c] * counts[k]
				coincidence[c][k] += val / denom

	if used_items == 0:
		return float("nan"), 0

	n_c = {c: sum(coincidence[c][k] for k in CANONICAL_LABELS) for c in CANONICAL_LABELS}
	n_total = sum(n_c.values())
	if n_total <= 1:
		return float("nan"), used_items

	do_num = sum(
		coincidence[c][k]
		for c in CANONICAL_LABELS
		for k in CANONICAL_LABELS
		if c != k
	)
	do_den = n_total - 1
	d_observed = do_num / do_den if do_den > 0 else float("nan")

	de_num = 0.0
	for c in CANONICAL_LABELS:
		for k in CANONICAL_LABELS:
			if c != k:
				de_num += n_c[c] * n_c[k]
	d_expected = de_num / (n_total * (n_total - 1))

	if math.isclose(d_expected, 0.0):
		return float("nan"), used_items
	alpha = 1.0 - (d_observed / d_expected)
	return alpha, used_items


def _llm_vs_reference_rows(
	llm_labels: Dict[str, str],
	refs: Dict[str, Dict[str, str]],
) -> List[Dict[str, object]]:
	rows: List[Dict[str, object]] = []
	for name, ref_labels in refs.items():
		overlap = sorted(set(llm_labels) & set(ref_labels))
		y_true = [ref_labels[idx] for idx in overlap]
		y_pred = [llm_labels[idx] for idx in overlap]
		n = len(overlap)
		matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
		acc = (matches / n) if n else float("nan")
		lo, hi = _wilson_interval(matches, n)
		rows.append(
			{
				"reference": name,
				"n_overlap": n,
				"accuracy": acc,
				"accuracy_ci95_low": lo,
				"accuracy_ci95_high": hi,
				"cohen_kappa": _cohen_kappa(y_true, y_pred, CANONICAL_LABELS),
				"macro_f1": _macro_f1(y_true, y_pred, CANONICAL_LABELS),
			}
		)
	return rows


def _majority_human_label(records: List[AnnotationRecord]) -> Dict[str, str]:
	all_ids = sorted(set().union(*(set(rec.labels_by_idx) for rec in records)))
	out: Dict[str, str] = {}
	for idx in all_ids:
		labels = [rec.labels_by_idx[idx] for rec in records if idx in rec.labels_by_idx]
		if not labels:
			continue
		counts = Counter(labels)
		max_count = max(counts.values())
		winners = [k for k, v in counts.items() if v == max_count]
		if len(winners) == 1:
			out[idx] = winners[0]
	return out


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if not rows:
		with path.open("w", encoding="utf-8", newline="") as f:
			f.write("\n")
		return
	fieldnames = list(rows[0].keys())
	with path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def analyze_dataset(dataset: str, annot_dir: Path, llm_file: Path, output_dir: Path) -> Dict[str, object]:
	annot_files = sorted(annot_dir.glob("*.csv"))
	if not annot_files:
		raise FileNotFoundError(f"No annotation CSV files found in {annot_dir}")

	records = [load_human_annotations(path) for path in annot_files]
	llm_labels = load_llm_labels(llm_file)

	pairwise = _pairwise_rows(records)
	pairwise_kappas = [row["cohen_kappa"] for row in pairwise if isinstance(row.get("cohen_kappa"), float)]
	pairwise_agreements = [
		row["percent_agreement"] for row in pairwise if isinstance(row.get("percent_agreement"), float)
	]
	pairwise_pearsons = [row["pearson_r"] for row in pairwise if isinstance(row.get("pearson_r"), float)]
	pairwise_spearmans = [row["spearman_rho"] for row in pairwise if isinstance(row.get("spearman_rho"), float)]

	# Light's kappa: mean of pairwise Cohen's kappa values across all rater pairs.
	lights_kappa = float("nan")
	if pairwise_kappas:
		lights_kappa = sum(pairwise_kappas) / len(pairwise_kappas)

	fleiss, fleiss_n = _fleiss_kappa_complete(records)
	alpha, alpha_n = _krippendorff_alpha_nominal(records)

	refs = {rec.annotator: rec.labels_by_idx for rec in records}
	majority = _majority_human_label(records)
	refs["human_majority"] = majority
	llm_vs = _llm_vs_reference_rows(llm_labels, refs)

	summary = {
		"dataset": dataset,
		"inputs": {
			"annotation_dir": str(annot_dir),
			"llm_file": str(llm_file),
		},
		"counts": {
			"n_annotators": len(records),
			"n_llm_labeled": len(llm_labels),
			"n_unique_human_items": len(set().union(*(set(r.labels_by_idx) for r in records))),
			"n_majority_items": len(majority),
		},
		"annotator_coverage": [
			{
				"annotator": r.annotator,
				"total_rows": r.total_rows,
				"indexed_rows": r.indexed_rows,
				"usable_rows": r.usable_rows,
				"missing_rows": r.missing_rows,
				"multi_label_rows": r.multi_label_rows,
				"duplicate_idx_rows": r.duplicate_idx_rows,
				"duplicate_idx_conflicts": r.duplicate_idx_conflicts,
			}
			for r in records
		],
		"human_agreement": {
			"pairwise": pairwise,
			"pairwise_summary": {
				"n_pairs": len(pairwise),
				"mean_pairwise_percent_agreement": (
					(sum(pairwise_agreements) / len(pairwise_agreements)) if pairwise_agreements else float("nan")
				),
				"mean_pairwise_cohen_kappa": lights_kappa,
				"mean_pairwise_pearson_r": (
					(sum(pairwise_pearsons) / len(pairwise_pearsons)) if pairwise_pearsons else float("nan")
				),
				"mean_pairwise_spearman_rho": (
					(sum(pairwise_spearmans) / len(pairwise_spearmans)) if pairwise_spearmans else float("nan")
				),
				"min_pairwise_cohen_kappa": min(pairwise_kappas) if pairwise_kappas else float("nan"),
				"max_pairwise_cohen_kappa": max(pairwise_kappas) if pairwise_kappas else float("nan"),
				"min_pairwise_pearson_r": min(pairwise_pearsons) if pairwise_pearsons else float("nan"),
				"max_pairwise_pearson_r": max(pairwise_pearsons) if pairwise_pearsons else float("nan"),
				"min_pairwise_spearman_rho": min(pairwise_spearmans) if pairwise_spearmans else float("nan"),
				"max_pairwise_spearman_rho": max(pairwise_spearmans) if pairwise_spearmans else float("nan"),
			},
			"lights_kappa": lights_kappa,
			"fleiss_kappa_complete_items": fleiss,
			"fleiss_n_complete_items": fleiss_n,
			"krippendorff_alpha_nominal": alpha,
			"krippendorff_n_items": alpha_n,
			"group_mean_pairwise_pearson_r": (
				(sum(pairwise_pearsons) / len(pairwise_pearsons)) if pairwise_pearsons else float("nan")
			),
			"group_mean_pairwise_spearman_rho": (
				(sum(pairwise_spearmans) / len(pairwise_spearmans)) if pairwise_spearmans else float("nan")
			),
		},
		"llm_vs_human": llm_vs,
	}

	output_dir.mkdir(parents=True, exist_ok=True)
	with (output_dir / f"{dataset}_agreement_summary.json").open("w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=True)

	_write_csv(output_dir / f"{dataset}_pairwise_human_agreement.csv", pairwise)
	_write_csv(output_dir / f"{dataset}_llm_vs_human.csv", llm_vs)
	_write_csv(output_dir / f"{dataset}_annotator_coverage.csv", summary["annotator_coverage"])

	return summary


def _default_paths() -> Tuple[Path, Path, Path]:
	scripts_dir = Path(__file__).resolve().parent
	context_dir = scripts_dir.parent
	repo_root = context_dir.parent
	annot_root = repo_root / "annot"
	samples_root = context_dir / "eval_results" / "samples"
	output_root = context_dir / "eval_results" / "annotation_agreement"
	return annot_root, samples_root, output_root


def main() -> None:
	annot_default, samples_default, output_default = _default_paths()

	parser = argparse.ArgumentParser(
		description="Evaluate human-human and LLM-human agreement for claimskg/liar annotation subsets."
	)
	parser.add_argument(
		"--datasets",
		nargs="+",
		default=["claimskg", "liar"],
		choices=["claimskg", "liar"],
		help="Datasets to evaluate.",
	)
	parser.add_argument(
		"--annot-root",
		type=Path,
		default=annot_default,
		help="Directory containing per-dataset human annotation folders.",
	)
	parser.add_argument(
		"--samples-root",
		type=Path,
		default=samples_default,
		help="Directory containing <dataset>_sampled_claims.csv LLM judgment files.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=output_default,
		help="Directory where evaluation reports are written.",
	)
	args = parser.parse_args()

	all_summaries: List[Dict[str, object]] = []
	for dataset in args.datasets:
		annot_dir = args.annot_root / dataset
		llm_file = args.samples_root / f"{dataset}_sampled_claims.csv"
		if not annot_dir.exists():
			raise FileNotFoundError(f"Annotation directory not found: {annot_dir}")
		if not llm_file.exists():
			raise FileNotFoundError(f"LLM sample file not found: {llm_file}")

		summary = analyze_dataset(dataset, annot_dir, llm_file, args.output_dir)
		all_summaries.append(summary)

		print(f"[{dataset}] annotators={summary['counts']['n_annotators']} llm_rows={summary['counts']['n_llm_labeled']}")
		print(
			f"[{dataset}] fleiss_kappa={summary['human_agreement']['fleiss_kappa_complete_items']:.4f} "
			f"krippendorff_alpha={summary['human_agreement']['krippendorff_alpha_nominal']:.4f}"
		)

	combined_path = args.output_dir / "combined_agreement_summary.json"
	with combined_path.open("w", encoding="utf-8") as f:
		json.dump(all_summaries, f, indent=2, ensure_ascii=True)
	print(f"Wrote combined summary to {combined_path}")


if __name__ == "__main__":
	main()
