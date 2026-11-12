"""Run the GP-SGBC parameter grid and summarize repeated runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from itertools import product
import json
import math
import os
from pathlib import Path
from time import perf_counter

from config import ALGORITHM_SETTINGS, DATASETS, EXPERIMENT, PARAMETERS

if EXPERIMENT.cpu_thread_limit is not None:
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(EXPERIMENT.cpu_thread_limit)

from core.data import load_dataset, minmax_scale
from core.metrics import evaluate_clustering, summarize_metric_rows
from gp_sgbc import GPSGBC, GPSGBCConfig


def parameter_grid(n_features: int) -> list[GPSGBCConfig]:
    configurations = []
    p1_values = sorted({math.ceil(n_features * r) for r in PARAMETERS["p1_ratios"]})
    for p1 in p1_values:
        if not 2 <= p1 <= n_features:
            continue
        p2_values = sorted({math.ceil(p1 * r) for r in PARAMETERS["p2_ratios"]})
        for p2, theta, weight, beta in product(
            p2_values, PARAMETERS["theta_values"], PARAMETERS["similarity_lambda"],
            PARAMETERS["redundancy_beta"],
        ):
            if p2 >= p1:
                continue
            configurations.append(GPSGBCConfig(
                p1=p1, p2=p2, purity=theta, pdmf_similarity_lambda=weight,
                redundancy_beta=beta, **ALGORITHM_SETTINGS,
            ))
    return configurations


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(name: str, experiment) -> dict:
    dataset = load_dataset(DATASETS[name])
    X = minmax_scale(dataset.X)
    grid = parameter_grid(dataset.n_features)
    output = experiment.output_root / experiment.run_id / name
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "algorithm": "GP-SGBC", "dataset": name,
        "dataset_path": str(DATASETS[name].path),
        "n_clusters": dataset.n_classes, "seeds": list(experiment.seeds),
        "nmi_average_method": experiment.nmi_average_method,
        "preprocessing": "global_featurewise_minmax_to_0_1",
        "local_fusion": "equal average of entropy and graph ranks",
        "selection_rule": "maximize mean ACC across all configured seeds",
        "configurations": [asdict(config) for config in grid],
    }
    metadata_path = output / "experiment_config.json"
    runs_path = output / "all_runs.csv"
    if experiment.resume and runs_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Experiment settings changed; choose a new --run-id")
        with runs_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = []
    write_json(metadata_path, metadata)
    completed = {
        (int(row["config_id"]), int(row["seed"])): row
        for row in rows if row["status"] == "ok"
    }
    metric_names = ("acc", "nmi", "ari", "ami", "f_measure", "macro_f1",
                    "pairwise_f1", "fmi", "purity", "rand_index")
    fields = ["dataset", "config_id", "seed", "status", "error",
              "runtime_seconds", "n_anchors", *metric_names]
    append = experiment.resume and runs_path.exists()
    with runs_path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not append:
            writer.writeheader()
        for config_id, config in enumerate(grid):
            for seed in experiment.seeds:
                if (config_id, seed) in completed:
                    continue
                row = dict.fromkeys(fields, "")
                row.update(dataset=name, config_id=config_id, seed=seed)
                started = perf_counter()
                try:
                    model = GPSGBC(config, dataset.n_classes, random_state=seed)
                    labels = model.fit_predict(X)
                    row["runtime_seconds"] = perf_counter() - started
                    row.update(evaluate_clustering(
                        dataset.y, labels,
                        nmi_average_method=experiment.nmi_average_method,
                    ).as_dict())
                    row.update(status="ok", n_anchors=model.result_.n_anchors)
                    completed[config_id, seed] = row
                except Exception as error:
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                    row["runtime_seconds"] = perf_counter() - started
                writer.writerow(row)
                stream.flush()
                print(f"{name}: config {config_id + 1}/{len(grid)}, seed {seed}: "
                      f"{row['status']}", flush=True)
    summaries = []
    for config_id, config in enumerate(grid):
        successful = [completed[config_id, seed] for seed in experiment.seeds
                      if (config_id, seed) in completed]
        summary = {"config_id": config_id, **asdict(config),
                   "success_runs": len(successful)}
        summary.update(dict.fromkeys(
            [f"{name}_{stat}" for name in (*metric_names, "runtime_seconds")
             for stat in ("mean", "std")], ""))
        summary.update(summarize_metric_rows(successful, (*metric_names, "runtime_seconds")))
        summaries.append(summary)
    write_csv(output / "grid_summary.csv", summaries)
    candidates = [row for row in summaries if row["success_runs"] == len(experiment.seeds)]
    best = max(candidates, key=lambda row: row["acc_mean"]) if candidates else None
    write_csv(output / "best_parameter_combination.csv", [best] if best else [])
    result = {"dataset": name, "planned_runs": len(grid) * len(experiment.seeds),
              "successful_runs": len(completed), "best_parameter_combination": best}
    write_json(output / "experiment_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="GP-SGBC experiments")
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS))
    parser.add_argument("--run-id", default=EXPERIMENT.run_id)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    experiment = replace(EXPERIMENT, datasets=tuple(args.datasets or EXPERIMENT.datasets),
                         run_id=args.run_id, resume=EXPERIMENT.resume and not args.no_resume)
    results = [run_dataset(name, experiment) for name in experiment.datasets]
    write_json(experiment.output_root / experiment.run_id / "summary.json", results)
    return 0 if all(item["best_parameter_combination"] is not None for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
