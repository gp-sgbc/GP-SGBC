"""Dataset paths and conference experiment settings."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "standardized"

PARAMETERS = {
    "p1_ratios": (0.25, 0.50, 0.75),
    "p2_ratios": (0.05, 0.10, 0.25, 0.50, 0.75),
    "theta_values": tuple(i / 100 for i in range(70, 100, 5)),
    "similarity_lambda": (0.1, 0.5, 0.9),
    "redundancy_beta": (0.0, 0.1, 0.3, 0.5),
}

# Local entropy and graph ranks are averaged in gp_sgbc/feature_selection.py.
ALGORITHM_SETTINGS = {
    "pdmf_epsilon": 1e-8,
    "mutual_knn": True,
    "self_tuning_graph": True,
    "anchor_neighbors": 5,
    "weighted_kmeans_beta": 3.0,
    "weighted_kmeans_max_iter": 20,
    "split_kmeans_max_iter": 3,
    "tcut_kmeans_max_iter": 100,
    "tcut_kmeans_n_init": 3,
    "ball_parallel_jobs": 32,
    "keep_matlab_split_rule": True,
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path


DATASETS = {
    name: DatasetConfig(name, DATA_ROOT / f"{name}.npz")
    for name in (
        "USPS", "COIL20", "ORL", "Yale", "warpPIE10P",
        "GLIOMA", "TOX_171", "ALLAML", "SuCancer",
    )
}


@dataclass(frozen=True)
class ExperimentConfig:
    datasets: tuple[str, ...] = tuple(DATASETS)
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5, 41, 22, 21, 70, 5244)
    nmi_average_method: str = "geometric"
    output_root: Path = ROOT / "results"
    run_id: str = "conference"
    resume: bool = True
    cpu_thread_limit: int | None = None


EXPERIMENT = ExperimentConfig()
