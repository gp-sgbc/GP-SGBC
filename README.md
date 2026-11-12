# GP-SGBC

Gaussian-PDMF feature selection and granular-ball spectral clustering.
The pipeline uses weighted K-means pseudo-labels, global entropy feature
selection, local entropy/graph rank averaging with redundancy control,
granular-ball anchors, and Transfer Cut clustering.

## Installation

Python 3.11:

```bash
pip install -r requirements.txt
```

Place each dataset at `data/standardized/<name>.npz`, containing a numeric
feature matrix `X` (samples × features) and a label vector `y`.
Datasets and experiment results are not included.

## Run

```bash
python run.py
python run.py --datasets ORL Yale --run-id conference
```

Edit `config.py` to set dataset paths, parameter grids, seeds, and parallelism.
The conference grid uses global ratios `{0.25, 0.50, 0.75}`, local ratios
`{0.05, 0.10, 0.25, 0.50, 0.75}`, purity thresholds `0.70` to `0.95` in steps
of `0.05`, similarity weights `{0.1, 0.5, 0.9}`, and redundancy penalties
`{0, 0.1, 0.3, 0.5}`. Feature counts use the ceiling of each ratio product,
with `p2 < p1`. Local entropy and graph ranks receive equal weight.
Both the runner and Python API use `pdmf_neighbors=5` and `graph_neighbors=5`
by default.

The runner normalizes features to `[0, 1]`, runs ten seeds, and saves per-run
metrics, parameter means and standard deviations, and the configuration with
the highest mean ACC under `results/<run-id>/`. NMI uses geometric
normalization; F-measure is class-size weighted. All reported metrics for the
selected configuration use the same runs. Completed runs are resumed by default.

## Python API

```python
from core.data import minmax_scale
from gp_sgbc import GPSGBC, GPSGBCConfig

model = GPSGBC(GPSGBCConfig(p1=100, p2=10), n_clusters=5, random_state=1)
labels = model.fit_predict(minmax_scale(X))
```
