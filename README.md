# CGDT-CESN

**CGDT-CESN** means **Correlation-Guided and Dynamics-Trained Clustered Echo State Network**. It is intended for forecasting multivariate time series: each variable is represented by a reservoir cluster, and the model can represent both the variable's own temporal dynamics and interactions between variables.

Paper: Hariharan Srinivasan, Masoom Bhargava, and Sayan Gupta, [“CGDT-CESN: Correlation-Guided and Dynamics-Trained Clustered Echo State Network for Multivariate Time-Series Prediction” (arXiv:2610.22430)](https://arxiv.org/abs/2610.22430).

## The two ideas

1. **Correlation guidance (CG):** estimate pairwise relationships between variables in training data and use them to set the probability of connections between their reservoir clusters. `DataDrivenCESN_ER` implements this wiring step using an Erdős–Rényi reservoir. It uses absolute Pearson correlation on levels or absolute Spearman correlation on first differences (the default).
2. **Dynamics training (DT):** fit the per-variable readouts and iteratively fit inter-cluster reservoir weights (`Winter`) from the observed dynamics. `CGDT_CESN.fit_ridge_winter` implements the readout and inter-cluster weight training using closed-form ridge solves.

`CGDT_CESN.fit_ridge_winter` uses both ideas in sequence: it first applies correlation guidance to choose the inter-cluster connections, then trains the weights on those connections from the observed dynamics. `DataDrivenCESN_ER` provides the correlation-guided wiring and readout workflow without dynamics training of inter-cluster weights.

## Input and output

Use numeric arrays shaped **`(timesteps, variables)`**, with columns in a consistent variable order. Rows must be in chronological order. The number of columns must equal `n_inputs`; `CGDT_CESN` targets must have one column per cluster (`n_clusters`, normally the number of variables). Use past observations as inputs and the values you want forecast as targets, aligned by timestep. For example, one-step-ahead prediction can use `X = series[:-1]` and `Y = series[1:]`.

This interface accepts general multivariate series: sensor readings, finance, weather, process measurements, or event counts. It does not require cascade data. When estimating correlation wiring, pass one training series or several separate training series, each with shape `(timesteps, variables)`; they can be represented as a list or a 3-D array `(sequences, timesteps, variables)`. Use training data only for this step.

Expected predictions have shape `(timesteps_after_washout, variables)`. `washout` discards early reservoir steps from fitting and returned predictions; set it below the sequence length (and generally use the same value when aligning predictions with targets). `reset=True` starts prediction from a zero reservoir state.

## Install

Run from the repository directory:

```bash
pip install -r requirements.txt
```

Import as `Package_CGDT` when the repository's parent directory is on `PYTHONPATH`, or import the modules directly when this directory is on `PYTHONPATH`.

## Pipeline A: correlation-guided clustered ESN

This is the most direct workflow for correlation-based wiring and readout training. The example uses one multivariate training series:

```python
import numpy as np
from Package_CGDT.cg_cesn import DataDrivenCESN_ER
from Package_CGDT.utils import compute_metrics

# series: (timesteps, variables), ordered in time
X_train, Y_train = series[:-1], series[1:]
model = DataDrivenCESN_ER(
    n_inputs=series.shape[1],
    neurons_per_cluster=50,
    spectral_radius=0.9,
    p_in=0.1,
    input_connectivity=0.1,
    seed=7,
)

# Correlation is estimated from X_train and used to rewire cluster links.
model.fit(X_train, Y_train, X_train_3d=X_train[None, ...], washout=20)
pred = model.predict(X_test, reset=True, washout=20)
scores = compute_metrics(Y_test[20:], pred, metrics=("mse", "rmse", "nrmse"))
```

For multiple independent training series, provide `X_train_3d=np.stack(series_list)` or `X_train_3d=series_list`. Correlation scores are combined across sequences, weighted by their usable length. If `X_train_3d` is omitted, the model keeps its initial inter-cluster wiring and only fits the readout. After fitting, the fitted correlation matrix and connection probabilities are available as `model.correlation_matrix_` and `model.P_out_matrix_`.

## Pipeline B: correlation-guided, dynamics-trained CGDT-CESN

Use `CGDT_CESN` to estimate correlation-guided connectivity, then fit readout weights and train inter-cluster weights on those selected connections. `fit_ridge_winter` alternates readout fitting with a ridge solve for inter-cluster weights; it does not use gradients or a learning rate. By default, correlation is estimated from `inputs`. You can pass separate training sequences through `correlation_data`.

```python
from Package_CGDT.cgdt_cesn import CGDT_CESN

model = CGDT_CESN(
    n_inputs=series.shape[1],
    neurons_per_cluster=50,
    spectral_radius=0.9,
    p_in=0.1,
    input_connectivity=0.1,
    seed=7,
)
model.fit_ridge_winter(
    X_train, Y_train,
    washout=20,
    n_epochs=10,
    ridge_lambda=1e-8,
    winter_lambda=1e-4,
    correlation_method="spearman_diff",  # or "pearson"
)
pred = model.predict(X_test, reset=True, washout=20)
```

`fit_ridge_winter` expects `inputs` shaped `(T, n_inputs)` and `targets` shaped `(T, n_clusters)`; for correlation-guided CGDT-CESN, `n_clusters` must equal `n_inputs`. `correlation_data` may be one `(T, variables)` array, a list of such arrays, or a 3-D `(sequences, timesteps, variables)` array. The trained correlation matrix and connectivity probabilities are exposed as `correlation_matrix_` and `P_out_matrix_`; the sampled topology is `inter_cluster_mask_`. Inter-cluster training updates weights on this mask and leaves absent connections at zero. Training diagnostics are stored in `loss_history_` and `nrmse_history_`; with `verbose=True`, they are printed during fitting.

For either model, `p_in` controls the scale of guided inter-cluster connection probabilities; in CGDT-CESN the constructor's `p_out` initialization is replaced by the correlation-guided topology when `fit_ridge_winter` runs. The separate `fit` method only fits readouts and does not run correlation guidance or dynamics training.

## Metrics

`utils.compute_metrics(y_true, y_pred, metrics=(...))` returns the requested metrics by name. Available names are `"mse"`, `"rmse"`, and `"nrmse"`; NRMSE is RMSE divided by the target range (max minus min), computed separately per output and averaged. Constant target columns produce `NaN` NRMSE.

## Modules

- `esn.py`: standard `EchoStateNetwork`.
- `reservoirs.py`: the Erdős–Rényi clustered reservoir used by CG-ESN.
- `cg_cesn.py`: correlation analysis and correlation-guided clustered ESNs.
- `cgdt_cesn.py`: clustered ESN with ridge-trained inter-cluster weights.
- `utils.py`: MSE, RMSE, and NRMSE utilities.
