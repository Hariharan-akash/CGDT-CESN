"""Correlation-guided clustered echo state networks (CG-CESN)."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import adfuller, kpss

try:
    from .reservoirs import (
        EchoStateNetworkWithErdosRenyiClusters,
    )
except ImportError:  # Direct imports when this directory is on PYTHONPATH.
    from reservoirs import (
        EchoStateNetworkWithErdosRenyiClusters,
    )

class DataDrivenInterClusterMixin:

    # ================================================================
    # Stationarity checks
    # ================================================================

    @staticmethod
    def check_stationarity(ts):
        result = {}

        # ADF
        try:
            _, p_adf, _, _, _, _ = adfuller(ts, autolag='AIC')
            result["ADF_p"] = p_adf
        except Exception:
            result["ADF_p"] = np.nan

        # KPSS
        try:
            _, p_kpss, _, _ = kpss(ts, regression='c', nlags='auto')
            result["KPSS_p"] = p_kpss
        except Exception:
            result["KPSS_p"] = np.nan

        return result

    @staticmethod
    def stationarity_analysis(data, var_names):
        records = []

        if isinstance(data, np.ndarray):
            for i, var in enumerate(var_names):
                ts = data[:, i]
                raw = DataDrivenInterClusterMixin.check_stationarity(ts)
                diff = DataDrivenInterClusterMixin.check_stationarity(np.diff(ts))

                records.append({
                    "variable": var,
                    "ADF_raw": raw["ADF_p"],
                    "KPSS_raw": raw["KPSS_p"],
                    "ADF_diff": diff["ADF_p"],
                    "KPSS_diff": diff["KPSS_p"],
                })

        elif isinstance(data, list):
            for seq in data:
                for i, var in enumerate(var_names):
                    ts = seq[:, i]
                    raw = DataDrivenInterClusterMixin.check_stationarity(ts)
                    diff = DataDrivenInterClusterMixin.check_stationarity(np.diff(ts))

                    records.append({
                        "variable": var,
                        "ADF_raw": raw["ADF_p"],
                        "KPSS_raw": raw["KPSS_p"],
                        "ADF_diff": diff["ADF_p"],
                        "KPSS_diff": diff["KPSS_p"],
                    })

        return pd.DataFrame(records)

    # ================================================================
    # Correlation functions
    # ================================================================

    @staticmethod
    def pearson_corr(X):
        C = np.corrcoef(X.T)
        C = np.nan_to_num(np.abs(C), nan=0.0)
        np.fill_diagonal(C, 1.0)
        return C

    @staticmethod
    def spearman_diff_corr(X):
        if len(X) < 3:
            return None

        X_diff = np.diff(X, axis=0)
        n = X_diff.shape[1]
        C = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                xi = X_diff[:, i]
                xj = X_diff[:, j]

                if np.std(xi) < 1e-8 or np.std(xj) < 1e-8:
                    val = 0.0
                else:
                    rho, _ = spearmanr(xi, xj)
                    val = abs(rho) if not np.isnan(rho) else 0.0

                C[i, j] = val
                C[j, i] = val

        np.fill_diagonal(C, 1.0)
        return C

    # ================================================================
    # Main pipeline
    # ================================================================

    @staticmethod
    def compute_pipeline(data, var_names, method="spearman_diff"):
        corr_map = {
            "pearson": DataDrivenInterClusterMixin.pearson_corr,
            "spearman_diff": DataDrivenInterClusterMixin.spearman_diff_corr,
        }

        if method not in corr_map:
            raise ValueError("Invalid method")

        corr_fn = corr_map[method]

        if isinstance(data, np.ndarray):
            C = corr_fn(data)
            if C is None:
                raise ValueError("Time series too short")
            return C

        elif isinstance(data, list):
            weighted_sum = None
            total_weight = 0.0

            for seq in data:
                T_i = len(seq)
                eff_len = (T_i - 1) if method == "spearman_diff" else T_i

                if eff_len < 2:
                    continue

                C_i = corr_fn(seq)
                if C_i is None or np.isnan(C_i).all():
                    continue

                weight = float(eff_len)

                if weighted_sum is None:
                    weighted_sum = weight * C_i
                else:
                    weighted_sum += weight * C_i

                total_weight += weight

            if weighted_sum is None or total_weight == 0:
                raise ValueError("No usable cascades")

            return weighted_sum / total_weight

        else:
            raise ValueError("Invalid data format")

    # ================================================================
    # P_out mapping
    # ================================================================

    @staticmethod
    def correlation_to_pout(C, p_in, n_clusters):
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")

        p_out_max = p_in / (n_clusters - 1)
        P = p_out_max * C
        np.fill_diagonal(P, 0.0)
        return P

    # ================================================================
    # Reservoir rewiring
    # ================================================================

    @staticmethod
    def build_inter_cluster_connections(W, P, n_clusters, neurons_per_cluster, spectral_radius):
        Nc = neurons_per_cluster

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                ci_s, ci_e = i * Nc, (i + 1) * Nc
                cj_s, cj_e = j * Nc, (j + 1) * Nc

                p_ij = P[i, j]

                mask_ij = np.random.rand(Nc, Nc) < p_ij
                mask_ji = np.random.rand(Nc, Nc) < p_ij

                W[ci_s:ci_e, cj_s:cj_e] = (np.random.rand(Nc, Nc) * 2 - 1) * mask_ij
                W[cj_s:cj_e, ci_s:ci_e] = (np.random.rand(Nc, Nc) * 2 - 1) * mask_ji

        rho = max(abs(np.linalg.eigvals(W)))
        if rho > 0:
            W *= spectral_radius / rho

        return W

    # ================================================================
    # Full pipeline — delegates to module-level function to avoid the
    # MRO self-injection bug that occurs when a @staticmethod is called
    # via ClassName.method() from inside a subclass method.
    # ================================================================
    @staticmethod
    def compute_and_apply_pout(
        data, var_names, W, p_in, n_clusters,
        neurons_per_cluster, spectral_radius,
        method="spearman_diff"
    ):
        return _module_compute_and_apply_pout(
            data, var_names, W, p_in, n_clusters,
            neurons_per_cluster, spectral_radius, method
        )
# ================================================================
# Module-level pipeline function — called by DataDrivenInterClusterMixin
# .compute_and_apply_pout() AND by the three DataDrivenCESN fit() methods.
# Kept outside the class so it is always a plain function call with no
# MRO descriptor involvement.
# ================================================================
def _module_compute_and_apply_pout(
    data, var_names, W, p_in, n_clusters,
    neurons_per_cluster, spectral_radius,
    method="spearman_diff"
):
    C = DataDrivenInterClusterMixin.compute_pipeline(data, var_names, method)
    P = DataDrivenInterClusterMixin.correlation_to_pout(C, p_in, n_clusters)
    W = DataDrivenInterClusterMixin.build_inter_cluster_connections(
        W, P, n_clusters, neurons_per_cluster, spectral_radius
    )
    return {
        "C": C,
        "p_out_max": p_in / (n_clusters - 1),
        "P_out": P,
        "W": W,
    }

# Public alias for backward compatibility
compute_and_apply_pout = _module_compute_and_apply_pout


class DataDrivenCESN_ER(DataDrivenInterClusterMixin,
                        EchoStateNetworkWithErdosRenyiClusters):
    """Erdős–Rényi CESN with data-driven inter-cluster connectivity."""

    def __init__(self, *args, p_in=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_in = p_in

    def fit(self, inputs, outputs, X_train_3d=None,
            ridge_lambda=0.0, reset=False, washout=100):
        if X_train_3d is not None:
            var_names = [f"var_{i}" for i in range(self.n_inputs)]
            data = [X_train_3d[i] for i in range(X_train_3d.shape[0])] \
                   if isinstance(X_train_3d, np.ndarray) else X_train_3d
            result = _module_compute_and_apply_pout(
                data, var_names, self.W, self.p_in,
                self.n_clusters, self.neurons_per_cluster, self.spectral_radius
            )
            self.W                   = result["W"]
            self.correlation_matrix_ = result["C"]
            self.p_out_max_          = result["p_out_max"]
            self.P_out_matrix_       = result["P_out"]
        super().fit(inputs, outputs,
                    ridge_lambda=ridge_lambda, reset=True, washout=washout)


