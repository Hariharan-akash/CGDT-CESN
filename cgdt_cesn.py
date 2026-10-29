import numpy as np
from scipy.linalg import pinv

try:
    from .cg_cesn import compute_and_apply_pout
except ImportError:  # Direct imports when this directory is on PYTHONPATH.
    from cg_cesn import compute_and_apply_pout


# ================================================================
# CESN_ER_RidgeWinter
# ================================================================
#
# Single class combining correlation-guided ER connectivity and dynamics training:
#
#   Connectivity — correlation-derived inter-cluster topology
#     Correlation-derived edge probabilities; sampled topology is fixed
#     while its inter-cluster weights are trained.
#
#   Forward pass  — EchoStateNetworkWithErdosRenyiClusters
#     Erdos-Renyi intra-cluster topology, per-cluster readout Wout.
#     Matches cesn_er.py exactly.
#
#   Backward pass — RidgeInterClusterMixin
#     Coordinate-descent ridge regression on Wout and Winter.
#     Matches cesn_ridge_winter_1.py with two numerical fixes:
#       1. Regularised inverse for x_c*(t)
#       2. Valid-timestep masking before arctanh
#
# Reservoir update (forward pass)
# --------------------------------
#   a(t)   = W · x(t-1)  +  Win · u(t)
#   x(t)   = (1 - α) · x(t-1)  +  α · tanh(a(t))
#   y_c(t) = Wout_c · x_c(t)
#   where α = leak_rate
#
# Winter ridge regression (backward pass)
# ----------------------------------------
# For cluster c:
#   a_c(t) = Wintra_c · x_c(t-1)
#           + Σ_{i≠c}  Winter[c,i] · x_i(t-1)
#           + Win_c · u(t)
#
# Wintra and Win are FIXED.  Winter[c,i] is solved via ridge.
#
# Step 1 — Regularised ideal state:
#   Minimise ||Wout_c·x_c* - y_c||^2 + inv_lambda·||x_c* - x_c(t-1)||^2
#   Solution: A = Wout_c^T·Wout_c + inv_lambda·I
#             x_c*(t) = A^{-1}·(Wout_c^T·y_c(t) + inv_lambda·x_c(t-1))
#
#   Why this matters: pure projection (inv_lambda=0) can send x_c*(t)
#   to ±1 when Wout_c is still imprecise (early epochs, small N).
#   The regularisation anchors x_c*(t) near the current state.
#
# Step 2 — Invert leaky update:
#   tanh(a_c*(t)) = (x_c*(t) - (1-α)·x_c(t-1)) / α
#   Only timesteps with |tanh(a_c*(t))| < clip_val are used.
#   Saturated timesteps (near ±1) produce arctanh ≈ ±∞ and
#   corrupt the Winter solve — masking them out prevents blowup.
#
# Step 3 — Subtract fixed contributions (valid timesteps only):
#   r_c(t) = arctanh(tanh(a_c*(t))) - Wintra·x_c(t-1) - Win·u(t)
#
# Step 4 — Ridge regression:
#   r_c = X_inter @ Winter[c,:]^T   X_inter = [x_i(t-1) for i≠c]
# ================================================================


class CGDT_CESN:
    """
    Correlation-guided ER CESN with coordinate-descent ridge training.

      - Wout   : per-cluster ridge regression  (closed form)
      - topology: correlation-guided inter-cluster connectivity
      - Winter : ridge regression on residual pre-activations over selected edges

    Parameters
    ----------
    n_inputs            : int
    neurons_per_cluster : int
    n_clusters          : int    default = n_inputs
    spectral_radius     : float
    leak_rate           : float  α in leaky integrator
    input_scaling       : float
    p_in                : float  intra-cluster ER probability
    p_out               : float  inter-cluster ER probability (init)
    input_connectivity  : float
    seed                : int
    """

    def __init__(
        self,
        n_inputs,
        neurons_per_cluster,
        n_clusters=None,
        spectral_radius=0.9,
        leak_rate=1.0,
        input_scaling=1.0,
        p_in=0.1,
        p_out=0.01,
        input_connectivity=0.1,
        seed=None,
    ):
        np.random.seed(seed)
        if n_clusters is None:
            n_clusters = n_inputs

        self.n_inputs            = n_inputs
        self.neurons_per_cluster = neurons_per_cluster
        self.n_clusters          = n_clusters
        self.n_reservoir         = n_clusters * neurons_per_cluster
        self.n_outputs           = n_inputs

        self.spectral_radius = spectral_radius
        self.leak_rate       = leak_rate
        self.input_scaling   = input_scaling
        self.p_in            = p_in
        self.p_out           = p_out

        self.Win              = np.zeros((self.n_reservoir, self.n_inputs))
        self.W                = np.zeros((self.n_reservoir, self.n_reservoir))
        self.structure_matrix = np.zeros_like(self.W)

        self._initialize_weights(input_connectivity, spectral_radius, input_scaling)

        self.Wout_clusters   = [
            np.zeros((1, self.neurons_per_cluster))
            for _ in range(self.n_clusters)
        ]
        self.reservoir_state = np.zeros((self.n_reservoir,))

    # ------------------------------------------------------------------ #
    #  Weight initialisation  (matches cesn_er.py exactly)               #
    # ------------------------------------------------------------------ #
    def _initialize_weights(self, input_connectivity, spectral_radius,
                             input_scaling):
        Nc = self.neurons_per_cluster

        for cluster_idx in range(self.n_clusters):
            cs = cluster_idx * Nc
            ce = (cluster_idx + 1) * Nc

            # Input weights — cluster-specific
            self.Win[cs:ce, cluster_idx] = (
                (np.random.rand(Nc) - 0.5) * 2 * input_scaling
            )
            mask = np.random.rand(Nc) < input_connectivity
            self.Win[cs:ce, cluster_idx] = np.multiply(
                self.Win[cs:ce, cluster_idx], mask
            )

            # Intra-cluster connections (Erdos-Renyi)
            for i in range(cs, ce):
                for j in range(cs, ce):
                    if i != j and np.random.rand() < self.p_in:
                        self.W[i, j]                = (np.random.rand() - 0.5) * 2
                        self.structure_matrix[i, j] = 1

        # Inter-cluster connections (Erdos-Renyi)
        for ci in range(self.n_clusters):
            for cj in range(ci + 1, self.n_clusters):
                ci_s, ci_e = ci * Nc, (ci + 1) * Nc
                cj_s, cj_e = cj * Nc, (cj + 1) * Nc
                for ni in range(ci_s, ci_e):
                    for nj in range(cj_s, cj_e):
                        if np.random.rand() < self.p_out:
                            self.W[ni, nj]                = (np.random.rand() - 0.5) * 2
                            self.W[nj, ni]                = (np.random.rand() - 0.5) * 2
                            self.structure_matrix[ni, nj] = 2
                            self.structure_matrix[nj, ni] = 2

        rhoW = max(abs(np.linalg.eigvals(self.W)))
        if rhoW > 0:
            self.W *= spectral_radius / rhoW

    # ------------------------------------------------------------------ #
    #  Forward pass  (matches cesn_er.py exactly)                        #
    # ------------------------------------------------------------------ #
    def _update_reservoir(self, input_signal):
        input_signal   = input_signal.reshape(-1)
        pre_activation = (np.dot(self.Win, input_signal)
                          + np.dot(self.W, self.reservoir_state))
        self.reservoir_state = (
            (1 - self.leak_rate) * self.reservoir_state
            + self.leak_rate * np.tanh(pre_activation)
        )

    def reset_reservoir(self):
        self.reservoir_state = np.zeros((self.n_reservoir,))

    def _collect_states(self, inputs, washout=0):
        """Run reservoir over inputs, return all states (T, n_reservoir)."""
        self.reset_reservoir()
        T      = len(inputs)
        states = np.zeros((T, self.n_reservoir))
        for t, x in enumerate(inputs):
            self._update_reservoir(x)
            states[t] = self.reservoir_state
        return states

    # ------------------------------------------------------------------ #
    #  Wout-only fit  (matches cesn_er.py exactly)                       #
    # ------------------------------------------------------------------ #
    def fit(self, inputs, outputs, ridge_lambda=0.0, reset=False,
            washout=100):
        """Train per-cluster Wout. Winter NOT modified."""
        if reset:
            self.reset_reservoir()

        Nc      = self.neurons_per_cluster
        n_steps = len(inputs)
        states  = np.zeros((n_steps, self.n_reservoir))
        for t, x in enumerate(inputs):
            self._update_reservoir(x)
            states[t] = self.reservoir_state

        states  = states[washout:]
        outputs = outputs[washout:]

        for cluster_idx in range(self.n_clusters):
            cs     = cluster_idx * Nc
            ce     = (cluster_idx + 1) * Nc
            Sc     = states[:, cs:ce]
            target = outputs[:, cluster_idx].reshape(-1, 1)

            if ridge_lambda > 0:
                I = np.identity(Nc)
                self.Wout_clusters[cluster_idx] = np.dot(
                    target.T,
                    np.dot(Sc, np.linalg.inv(
                        np.dot(Sc.T, Sc) + ridge_lambda * I
                    ))
                )
            else:
                self.Wout_clusters[cluster_idx] = np.dot(
                    target.T, pinv(Sc.T)
                )

    def predict(self, inputs, reset=False, washout=100):
        """
        Predict using per-cluster readouts.
        Signature matches cesn_er.py: reset=False, washout=100.
        """
        if reset:
            self.reset_reservoir()

        Nc      = self.neurons_per_cluster
        n_steps = len(inputs)
        preds   = np.zeros((n_steps, self.n_outputs))
        for t, x in enumerate(inputs):
            self._update_reservoir(x)
            for c in range(self.n_clusters):
                cs, ce      = c * Nc, (c + 1) * Nc
                preds[t, c] = np.dot(
                    self.Wout_clusters[c],
                    self.reservoir_state[cs:ce]
                ).item()
        return preds[washout:]

    # ------------------------------------------------------------------ #
    #  Backward pass helpers                                              #
    # ------------------------------------------------------------------ #
    def _build_inter_cluster_mask(self):
        if hasattr(self, "inter_cluster_mask_"):
            return self.inter_cluster_mask_
        Nc   = self.neurons_per_cluster
        K    = self.n_clusters
        N    = self.n_reservoir
        mask = np.zeros((N, N), dtype=bool)
        for c in range(K):
            for i in range(K):
                if i != c:
                    mask[c*Nc:(c+1)*Nc, i*Nc:(i+1)*Nc] = True
        return mask

    def _ridge_wout(self, states, targets, ridge_lambda, washout):
        """Per-cluster ridge on Wout."""
        Nc = self.neurons_per_cluster
        S  = states[washout:]
        Y  = targets[washout:]
        for c in range(self.n_clusters):
            cs, ce = c * Nc, (c + 1) * Nc
            Sc     = S[:, cs:ce]
            yc     = Y[:, c].reshape(-1, 1)
            I      = np.eye(Nc)
            self.Wout_clusters[c] = (
                yc.T @ Sc @ np.linalg.inv(Sc.T @ Sc + ridge_lambda * I)
            )

    def _ridge_winter(self, states, inputs, targets, washout,
                      winter_lambda, inv_lambda=1e-1, clip_val=0.95):
        """
        Ridge regression for Winter[c,:] per cluster.

        Two numerical safeguards over plain projection:

        1. Regularised ideal state x_c*(t)
           Minimises ||Wout_c·x_c* - y_c||^2 + inv_lambda·||x_c* - x_c(t-1)||^2
           Prevents x_c* from reaching saturation when Wout_c is imprecise
           (early epochs or small training N).
           inv_lambda=0 recovers the plain projection.

        2. Valid-timestep masking
           After inverting the leaky update, timesteps where
           |tanh(a_c*(t))| >= clip_val are discarded.
           arctanh blows up near ±1 — these timesteps corrupt the
           Winter solve and cause the observed NRMSE blowup at
           intermediate N (e.g. N=1000 in VAR experiments).
        """
        Nc    = self.neurons_per_cluster
        K     = self.n_clusters
        N     = self.n_reservoir
        alpha = self.leak_rate

        S = states[washout:]      # (T', N)
        Y = targets[washout:]     # (T', K)
        U = inputs[washout:]      # (T', n_inputs)

        # x(t-1): previous state for each row of S
        if washout > 0:
            S_prev = np.vstack([
                states[washout - 1:washout],   # (1, N)
                S[:-1],                         # (T'-1, N)
            ])                                  # (T', N)
        else:
            S_prev = np.vstack([
                np.zeros((1, N)),
                S[:-1],
            ])

        for c in range(K):
            cs, ce = c * Nc, (c + 1) * Nc

            # ── Step 1: regularised ideal state ──────────────────────
            wout_c   = self.Wout_clusters[c]           # (1, Nc)
            x_c_prev = S_prev[:, cs:ce]                # (T', Nc)
            yc       = Y[:, c].reshape(-1, 1)          # (T', 1)

            # A = Wout_c^T · Wout_c + inv_lambda · I
            A     = wout_c.T @ wout_c + inv_lambda * np.eye(Nc)
            A_inv = np.linalg.inv(A)                   # (Nc, Nc)

            # x_c*(t) = A^{-1} · (Wout_c^T · y_c(t) + inv_lambda · x_c(t-1))
            x_c_star = (yc @ wout_c + inv_lambda * x_c_prev) @ A_inv.T
            # x_c_star : (T', Nc)

            # ── Step 2: invert leaky update, mask saturated timesteps
            tanh_a_star = (x_c_star - (1.0 - alpha) * x_c_prev) / alpha
            # (T', Nc)

            valid_mask = np.all(np.abs(tanh_a_star) < clip_val, axis=1)
            if valid_mask.sum() < 10:
                continue   # too few valid timesteps — skip cluster

            a_c_star = np.arctanh(tanh_a_star[valid_mask])      # (M, Nc)

            # ── Step 3: subtract fixed intra + input contributions ───
            W_intra       = self.W[cs:ce, cs:ce]                # (Nc, Nc)
            intra_contrib = x_c_prev[valid_mask] @ W_intra.T    # (M, Nc)

            win_c         = self.Win[cs:ce, c:c+1]              # (Nc, 1)
            input_contrib = U[valid_mask, c:c+1] @ win_c.T      # (M, Nc)

            residual = a_c_star - intra_contrib - input_contrib  # (M, Nc)

            # ── Step 4: solve only weights on correlation-selected edges ─
            valid_states = S_prev[valid_mask]
            for neuron_offset in range(Nc):
                row = cs + neuron_offset
                connected = self.inter_cluster_mask_[row].copy()
                connected[cs:ce] = False
                source_indices = np.flatnonzero(connected)
                if source_indices.size == 0:
                    continue

                X_inter = valid_states[:, source_indices]
                I = np.eye(source_indices.size)
                weights = residual[:, neuron_offset] @ X_inter @ np.linalg.inv(
                    X_inter.T @ X_inter + winter_lambda * I
                )
                self.W[row, source_indices] = weights

    def _renormalise_spectral_radius(self):
        rho = max(abs(np.linalg.eigvals(self.W)))
        if rho > 0:
            self.W *= self.spectral_radius / rho

    # ------------------------------------------------------------------ #
    #  Coordinate-descent training                                        #
    # ------------------------------------------------------------------ #
    def fit_ridge_winter(
        self,
        inputs,
        targets,
        ridge_lambda=1e-8,
        winter_lambda=1e-4,
        inv_lambda=1e-1,
        clip_val=0.95,
        washout=100,
        n_epochs=10,
        verbose=True,
        correlation_data=None,
        correlation_method="spearman_diff",
    ):
        """
        Alternates between:
          Stage 1 : per-cluster ridge on Wout  (closed form)
          Stage 2 : per-cluster ridge on Winter (closed form)

        No gradients, no learning rate.

        Parameters
        ----------
        inputs        : (T, n_inputs)
        targets       : (T, n_clusters)
        ridge_lambda  : float   Wout ridge regularisation
        winter_lambda : float   Winter ridge regularisation
        inv_lambda    : float   x_c* inversion regularisation (default 0.1)
        clip_val      : float   arctanh saturation threshold (default 0.95)
        washout       : int
        n_epochs      : int
        verbose       : bool
        correlation_data : (T, n_inputs), sequence of arrays, or
                           (sequences, T, n_inputs); defaults to inputs
        correlation_method : 'spearman_diff' (default) or 'pearson'
        """
        inputs  = np.array(inputs)
        targets = np.array(targets)

        if inputs.ndim != 2 or inputs.shape[1] != self.n_inputs:
            raise ValueError("inputs must have shape (timesteps, n_inputs)")
        if self.n_clusters != self.n_inputs:
            raise ValueError(
                "correlation-guided CGDT-CESN requires n_clusters == n_inputs"
            )
        if targets.ndim != 2 or targets.shape != (len(inputs), self.n_clusters):
            raise ValueError(
                "targets must have shape (timesteps, n_clusters) and align with inputs"
            )

        if correlation_data is None:
            correlation_data = inputs
        elif isinstance(correlation_data, np.ndarray) and correlation_data.ndim == 3:
            correlation_data = [seq for seq in correlation_data]

        var_names = [f"var_{i}" for i in range(self.n_inputs)]
        guided = compute_and_apply_pout(
            correlation_data,
            var_names,
            self.W,
            self.p_in,
            self.n_clusters,
            self.neurons_per_cluster,
            self.spectral_radius,
            method=correlation_method,
        )
        self.W = guided["W"]
        self.correlation_matrix_ = guided["C"]
        self.P_out_matrix_ = guided["P_out"]
        self.p_out_max_ = guided["p_out_max"]
        if hasattr(self, "inter_cluster_mask_"):
            del self.inter_cluster_mask_
        self.inter_cluster_mask_ = self._build_inter_cluster_mask() & (self.W != 0)

        self.loss_history_  = []
        self.nrmse_history_ = []

        if verbose:
            print(f"CGDT-CESN ER RidgeWinter  |  epochs={n_epochs}  "
                  f"ridge={ridge_lambda}  winter={winter_lambda}  "
                  f"inv_lambda={inv_lambda}  clip={clip_val}")
            print(f"{'Epoch':>6}  {'MSE Loss':>12}  {'Train NRMSE':>13}")
            print("─" * 38)

        for epoch in range(1, n_epochs + 1):

            # Stage 1 — collect states
            states = self._collect_states(inputs, washout=0)

            # Stage 2 — ridge on Wout
            self._ridge_wout(states, targets, ridge_lambda, washout)

            # Monitor
            Nc    = self.neurons_per_cluster
            S_eff = states[washout:]
            Y_eff = targets[washout:]
            preds = np.column_stack([
                (self.Wout_clusters[c] @
                 S_eff[:, c*Nc:(c+1)*Nc].T).ravel()
                for c in range(self.n_clusters)
            ])
            mse_loss   = float(np.mean((preds - Y_eff) ** 2))
            nrmse_list = []
            for c in range(self.n_clusters):
                yc    = Y_eff[:, c]
                denom = yc.max() - yc.min()
                if denom > 0:
                    nrmse_list.append(
                        np.sqrt(np.mean((preds[:, c] - yc) ** 2)) / denom
                    )
            avg_nrmse = float(np.mean(nrmse_list)) if nrmse_list else np.nan

            self.loss_history_.append(mse_loss)
            self.nrmse_history_.append(avg_nrmse)

            if verbose:
                print(f"{epoch:>6}  {mse_loss:>12.6f}  {avg_nrmse:>13.6f}")

            # Stage 3 — ridge on Winter (skip on final epoch)
            if epoch < n_epochs:
                self._ridge_winter(
                    states, inputs, targets,
                    washout, winter_lambda, inv_lambda, clip_val
                )
                self._renormalise_spectral_radius()

        if verbose:
            print("─" * 38)
            print(f"Done. Final train NRMSE: {self.nrmse_history_[-1]:.6f}")


# Descriptive compatibility alias.
CESN_ER_RidgeWinter = CGDT_CESN
