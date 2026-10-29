import numpy as np
from scipy.linalg import pinv


# CESN - Erdős-Rényi intra-cluster connections
class EchoStateNetworkWithErdosRenyiClusters:
    def __init__(self, n_inputs, neurons_per_cluster, n_clusters=None, spectral_radius=0.9, leak_rate=1.0,
                 input_scaling=1.0, p_in=0.1, p_out=0.01, input_connectivity=0.1, seed=None):
        np.random.seed(seed)
        if n_clusters is None:
            n_clusters = n_inputs

        self.n_inputs = n_inputs
        self.neurons_per_cluster = neurons_per_cluster
        self.n_clusters = n_clusters
        self.n_reservoir = n_clusters * neurons_per_cluster
        self.n_outputs = n_inputs  # Each cluster is responsible for one output

        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.p_in = p_in
        self.p_out = p_out

        # Initialize input weights and recurrent weights
        self.Win = np.zeros((self.n_reservoir, self.n_inputs))
        self.W = np.zeros((self.n_reservoir, self.n_reservoir))
        self.structure_matrix = np.zeros_like(self.W)  # Matrix to store the structure of clusters

        self._initialize_weights(input_connectivity, spectral_radius, input_scaling)

        # Instead of a single readout, we create one readout per cluster.
        # Each readout maps from neurons_per_cluster to 1 output.
        self.Wout_clusters = [np.zeros((1, self.neurons_per_cluster)) for _ in range(self.n_clusters)]
        
        self.reservoir_state = np.zeros((self.n_reservoir,))

    def _initialize_weights(self, input_connectivity, spectral_radius, input_scaling):
        # Set up input weights to be cluster-specific
        for cluster_idx in range(self.n_clusters):
            cluster_start = cluster_idx * self.neurons_per_cluster
            cluster_end = (cluster_idx + 1) * self.neurons_per_cluster

            # Each input i (cluster_idx) connects only to its designated cluster neurons.
            self.Win[cluster_start:cluster_end, cluster_idx] = ((np.random.rand(self.neurons_per_cluster) - 0.5) * 2 
                                                                 * input_scaling)
            mask_input = np.random.rand(self.neurons_per_cluster) < input_connectivity
            self.Win[cluster_start:cluster_end, cluster_idx] = np.multiply(self.Win[cluster_start:cluster_end, cluster_idx],
                                                                            mask_input)

            # Create intra-cluster connections (Erdős-Rényi) for neurons within the same cluster.
            for i in range(cluster_start, cluster_end):
                for j in range(cluster_start, cluster_end):
                    if i != j and np.random.rand() < self.p_in:
                        self.W[i, j] = (np.random.rand() - 0.5) * 2
                        self.structure_matrix[i, j] = 1  # Mark as intra-cluster connection

        # Create inter-cluster connections in range [-1, 1]
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                cluster_i_start = i * self.neurons_per_cluster
                cluster_i_end = (i + 1) * self.neurons_per_cluster
                cluster_j_start = j * self.neurons_per_cluster
                cluster_j_end = (j + 1) * self.neurons_per_cluster

                for ni in range(cluster_i_start, cluster_i_end):
                    for nj in range(cluster_j_start, cluster_j_end):
                        if np.random.rand() < self.p_out:
                            self.W[ni, nj] = (np.random.rand() - 0.5) * 2
                            self.W[nj, ni] = (np.random.rand() - 0.5) * 2
                            self.structure_matrix[ni, nj] = 2  # Mark as inter-cluster connection
                            self.structure_matrix[nj, ni] = 2

        # Normalize recurrent weight matrix to have the specified spectral radius
        rhoW = max(abs(np.linalg.eigvals(self.W)))
        self.W *= spectral_radius / rhoW


    def _update_reservoir(self, input_signal):
        input_signal = input_signal.reshape(-1)  # Ensure input_signal is 1D
        pre_activation = np.dot(self.Win, input_signal) + np.dot(self.W, self.reservoir_state)
        self.reservoir_state = (1 - self.leak_rate) * self.reservoir_state + self.leak_rate * np.tanh(pre_activation)

    def reset_reservoir(self):
        self.reservoir_state = np.zeros((self.n_reservoir,))

    def fit(self, inputs, outputs, ridge_lambda=0.0, reset=False,washout=100):
        if reset:
            self.reset_reservoir()
        
        n_steps = len(inputs)
        states = np.zeros((n_steps, self.n_reservoir))
        for t, input_signal in enumerate(inputs):
            self._update_reservoir(input_signal)
            states[t] = self.reservoir_state

        states = states[washout:]
        outputs = outputs[washout:]

        # For each cluster, train the readout using only the states from that cluster.
        for cluster_idx in range(self.n_clusters):
            cluster_start = cluster_idx * self.neurons_per_cluster
            cluster_end = (cluster_idx + 1) * self.neurons_per_cluster
            cluster_states = states[:, cluster_start:cluster_end]  # (n_steps, neurons_per_cluster)
            target = outputs[:, cluster_idx].reshape(-1, 1)  # (n_steps, 1) for this cluster

            if ridge_lambda > 0:
                I = np.identity(self.neurons_per_cluster)
                self.Wout_clusters[cluster_idx] = np.dot(
                    target.T,
                    np.dot(cluster_states, np.linalg.inv(np.dot(cluster_states.T, cluster_states) + ridge_lambda * I))
                )
            else:
                self.Wout_clusters[cluster_idx] = np.dot(target.T, pinv(cluster_states.T))

    def predict(self, inputs, reset=False,washout=100):
        """
        Predict outputs given input sequences using separate readouts for each cluster.
        
        Parameters:
        - inputs: array-like, shape (timesteps, input_dim)
        - reset: bool, if True, reset the reservoir state before prediction.
        
        Returns:
        - predictions: array, shape (timesteps, output_dim)
        """
        if reset:
            self.reset_reservoir()
        
        n_steps = len(inputs)
        predictions = np.zeros((n_steps, self.n_outputs))
        for t, input_signal in enumerate(inputs):
            self._update_reservoir(input_signal)
            for cluster_idx in range(self.n_clusters):
                cluster_start = cluster_idx * self.neurons_per_cluster
                cluster_end = (cluster_idx + 1) * self.neurons_per_cluster
                cluster_state = self.reservoir_state[cluster_start:cluster_end]
                # Extract the scalar value from the dot product
                pred_value = np.dot(self.Wout_clusters[cluster_idx], cluster_state).item()
                predictions[t, cluster_idx] = pred_value
            
        return predictions[washout:]
