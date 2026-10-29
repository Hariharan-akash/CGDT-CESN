import numpy as np
from scipy.linalg import pinv


class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, n_outputs, spectral_radius, leak_rate,
                 input_scaling, rc_connectivity, input_connectivity, seed):
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        
        # Initialize input weights with [-1, 1] range and scaling
        self.Win = (np.random.rand(n_reservoir, n_inputs) - 0.5) * 2 * input_scaling
        mask_input = np.random.rand(n_reservoir, n_inputs) < input_connectivity
        self.Win = np.multiply(self.Win, mask_input)  # Apply input connectivity mask

        # Initialize recurrent weights with [-1, 1] range and sparsity
        W = (np.random.rand(n_reservoir, n_reservoir) - 0.5) * 2
        mask_rc = np.random.rand(n_reservoir, n_reservoir) < rc_connectivity
        W = np.multiply(W, mask_rc)  # Apply recurrent connectivity mask
        
        rhoW = max(abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / rhoW)

        # Output weights 
        self.Wout = np.zeros((n_outputs, n_reservoir))
        
        # Reservoir state initialization
        self.reservoir_state = np.zeros((n_reservoir,))
        
    def _update_reservoir(self, input_signal):
        #computation of preactivation and updation of reservoir state
        pre_activation = np.dot(self.Win, input_signal) + np.dot(self.W, self.reservoir_state)
        self.reservoir_state = (1 - self.leak_rate) * self.reservoir_state + self.leak_rate * np.tanh(pre_activation)
        
    def reset_reservoir(self):
        #resetting reservoir state to zero
        self.reservoir_state = np.zeros((self.n_reservoir,))

    
    def fit(self, inputs, outputs, ridge_lambda=0.0, reset=False,washout = 100):     
        #Train the ESN on input-output pairs.
        if reset:
            self.reset_reservoir()  # Reset reservoir state if specified
        
        states = np.zeros((len(inputs), self.n_reservoir))
        for t, input_signal in enumerate(inputs):
            self._update_reservoir(input_signal)
            states[t] = self.reservoir_state

        states = states[washout:]
        outputs = outputs[washout:]

        # Train output weights using ridge regression
        if ridge_lambda > 0:
            I = np.identity(self.n_reservoir)
            self.Wout = np.dot(np.dot(outputs.T, states), 
                               np.linalg.inv(np.dot(states.T, states) + ridge_lambda * I))
        else:
            self.Wout = np.dot(outputs.T, pinv(states.T))

    def predict(self, inputs, reset=False,washout=100):
        """
        Predict outputs given input sequences.

        Parameters:
        - inputs: array-like, shape (timesteps, input_dim)
        - reset: bool, default False. If True, reset the reservoir state before prediction.

        Returns:
        - predictions: array, shape (timesteps, output_dim)
        """
        if reset:
            self.reset_reservoir()  # Reset reservoir state if specified
        
        predictions = np.zeros((len(inputs), self.n_outputs))
        for t, input_signal in enumerate(inputs):
            self._update_reservoir(input_signal)
            predictions[t] = np.dot(self.Wout, self.reservoir_state)
        return predictions[washout:]
