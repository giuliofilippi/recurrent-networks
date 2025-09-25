# imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# network class
class Recurrent_Network(nn.Module):
    def __init__(self, n_PN, n_KC, W, R):
        super().__init__()
        self.n_PN = n_PN
        self.n_KC = n_KC
        self.W = nn.Parameter(W)
        self.R = nn.Parameter(R)

    def forward(self, x_t, h_prev=None):
        """
        x_t: input at current timestep [n_PN]
        h_prev: previous KC activity [n_KC]
        """
        if h_prev is None:
            h_prev = torch.zeros(self.n_KC, device=x_t.device)

        # PN → KC
        pn_input = torch.matmul(x_t, self.W)  # [n_KC]
        
        # KC recurrence
        h_new = pn_input + torch.matmul(h_prev, self.R)  # [n_KC]
        return h_new
    
    def apply_apl(self, forwarded_activations):
        """
        Apply APL global inhibition.
        forwarded_activations: [n_KC]
        """
        # Compute population activity (mean or sum, depending on desired effect)
        inhibition_signal = forwarded_activations.mean()

        # Subtract from all KCs
        inhibited = forwarded_activations - inhibition_signal

        # Optionally apply nonlinearity (ReLU ensures no negative KC activity)
        inhibited = torch.relu(inhibited)

        return inhibited

    def run_sequence(self, X_seq):
        """
        X_seq: input sequence of shape [timesteps, n_PN]
        Returns: KC activity over time [timesteps, n_KC]
        """
        timesteps = X_seq.shape[0]
        h = torch.zeros(self.n_KC)  # initial KC activity
        KC_seq = []

        for t in range(timesteps):
            x_t = X_seq[t]
            h = self.forward(x_t, h)
            h_inh = self.apply_apl(h)
            KC_seq.append(h_inh.unsqueeze(0))  # keep timestep dim

        return torch.cat(KC_seq, dim=0)