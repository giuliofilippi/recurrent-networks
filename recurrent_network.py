# imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# network class
class Recurrent_Network(nn.Module):
    def __init__(self, N_PN, N_KC):
        super().__init__()