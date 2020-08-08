import numpy as np

import torch

DEFAULT_SEED = 1

def main():
    seed = DEFAULT_SEED

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Set up the dataset loader
    device = torch.device('cuda')
