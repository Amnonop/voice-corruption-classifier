import time

import numpy as np

import torch
from torch.optim import Adam, SGD, Optimizer
from torch.nn import Module as TorchModule

import higher

from dataset_n_shot import AudioNShot
from commons import SAMPLE_SIZE
from siamese import SiameseM5

DEFAULT_SEED = 1
NUM_EPOCHS = 100
INNER_ITERATIONS = 5

def train(dataset: AudioNShot, net: TorchModule, device, meta_optimizer: Optimizer, epoch_num, log):
    net.train()
    iterations = dataset.x_train.shape[0] # batch size

    for batch_index in range(iterations):
        start_time = time.time()

        # Get support and query sets
        x_support, y_support, x_query, y_query = dataset.next()
        task_num, set_size, c, sample_size = x_support.size()
        query_size = x_query.size(1)

        # Set inner optimizer
        inner_itteration = INNER_ITERATIONS
        optimizer = SGD(net.parameters(), lr=1e-1)

        query_losses = []
        query_accuracies = []
        meta_optimizer.zero_grad()
        for i in range(task_num):
            with higher.innerloop_ctx(net, optimizer, copy_initial_weights=False) as (fnet, diffopt):
                for _ in range(inner_itteration):
                    outputs = fnet(x_support[i])
 

def main():
    seed = DEFAULT_SEED
    batch_size = 32
    n_way = 5
    k_shot = 5
    k_query = 15

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Set up the dataset loader
    device = torch.device('cuda')
    dataset = AudioNShot('./data', batch_size, n_way, k_shot, k_query, SAMPLE_SIZE, device)

    net = SiameseM5()

    meta_optimizer = Adam(net.parameters(), lr=1e-3)

    log = []
    for epoch in range(NUM_EPOCHS):
        train(dataset, net, device, meta_optimizer, epoch, log)
