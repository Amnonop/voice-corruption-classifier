import time

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import torch
from torch.optim import Adam, SGD, Optimizer
from torch.nn import Module as TorchModule
import torch.nn.functional as F

import higher

from dataset_n_shot import AudioNShot
from commons import SAMPLE_SIZE
from siamese import SiameseM5
from m5 import M5

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
                    support_outputs = fnet(x_support[i])
                    support_loss = F.cross_entropy(support_outputs, y_support[i])
                    diffopt.step(support_loss)

                query_outputs = fnet(x_query[i])
                query_loss = F.cross_entropy(query_outputs, y_query[i])
                query_losses.append(query_loss.detach())
                query_accuracy = (query_outputs.argmax(dim=1) == y_query[i]).sum().item() / query_size
                query_accuracies.append(query_accuracy)

                query_loss.backward()

        meta_optimizer.step()
        query_losses = sum(query_losses) / task_num
        query_accuracies = 100. * sum(query_accuracies) / task_num
        i = epoch_num + float(batch_index) / iterations
        iterration_time = time.time() - start_time
        if batch_index % 4 == 0:
            print(f'[Epoch {i:.2f}] Train Loss: {query_losses:.2f} | Acc: {query_accuracies:.2f} | Time: {iterration_time:.2f}')

        log.append({
            'epoch': i,
            'loss': query_losses,
            'acc': query_accuracies,
            'mode': 'train',
            'time': time.time(),
        })

def test(dataset: AudioNShot, net: TorchModule, device, epoch_num: int, log: list):
    net.train()
    test_itterations = dataset.x_test.shape[0]

    query_losses = []
    query_accuracies = []

    for batch_index in range(test_itterations):
        x_support, y_support, x_query, y_query = dataset.next('test')
        task_num, set_size, c, sample_size = x_support.size()
        query_size = x_query.size(1)

        inner_itterations = INNER_ITERATIONS
        optimizer = SGD(net.parameters(), lr=1e-1)

        for i in range(task_num):
            with higher.innerloop_ctx(net, optimizer, track_higher_grads=False) as (fnet, diffopt):
                for _ in range(inner_itteration):
                    support_outputs = fnet(x_support[i])
                    support_loss = F.cross_entropy(support_outputs, y_support[i])
                    diffopt.step(support_loss)

                query_outputs = fnet(x_query[i]).detach()
                query_loss = F.cross_entropy(query_outputs, y_query[i], reduction='none')
                query_losses.append(query_loss.detach())
                query_accuracy = (query_outputs.argmax(dim=1) == y_query[i]).detach()
                query_accuracies.append(query_accuracy)

    query_losses = torch.cat(query_losses).mean().item()
    query_accuracies = 100. * torch.cat(query_accuracies).float().mean().item()
    print(f'[Epoch {epoch_num + 1:.2f}] Test Loss: {query_losses:.2f} | Acc: {query_accuracies:.2f}')
    log.append({
        'epoch': epoch_num + 1,
        'loss': query_losses,
        'acc': query_accuracies,
        'mode': 'test',
        'time': time.time(),
    })

def plot(log):
    # Generally you should pull your plotting code out of your training
    # script but we are doing it here for brevity.
    df = pd.DataFrame(log)

    fig, ax = plt.subplots(figsize=(6, 4))
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(70, 100)
    fig.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    fname = 'maml-accs.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)
 

def main():
    seed = DEFAULT_SEED
    batch_size = 32
    n_way = 5
    k_shot = 5
    k_query = 5

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Set up the dataset loader
    # device = torch.device('cuda')
    dataset = AudioNShot('./data', batch_size, n_way, k_shot, k_query, SAMPLE_SIZE)

    # net = SiameseM5()
    net = M5(num_classes=n_way)

    meta_optimizer = Adam(net.parameters(), lr=1e-3)

    log = []
    for epoch in range(NUM_EPOCHS):
        train(dataset, net, None, meta_optimizer, epoch, log)
        test(dataset, net, None, epoch, log)
        plot(log)

if __name__ == "__main__":
    main()
