from pathlib import Path
import uuid
import time

import torch
from torch.optim import Adam
from torch.nn.functional import pairwise_distance

from commons import *
from configuration import Configuration
from utils import create_results_directories
from dataset_transforms import TransformsComposer, ToTensor, Rescale
from siamese_loader import SiameseLoader
from siamese import Siamese
from contrastive_loss import ContrastiveLoss

DATA_DIR = './data'

def main():
    config_filename = Path.cwd().joinpath(CONFIGS_DIR).joinpath(CONFIG_FILENAME)
    config = Configuration(config_filename)

    data_dir_path = Path.cwd().joinpath(DATA_DIR)

    results_dir_path = Path.cwd().joinpath('./siamese_results')
    current_run_path = create_results_directories(results_dir_path)

    transforms = TransformsComposer([Rescale(output_size=SAMPLE_SIZE), ToTensor()])

    # Load data
    data_loader = SiameseLoader(data_dir_path, transforms)

    batch = data_loader.get_batch(4)
    one_shot = data_loader.make_oneshot_task(20)

    model = Siamese()
    criterion = ContrastiveLoss(margin=0.05)
    optimizer = Adam(model.parameters(), lr=0.0001)

    evaluate_every = 10  # interval for evaluating on one-shot tasks
    loss_every = 20  # interval for printing loss (iterations)
    batch_size = 32
    num_iterations = 50
    N_way = 20  # how many classes for testing one-shot tasks>
    n_val = 250  # how many one-shot tasks to validate on?
    best = -1

    state_filename = f'{uuid.uuid1()}_state_{num_iterations}_iters.pth'
    state_path = current_run_path.joinpath('best_snapshot').joinpath(state_filename)

    loss_history = []
    loss_iterations = []

    print("Starting training process!")
    print("-------------------------------------")
    t_start = time.time()
    for i in range(1, num_iterations):
        (inputs, targets) = data_loader.get_batch(batch_size)

        # switch model to training mode
        model.train()

        # clear gradient accumulators
        optimizer.zero_grad()

        # forward pass
        out1, out2 = model(inputs[0], inputs[1])

        # calculate loss of the network output with respect to the training labels
        loss = criterion(out1, out2, targets)

        # backpropagate and update optimizer learning rate
        loss.backward()
        optimizer.step()

        print("\n ------------- \n")
        print("Loss: {0}".format(loss))

        if i % evaluate_every == 0:
            print("Time for {0} iterations: {1}".format(i, time.time() - t_start))
            val_acc = test_oneshot(model, data_loader, N_way, n_val)
            if val_acc >= best:
                print("Current best: {0}, previous best: {1}".format(val_acc, best))
                print("Saving weights to: {0} \n".format(state_path))
                torch.save(model.state_dict(), state_path)
                best = val_acc

        if i % loss_every == 0:
            print("iteration {}, training loss: {:.2f},".format(i, loss.item()))
            loss_iterations.append(i)
            loss_history.append(loss.item())

    # weights_path_2 = os.path.join(data_path, "model_weights.h5")
    # model.load_weights(weights_path_2)

def test_oneshot(model, data_loader, N, k):
    number_correct = 0

    print(f'Evaluating model on {k} random {N} way one-shot learning tasks.')

    model.eval()
    with torch.no_grad():
        for i in range(k):
            inputs, targets = data_loader.make_oneshot_task(N)
            output1, output2 = model(inputs[0], inputs[1])

            euclidean_distance = pairwise_distance(output1, output2)
            print(f'Eval iter {i}: targets: {targets}')
            print(f'Eval iter {i}: distance: {euclidean_distance}')

            # Check the index of the minimal distance fits the index of the
            # pair that is the similar pair
            #true_index = torch.argmin(targets)
            #acc_figure = (euclidean_distance[true_index] - torch.min(euclidean_distance))/euclidean_distance[true_index]
            #print(f'acc_figure {acc_figure}')
            if torch.argmin(euclidean_distance) == torch.argmin(targets):
                number_correct += 1

    percent_correct = (100.0 * number_correct / k)
    print(f'Got an average of {percent_correct}% {N} way one-shot learning accuracy')
    return percent_correct


if __name__ == '__main__':
    main()
