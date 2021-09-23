train_network_flag = True
test_network_flag = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # TensorBoard support

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models

import time
import pandas as pd
import json

# import modules to build RunBuilder and RunManager helper classes
from collections import OrderedDict
from collections import namedtuple
from itertools import product

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# import scikitplot as skplt
import numpy as np


torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

# Use standard FashionMNIST dataset
train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()  # ,
    ])
)

test_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

"""Network Class"""


# Build the neural network, expand on top of nn.Module
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

        # define forward function

    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # conv 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # fc1
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)

        return t


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# Read in the hyper-parameters and return a Run namedtuple containing all the
# combinations of hyper-parameters
class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


# helper function to calculate all predictions of train set
def get_all_preds(model, loader):
    model.eval()
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds),
            dim=0
        )
    model.train()
    return all_preds


def calc_accuracy(network, data_loader, data_set):
    with torch.no_grad():
        network.eval()
        preds = get_all_preds(network, data_loader)
        network.train()
    cm = confusion_matrix(data_set.targets, preds.argmax(dim=1))
    return np.trace(cm)


# Helper class, help track loss, accuracy, epoch time, run time,
# hyper-parameters etc. Also record to TensorBoard and write into csv, json
class RunManager():
    def __init__(self):
        # tracking every epoch count, loss, accuracy, time
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        # tracking every run count, run data, hyper-params used, time
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        # record model, loader and TensorBoard
        self.network = None
        self.loader = None
        self.tb = None

    # record the count, hyper-param, model, loader of each run
    # record sample images and network graph to TensorBoard
    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)

    # when run ends, close TensorBoard, zero epoch count
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    # zero epoch count, loss, accuracy,
    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    #
    def end_epoch(self):
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)
        # calculate epoch duration and run duration(accumulate)
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        # record epoch loss and accuracy
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)
        # Record epoch loss and accuracy to TensorBoard
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        # Write into 'results' (OrderedDict) for all run related data
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["loss"] = loss
        results["training_accuracy"] = accuracy
        results["accuracy_train"] = calc_accuracy(network, train_loader, train_set) / len(train_loader.dataset)
        results["accuracy_test"] = calc_accuracy(network, test_loader, test_set) / len(test_loader.dataset)
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration

        # Record hyper-params into 'results'
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        # calc_acc_on_test(network)
        # display epoch information and show progress
        clear_output(wait=True)
        display(df)

    # accumulate loss of batch into entire epoch loss
    def track_loss(self, loss):
        # multiply batch size so variety of batch sizes can be compared
        self.epoch_loss += loss.item() * self.loader.batch_size

    # accumulate number of corrects of batch into entire epoch num_correct
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    # save end results of all runs into csv, json for further a
    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns',
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)


# put all hyper params into a OrderedDict, easily expandable
params = OrderedDict(
    lr=[.001],
    batch_size=[250],
    shuffle=[True]
)
epochs = 30

if train_network_flag == True:

    m = RunManager()

    # get all runs from params using RunBuilder class
    for run in RunBuilder.get_runs(params):

        # if params changes, following line of code should reflect the changes too
        network = Network()
        loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size)
        optimizer = optim.Adam(network.parameters(), lr=run.lr)

        m.begin_run(run, network, loader)
        for epoch in range(epochs):
            loss_values = []
            running_loss = 0.0
            m.begin_epoch()
            for batch in loader:
                images = batch[0]
                labels = batch[1]
                preds = network(images)
                loss = F.cross_entropy(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                m.track_loss(loss)
                m.track_num_correct(preds, labels)

                running_loss += loss.item()
                loss_values.append(running_loss)

            m.end_epoch()
        m.end_run()
    df = pd.DataFrame.from_dict(m.run_data, orient='columns')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    df.plot(ax=ax1, x='epoch', y='accuracy_train', c='Red', label='train_accuracy')
    df.plot(ax=ax1, x='epoch', y='accuracy_test', c='DarkBlue', label='test_accuracy')
    ax1.legend()
    plt.show()
    # when all runs are done, save results to files
    m.save('results')
    model_save_name = 'ex1_037596236_301219879_unreg.pt'
    path = F"/content/drive/Shared drives/Deep Learning course/ex1_037596236_301219879/{model_save_name}"
    torch.save(network.state_dict(), path)
    torch.no_grad()

"""*Load* model from drive"""

if test_network_flag == True:
    import torch
    from torchvision import datasets, models, transforms

    model = Network()

    model_save_name = 'ex1_037596236_301219879_unreg.pt'
    path = F"/content/drive/Shared drives/Deep Learning course/ex1_037596236_301219879/{model_save_name}"

    prediction_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)
    model.load_state_dict(torch.load(path))
    test_preds = get_all_preds(model, prediction_loader)

cm = confusion_matrix(test_set.targets, test_preds.argmax(dim=1))
cm

skplt.metrics.plot_confusion_matrix(test_set.targets, test_preds.argmax(dim=1), normalize=True)
