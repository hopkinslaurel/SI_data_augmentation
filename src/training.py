import numpy as np
from time import time
import pandas
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import v2
import torch.nn.functional as F
import torch.nn as nn
from src.datasets import satellite_dataloader
import sys
from time import time


# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def train_model(model, cuda, dataloader, optimizer, epoch, criterion, num_classes, model_type, print_every=100,
                mixing_method=None, cutmix_alpha=1.0, cutmix_num_pairs=1):
    """
    Trains a model for one epoch using the provided dataloader.
    """
    model.train()
    t0 = time()
    sum_loss = 0
    n_train, n_batches = len(dataloader.dataset), len(dataloader)
    print_sum_loss = 0
    idx = 0

    if mixing_method is not None:
        regression = model_type == 'regression'
        if mixing_method == 'CutMix':
            cutmix = v2.CutMix(num_classes=num_classes)
        elif mixing_method == 'Sat-CutMix':
            cutmix = v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha, num_pairs=cutmix_num_pairs, regression=regression)
        elif mixing_method == 'Sat-SlideMix':
            cutmix = v2.SlideMix(num_classes=num_classes, alpha=cutmix_alpha, num_pairs=cutmix_num_pairs, regression=regression)

    for img, label, _ in dataloader:
        if cuda:
            img = img.cuda()
            label = label.cuda()

        if mixing_method is not None:
            label_pre = label
            img, label = cutmix(img, label_pre)

        optimizer.zero_grad()

        outputs = torch.squeeze(model(img)).to(torch.float64)
        if num_classes == 1:
            outputs = outputs.double()
            label = label.double()

        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

        if (idx + 1) * dataloader.batch_size % print_every == 0:
            print_avg_loss = (sum_loss - print_sum_loss) / (
                print_every / dataloader.batch_size)
            print('Epoch {}: [{}/{} ({:0.0f}%)], Avg loss: {:0.4f}'.format(
                epoch, (idx + 1) * dataloader.batch_size, n_train,
                100 * (idx + 1) / n_batches, print_avg_loss))
            print_sum_loss = sum_loss
        idx += 1
    avg_loss = sum_loss / n_batches
    print('\nTrain Epoch {}: Loss {:0.4f}, Time {:0.3f}s'.format(epoch, avg_loss, time()-t0))
    return avg_loss


def validate_model(model, cuda, dataloader, epoch, criterion, num_classes, timeit=False):
    """
    Validates model using the provided dataloader.
    """

    with torch.no_grad():   # added per https://discuss.pytorch.org/t/out-of-memory-error-during-evaluation-but-training-works-fine/12274/4
        model.eval()
        t0 = time()
        sum_loss = 0
        n_train, n_batches = len(dataloader.dataset), len(dataloader)

        for img, label, _ in dataloader:
            if cuda:
                img = img.cuda()
                label = label.cuda()

            if timeit:
                t_start = time()
                outputs = torch.squeeze(model(img)).to(torch.float64)
                t_end = time()
            else:
                outputs = torch.squeeze(model(img)).to(torch.float64)
            if num_classes == 1:
                outputs = outputs.double()
                label = label.double()

            loss = criterion(outputs, label)

            sum_loss += loss.item()
        avg_loss = sum_loss / n_batches
        print('Test Epoch {}: Loss {:0.4f}, Time {:0.3f}s'.format(epoch, avg_loss, time()-t0))
    if timeit:
        return avg_loss, t_end-t_start
    else:
        return avg_loss
