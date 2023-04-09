import torch
import numpy as np
from torch.autograd import Variable
from torch.autograd.function import Function

def dice(outputs, labels):

    outputs, labels = outputs.float(), labels.float()

    outputs_flat = outputs.view(outputs.size(0), -1)
    labels_flat = labels.view(labels.size(0), -1)

    # intersect = torch.dot(outputs_flat, labels_flat)
    intersect = torch.sum(outputs_flat * labels_flat, dim=1)
    # union = torch.add(torch.sum(outputs_flat), torch.sum(labels_flat))
    union = torch.sum(outputs_flat, dim=1) + torch.sum(labels_flat, dim=1)
    dice = 1 - (2 * intersect + 1e-5) / (union + 1e-5)
    return dice