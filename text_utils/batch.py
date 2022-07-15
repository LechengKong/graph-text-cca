import sys
import torch
import dgl
import numpy as np
import torch
import random
import time


class GTBatch:
    def __init__(self, samples) -> None:
        node_id, text, label = zip(*samples)
        self.ls = []
        self.ls.append(torch.tensor(np.stack(node_id),dtype=torch.long))
        self.ls.append(torch.tensor(np.stack(text),dtype=torch.long))
        self.ls.append(torch.tensor(np.array(label), dtype=torch.float))

    def to_device(self, device):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].to(device)

    def to_name(self):
        self.node_ind = self.ls[0]
        self.text = self.ls[1]
        self.labels = self.ls[2]

def collate_gt(samples):
    return GTBatch(samples)