from gnnfree.utils.datasets import DatasetWithCollate

from text_utils.batch import *


class NodeTextDataset(DatasetWithCollate):
    def __init__(self, data, graph, label_dict, label='los'):
        self.data = data
        self.num_sample = len(data)
        self.g = graph
        self.label_ind = label_dict[label]

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        node_id, context, labels = self.data[index]
        out_label = labels[self.label_ind]
        return node_id, context, out_label

    def get_collate_fn(self):
        return collate_gt

class NodeTextLabelDataset(DatasetWithCollate):
    def __init__(self, data, graph):
        self.data = data
        self.num_sample = len(data)
        self.g = graph

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        node_id, context, labels = self.data[index]
        return node_id, context, labels

    def get_collate_fn(self):
        return collate_gt