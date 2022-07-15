import torch
import torch.nn as nn
from torch.optim import Adam
import pickle as pkl
import numpy as np
import dgl
import time
import os.path as osp
import argparse
import random
import pandas as pd
from models.graph_text_model import CCAModel, EmbTextModel, GraphModel, ProjTaskModel, TaskModel, TextModel
from text_utils.text import *
from utils.graph import construct_reverse_graph_from_edges
from datasets.dataset import NodeTextDataset, NodeTextLabelDataset
from managers.learner import *
from managers.trainer import *
from managers.manager import Manager
from nn.loss import *
import scipy.stats as st

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='graph-text')

    parser.add_argument("--data_path", type=str, default="/project/tantra/jerry.kong/med_data")
    parser.add_argument("--model_path", type=str, default="/project/tantra/jerry.kong/med_model")
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--num_bases", type=int, default=4)
    parser.add_argument("--attn_rel_emb_dim", type=int, default=64)
    parser.add_argument("--rel_emb_dim", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--node_emb_dim", type=int, default=100)
    parser.add_argument("--num_gcn_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--edge_dropout", type=float, default=0)
    parser.add_argument("--gnn_agg_type", type=str, default='sum')
    parser.add_argument("--JK", type=str, default='last')
    parser.add_argument("--corr_dim", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--pre_num_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--task_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--learning_task", type=str, default='cls_auc')

    parser.add_argument("--word_emb_dim", type=int, default=100)

    parser.add_argument("--icdver", type=int, default=10)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--label", type=str, default='postop_del')

    torch.manual_seed(115)
    random.seed(115)
    np.random.seed(115)

    params = parser.parse_args()

    if torch.cuda.is_available() and params.gpuid!=-1:
        params.device = torch.device('cuda:0')
    else:
        params.device = torch.device('cpu')


    prefix = params.label +'_'+ 'cca_'
    print(prefix)


    data = open_and_load_pickle(osp.join(params.data_path, 'epic_data.pkl'))
    [w2i, i2w] = open_and_load_pickle(osp.join(params.data_path, 'epic-word-index-map-10.pkl'))
    [e2n,n2e] = open_and_load_pickle(osp.join(params.data_path,'epic-icd-10.pkl'))
    links = open_and_load_pickle(osp.join(params.data_path,'graphepicicd-10.pkl'))
    params.num_rels = 12

    params.inp_dim = params.node_emb_dim
    params.aug_num_rels = params.num_rels*2
    params.model_file_name = "gt.pth"

    vocab_size, padding_index = process_vocab(w2i, i2w)
    params.padding_index = padding_index
    params.vocab_size = vocab_size
    print(padding_index)
    print('vocab size:',params.vocab_size)

    loi = params.label

    texts = data['AN_PROC_NAME']
    codes = data['codes']
    labels = data[loi].astype(float).values

    num_nodes = np.max(np.array(list(e2n.values())))+1

    max_ft_len = sequence_max_len(texts)

    max_codes_len = sequence_max_len(codes)

    print("max text len:", max_ft_len)
    print("max procedure code length", max_codes_len)
    links = np.array(links)
    g = construct_reverse_graph_from_edges(links.T, num_nodes, num_rel=params.num_rels)
    g.ndata['feat'] = torch.ones([g.num_nodes(), 1], dtype=torch.float32)
    params.num_nodes = g.num_nodes()

    texts_ind = value2index(texts, w2i, max_ft_len, padding_index)
    codes_ind = value2index(codes, e2n, max_codes_len)

    valid_task_ind = np.logical_not(np.isnan(labels))
    valid_task_ind = valid_task_ind.nonzero()[0]
    task_ind = valid_task_ind
    task_texts = np.array([texts_ind[i] for i in task_ind])
    task_codes = np.array([codes_ind[i] for i in task_ind])
    task_labels = labels[task_ind]
    data = [task_codes, task_texts, task_labels]
    fold = 10
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    folds = []
    for _, t_index in kfold.split(np.zeros_like(np.array(task_labels)), np.array(task_labels)):
        folds.append(t_index)
    res_col = []
    for i in range(10):
        test_arr = np.zeros(len(task_labels), dtype=bool)
        test_arr[folds[i]]=1
        val_arr = np.zeros(len(task_labels), dtype=bool)
        val_arr[folds[int((i+1)%fold)]]=1
        train_arr = np.logical_not(np.logical_or(test_arr, val_arr))
        train_ind = train_arr.nonzero()[0]
        test_ind = test_arr.nonzero()[0]
        val_ind = val_arr.nonzero()[0]
        train = [[v[i] for v in data] for i in train_ind]
        test = [[v[i] for v in data] for i in test_ind]
        valid = [[v[i] for v in data] for i in val_ind]

        train_set = NodeTextLabelDataset(train, g)
        valid_set = NodeTextLabelDataset(valid, g)
        test_set = NodeTextLabelDataset(test, g)

        # text_model = TextModel(params)
        text_model = EmbTextModel(params)

        if params.learning_task == "cls_auc":
            task_evlter = GCEvaluator('auc_eval')
            params.task_dim = 1
            trainer = MaxTrainer(params)
            eval_metric = 'auc'
            task_loss = GCBinaryLoss()
        else:
            task_evlter = MSEEvaluator('mse_eval')
            params.task_dim = 1
            trainer = Trainer(params)
            eval_metric = 'mse'
            task_loss = MSELoss()
        text_pred_model = TaskModel(text_model, params).to(params.device)

        task_train_learner = ForwardLearner('t_task', train_set, text_pred_model, task_loss, Adam, params.task_batch_size)
        # task_train_learner = MLPForwardLearner('mlp_task', train_set, text_pred_model, task_loss, Adam, params.task_batch_size)
        task_train_learner.setup_optimizer([{'lr':params.lr, 'weight_decay':params.l2}])

        task_test_learner = ForwardLearner('t_task', test_set, text_pred_model, None, None, params.task_batch_size)
        task_val_learner = ForwardLearner('t_task', valid_set, text_pred_model, None, None, params.task_batch_size)
        
        manager = Manager(prefix+'text_model_task')
        manager.train([task_train_learner, [task_val_learner]], trainer, task_evlter, eval_metric, device=params.device, num_epochs=params.num_epochs)

        task_test_learner.load_model(prefix+'text_model_task'+"_best.pth", params.device)

        res = manager.eval(task_test_learner, trainer, task_evlter, device=params.device)
        print(res)
        res_col.append(res)
    
    res_dict = {}

    for res in res_col:
        for k in res:
            if k not in res_dict:
                res_dict[k] = []
            res_dict[k].append(res[k])
    print(res_dict)
    for k in res_dict:
        metric = np.array(res_dict[k])
        interval = st.t.interval(0.95, len(metric)-1, loc=np.mean(metric), scale=st.sem(metric))
        print(k, np.mean(metric), interval)
