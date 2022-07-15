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
from graph_text_model import CCAModel, GraphModel, ProjTaskModel, TextModel
from text_utils.text import *
from gnnfree.utils.graph import construct_graph_from_edges
from datasets.dataset import NodeTextDataset, NodeTextLabelDataset
from learner import *
from gnnfree.managers.trainer import *
from gnnfree.managers.manager import Manager
from gnnfree.nn.loss import *
from gnnfree.utils.evaluators import *
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
    parser.add_argument("--finetune_lr", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--pre_num_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--task_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--learning_task", type=str, default='cls_auc')

    parser.add_argument("--word_emb_dim", type=int, default=100)

    parser.add_argument("--icdver", type=int, default=10)
    parser.add_argument("--gpuid", type=int, default=0)

    parser.add_argument("--finetune_method", type=str, default='full')
    parser.add_argument("--label", type=str, default='postop_del')

    torch.manual_seed(115)
    random.seed(115)
    np.random.seed(115)

    params = parser.parse_args()

    if torch.cuda.is_available() and params.gpuid!=-1:
        params.device = torch.device('cuda:0')
    else:
        params.device = torch.device('cpu')

    prefix = params.label +'_'+ params.finetune_method+'_'+ 'cca_'
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

    texts_ind = value2index(texts, w2i, max_ft_len, padding_index)
    codes_ind = value2index(codes, e2n, max_codes_len)

    # print(code_mat)
    # train_data, valid = data_split(code_mat, 0.1)
    # graph_train, reg_train = data_split(train_data, 0.4)

    # train = graph_train

    train, test, valid = split_multiple_set([codes_ind, texts_ind, labels], [0.8, 0.1, 0.1])

    links = np.array(links)
    g = construct_graph_from_edges(links.T[0], links.T[2], num_nodes, inverse_edge=True, edge_type=links.T[1])
    g.ndata['feat'] = torch.ones([g.num_nodes(), 1], dtype=torch.float32)
    params.num_nodes = g.num_nodes()

    train_set = NodeTextLabelDataset(train, g)
    valid_set = NodeTextLabelDataset(valid, g)
    test_set = NodeTextLabelDataset(test, g)

    graph_model = GraphModel(params)
    text_model = TextModel(params)

    cca_model = CCAModel(text_model, graph_model, params).to(params.device)

    train_learner = GraphTextCCALearner('cca', train_set, cca_model, IDLoss(), Adam, params.batch_size)
    train_learner.setup_optimizer([{'lr':params.lr, 'weight_decay':params.l2}])

    val_learner = GraphTextCCALearner('cca', valid_set, cca_model, None, None, params.batch_size)

    trainer = Trainer(params)

    manager = Manager('cbow_model_cca')

    eval_metric = 'loss'

    manager.train([train_learner, [val_learner]], trainer, LossEvaluator('CCA_eval'), eval_metric, device=params.device, num_epochs=params.pre_num_epochs)

    train_learner.load_model('cbow_model_cca'+"_best.pth")

    graph_out_learner = GraphForwardLearner('g_for', train_set, graph_model, None, None, params.batch_size)
    text_out_learner = ForwardLearner('t_for', train_set, text_model, None, None, params.batch_size)
    g_res = manager.eval(graph_out_learner, trainer, CollectionEvaluator('col'), device=params.device)
    t_res = manager.eval(text_out_learner, trainer, CollectionEvaluator('col'), device=params.device)
    with torch.no_grad():
        corr, U, V = cca_model.corr_eval(t_res['res_col'], g_res['res_col'])
        print('corr captured:', corr)

    valid_task_ind = np.logical_not(np.isnan(labels))
    valid_task_ind = valid_task_ind.nonzero()[0]
    perm = np.random.permutation(len(valid_task_ind))
    res_col = []
    for i in range(10):
        selected_ind = perm[i*int(len(perm)/10):(i+1)*int(len(perm)/10)]
        task_ind = valid_task_ind[selected_ind]
        task_texts = [texts_ind[i] for i in task_ind]
        task_codes = [codes_ind[i] for i in task_ind]
        task_labels = labels[task_ind]

        task_train_set, task_valid_set, task_test_set = split_multiple_set([task_codes, task_texts, task_labels], [0.8,0.1,0.1])

        task_train_set = NodeTextLabelDataset(task_train_set, g)
        task_valid_set = NodeTextLabelDataset(task_valid_set, g)
        task_test_set = NodeTextLabelDataset(task_test_set, g)

        if params.learning_task == "cls_auc":
            task_evlter = BinaryHNEvaluator('auc_eval')
            params.task_dim = 1
            trainer = MaxTrainer(params)
            eval_metric = 'auc'
            task_loss = BinaryLoss()
        else:
            task_evlter = MSEEvaluator('mse_eval')
            params.task_dim = 1
            trainer = Trainer(params)
            eval_metric = 'mse'
            task_loss = MSELoss()
        text_pred_model = ProjTaskModel(text_model, U, params).to(params.device)

        if params.finetune_method == 'partial':
            task_train_learner = MLPForwardLearner('t_task', task_train_set, text_pred_model, task_loss, Adam, params.task_batch_size)
        else:
            task_train_learner = ForwardLearner('t_task', task_train_set, text_pred_model, task_loss, Adam, params.task_batch_size)
        task_train_learner.setup_optimizer([{'lr':params.lr, 'weight_decay':params.l2}])

        task_test_learner = ForwardLearner('t_task', task_test_set, text_pred_model, None, None, params.task_batch_size)
        task_val_learner = ForwardLearner('t_task', task_valid_set, text_pred_model, None, None, params.task_batch_size)
        
        manager = Manager(prefix+'cbow_model_task')
        manager.train([task_train_learner, [task_val_learner]], trainer, task_evlter, eval_metric, device=params.device, num_epochs=params.num_epochs)

        # finetune_train_learner = ForwardLearner('finetune_task', task_train_set, text_pred_model, task_loss, Adam, params.task_batch_size)
        # finetune_train_learner.setup_optimizer([{'lr':params.finetune_lr, 'weight_decay':params.l2}])

        # manager.train([finetune_train_learner, [task_val_learner]], trainer, task_evlter, eval_metric, device=params.device, num_epochs=params.num_epochs)

        task_test_learner.load_model(prefix+'cbow_model_task'+"_best.pth", params.device)

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
