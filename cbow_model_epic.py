import itertools
import torch
import torch.nn as nn
from torch.optim import Adam
import pickle as pkl
import numpy as np
import os
import os.path as osp
import argparse
import random
import pandas as pd
from datetime import datetime
from gnnfree.utils.evaluators import BinaryHNEvaluator, CollectionEvaluator, LossEvaluator, MSEEvaluator

from gnnfree.utils.graph import construct_graph_from_edges
from gnnfree.managers.manager import Manager
from gnnfree.managers.trainer import *
from gnnfree.nn.loss import *
from gnnfree.utils.utils import *

from graph_text_model import CCAModel, GraphModel, ProjTaskModel, TextModel
from text_utils.text import *

from dataset import NodeTextDataset, NodeTextLabelDataset
from learner import *

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
    parser.add_argument("--graph_dropout", type=float, default=0)
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
    parser.add_argument("--task_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--learning_task", type=str, default='cls_auc')

    parser.add_argument("--word_emb_dim", type=int, default=100)

    parser.add_argument("--icdver", type=int, default=10)
    parser.add_argument("--gpuid", type=int, default=0)

    parser.add_argument("--finetune_method", type=str, default='full')
    parser.add_argument("--label", type=str, default='postop_del')
    parser.add_argument("--psearch", type=bool, default=False)

    torch.manual_seed(115)
    random.seed(115)
    np.random.seed(115)

    params = parser.parse_args()

    if not osp.exists('./saved_exp'):
        os.mkdir('./saved_exp')

    curtime = datetime.now()
    params.exp_dir = osp.join('./saved_exp', str(curtime))
    prefix = params.label+"_"+params.finetune_method+"_"+"cca_"
    os.mkdir(params.exp_dir)

    save_params(osp.join(params.exp_dir,'command'), params)
    params.cca_model_name = osp.join(params.exp_dir,'ccamatched_')
    params.task_model_name = osp.join(params.exp_dir,prefix)

    if torch.cuda.is_available() and params.gpuid!=-1:
        params.device = torch.device('cuda:0')
    else:
        params.device = torch.device('cpu')

    


    data = open_and_load_pickle(osp.join(params.data_path, 'epic_data.pkl'))
    [w2i, i2w] = open_and_load_pickle(osp.join(params.data_path, 'epic-word-index-map-10.pkl'))
    [e2n,n2e] = open_and_load_pickle(osp.join(params.data_path,'epic-icd-10.pkl'))
    links = open_and_load_pickle(osp.join(params.data_path,'graphepicicd-10.pkl'))
    params.num_rels = 12

    params.inp_dim = params.node_emb_dim
    params.aug_num_rels = params.num_rels*2

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

    data = [codes_ind, texts_ind, labels]

    # print(code_mat)
    # train_data, valid = data_split(code_mat, 0.1)
    # graph_train, reg_train = data_split(train_data, 0.4)

    # train = graph_train

    links = np.array(links)
    g = construct_graph_from_edges(links.T[0], links.T[2], num_nodes, True, links.T[1], params.num_rels)

    params.num_nodes = g.num_nodes()

    # valid_task_ind = np.logical_not(np.isnan(labels))
    # valid_task_ind = valid_task_ind.nonzero()[0]
    # task_ind = valid_task_ind
    # task_texts = np.array([texts_ind[i] for i in task_ind])
    # task_codes = np.array([codes_ind[i] for i in task_ind])
    # task_labels = labels[task_ind]
    # data = [task_codes, task_texts, task_labels]

    if params.learning_task == "cls_auc":
        task_evlter = BinaryHNEvaluator('auc_eval')
        params.task_dim = 1
        task_trainer = MaxTrainer(params)
        task_eval_metric = 'auc'
        task_loss = BinaryLoss()
    else:
        task_evlter = MSEEvaluator('mse_eval')
        params.task_dim = 1
        task_trainer = Trainer(params)
        task_eval_metric = 'mse'
        task_loss = MSELoss()

    def data_construct(data, label, index):
        dset = [[v[i] for v in data] for i in index]
        format_data = NodeTextLabelDataset(dset, g)
        return format_data

    def learn_one_fold(train,test,val):
        print('train size: {}, test size: {}, val size: {}'.format(len(train),len(test),len(val)))
        graph_model = GraphModel(params)
        text_model = TextModel(params)
        cca_model = CCAModel(text_model, graph_model, params).to(params.device)

        train_learner = GraphTextCCALearner('cca', train, cca_model, IDLoss(), Adam, params.batch_size)
        train_learner.setup_optimizer([{'lr':params.lr, 'weight_decay':params.l2}])

        val_learner = GraphTextCCALearner('cca', val, cca_model, None, None, params.batch_size)

        trainer = Trainer(params)

        manager = Manager(params.cca_model_name)

        eval_metric = 'loss'

        manager.train([train_learner, val_learner], trainer, LossEvaluator('loss_eval'), eval_metric, device=params.device, num_epochs=params.pre_num_epochs, eval_every=10)

        train_learner.load_model(params.cca_model_name+"best.pth")

        graph_out_learner = GraphForwardLearner('g_for', train, graph_model, None, None, params.batch_size)
        text_out_learner = ForwardLearner('t_for', train, text_model, None, None, params.batch_size)
        g_res = manager.eval(graph_out_learner, trainer, CollectionEvaluator('col'), device=params.device)
        t_res = manager.eval(text_out_learner, trainer, CollectionEvaluator('col'), device=params.device)
        with torch.no_grad():
            corr, U, V = cca_model.corr_eval(t_res['res_col'], g_res['res_col'])
            print('corr captured:', corr)

        text_pred_model = ProjTaskModel(text_model, U, params).to(params.device)

        if params.finetune_method == 'partial':
            task_train_learner = MLPForwardLearner('t_task', train, text_pred_model, task_loss, Adam, params.task_batch_size)
        else:
            task_train_learner = ForwardLearner('t_task', train, text_pred_model, task_loss, Adam, params.task_batch_size)
        task_train_learner.setup_optimizer([{'lr':params.finetune_lr, 'weight_decay':params.l2}])

        task_test_learner = ForwardLearner('t_task', test, text_pred_model, None, None, params.task_batch_size)
        task_val_learner = ForwardLearner('t_task', val, text_pred_model, None, None, params.task_batch_size)
        
        manager = Manager(params.task_model_name)
        manager.train([task_train_learner, task_val_learner], task_trainer, task_evlter, task_eval_metric, device=params.device, num_epochs=params.num_epochs, eval_every=5)

        task_train_learner.load_model(params.task_model_name+"best.pth", params.device)

        val_res = manager.eval(task_val_learner, task_trainer, task_evlter, device=params.device)
        test_res = manager.eval(task_test_learner, task_trainer, task_evlter, device=params.device)

        test_res['corr'] = corr.item()

        return val_res, test_res


    if params.psearch:
        dropout_list = [0, 0.1, 0.2]
    else:
        dropout_list = [params.dropout]
    comb = itertools.product(dropout_list)
    best_res = []
    best_config = []
    for drop, in comb:
        params.dropout = drop
        val_res_col = []
        test_res_col = []
        val_res, test_res = cv_with_valid_agnostic(data, labels, data_construct, 5, learn_one_fold)

        print(val_res)
        print(test_res)

        val_metric_res, test_metric_res = np.array(val_res[task_eval_metric]), np.array(test_res[task_eval_metric])
        val_mean, val_std = np.mean(val_metric_res), np.std(val_metric_res)
        test_mean, test_std = np.mean(test_metric_res), np.std(test_metric_res)
        print('val mean:{}, val_std: {}'.format(val_mean, val_std))
        print('test mean:{}, test_std: {}'.format(test_mean, test_std))
        if len(best_res) == 0 or best_res[0]<val_mean:
            best_res = [val_mean, test_mean, test_std]
            best_config = [drop]

    print(best_res)
    print(best_config)

    with open(osp.join(params.exp_dir, 'result'), 'a') as f:
        f.write("\n\n")
        f.write('Dataset: Epic; Label:{} \n'.format(params.label))
        f.write('Model: cca_text_model\n')
        f.write('Optimize wrt {}\n'.format(task_eval_metric))
        f.write('valbest: {}, best res: {}, std: {}\n'.format(best_res[0], best_res[1], best_res[2]))
        f.write('best hparams: {}\n'.format(str(best_config)))