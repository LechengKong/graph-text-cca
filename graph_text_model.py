import torch
import torch.nn as nn
from gnnfree.nn.loss import CCALoss
from gnnfree.nn.models.GNN import RGCN


class CCAModel(torch.nn.Module):
    def __init__(self, model1, model2, params):
        super(CCAModel, self).__init__()
        self.params = params
        self.model1 = model1
        self.model2 = model2
        self.corr_eval = CCALoss(self.params.corr_dim)

    def forward(self, input1, input2):
        output1 = self.model1(input1)
        output2 = self.model2(input2)
        corr, Ub, Vb = self.corr_eval(output1, output2)
        closs = -corr
        return closs


class GraphModel(torch.nn.Module):
    def __init__(self,params):
        super(GraphModel, self).__init__()
        self.params = params
        self.gnn = RGCN(params.num_gcn_layers, params.aug_num_rels, params.node_emb_dim, params.emb_dim, dropout=params.graph_dropout, JK=params.JK)

        self.node_emb = nn.Parameter(torch.Tensor(self.params.num_nodes, self.params.node_emb_dim))
        nn.init.xavier_uniform_(self.node_emb, gain=nn.init.calculate_gain('relu'))

        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.params.emb_dim, 2*self.params.emb_dim), torch.nn.BatchNorm1d(2*self.params.emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*self.params.emb_dim, self.params.emb_dim))

        self.use_precomputed_emb = False

    def forward(self, data):
        g, node_ids = data
        if not self.use_precomputed_emb:
            self.graph_update(g)
        return self.get_graph_embedding(g, node_ids)

    def graph_update(self, g):
        g.ndata['feat'] = self.node_emb
        g.ndata['repr'] = self.gnn(g)

    def get_graph_embedding(self, g, node_ids):
        node_repr = g.ndata['repr'][node_ids]
        m_nodes = torch.sign(node_ids+1)
        node_repr = (node_repr*m_nodes.unsqueeze(2)).sum(dim=1)
        node_repr = self.mlp(node_repr)
        # node_repr = self.peer_fc(node_repr)
        # return g.ndata['node_emb'][node_ids]
        return node_repr

class TextModel(torch.nn.Module):
    def __init__(self, params):
        super(TextModel, self).__init__()
        self.params = params

        self.embeddings = nn.Embedding(self.params.vocab_size, self.params.word_emb_dim)

        self.ft_rnn = torch.nn.LSTM(self.params.word_emb_dim, self.params.emb_dim, batch_first=True, bidirectional=True)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(2*self.params.emb_dim, 2*self.params.emb_dim), torch.nn.BatchNorm1d(2*self.params.emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*self.params.emb_dim, self.params.emb_dim))

    def forward(self, data):
        context = data[0]
        embeds = self.embeddings(context)
        repr, (_,_) = self.ft_rnn(embeds)
        repr = repr[:,0,:]
        # repr = embeds.mean(dim=1)
        return self.mlp(repr)

class ProjTaskModel(torch.nn.Module):
    def __init__(self, model, proj, params):
        super(ProjTaskModel, self).__init__()
        self.params = params
        
        self.model = model
        self.proj = proj

        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.params.corr_dim, 2*self.params.corr_dim), torch.nn.BatchNorm1d(2*self.params.corr_dim), torch.nn.ReLU(), torch.nn.Dropout(params.dropout), torch.nn.Linear(2*self.params.corr_dim, self.params.task_dim))

    def forward(self, inp):
        out = self.model(inp)
        return self.mlp(torch.matmul(out, self.proj))

class TaskModel(torch.nn.Module):
    def __init__(self, model, params):
        super(TaskModel, self).__init__()
        self.params = params
        
        self.model = model

        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.params.emb_dim, 2*self.params.emb_dim), torch.nn.BatchNorm1d(2*self.params.emb_dim), torch.nn.ReLU(),torch.nn.Dropout(params.dropout), torch.nn.Linear(2*self.params.emb_dim, self.params.task_dim))

    def forward(self, inp):
        out = self.model(inp)
        return self.mlp(out)

class EmbTextModel(torch.nn.Module):
    def __init__(self, params):
        super(EmbTextModel, self).__init__()
        self.params = params

        self.embeddings = nn.Embedding(self.params.vocab_size, self.params.word_emb_dim)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.params.word_emb_dim, 2*self.params.emb_dim), torch.nn.BatchNorm1d(2*self.params.emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*self.params.emb_dim, self.params.emb_dim))

    def forward(self, data):
        context = data[0]
        size = context.size()
        embeds = self.embeddings(context).view(-1, self.params.word_emb_dim)
        res = self.mlp(embeds).view(size[0],size[1],-1)
        return res.sum(dim=1)

class EmbGraphModel(torch.nn.Module):
    def __init__(self, params):
        super(EmbTextModel, self).__init__()
        self.params = params

        self.embeddings = nn.Embedding(self.params.vocab_size, self.params.word_emb_dim)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.params.word_emb_dim, 2*self.params.emb_dim), torch.nn.BatchNorm1d(2*self.params.emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*self.params.emb_dim, self.params.emb_dim))

    def forward(self, data):
        context = data[0]
        size = context.size()
        embeds = self.embeddings(context).view(-1, self.params.word_emb_dim)
        res = self.mlp(embeds).view(size[0],size[1],-1)
        return res.sum(dim=1)


class GraphLabelingModel(torch.nn.Module):
    def __init__(self,params, labels):
        super(GraphLabelingModel, self).__init__()
        self.params = params
        self.gnn = RGCN(params.num_gcn_layers, params.aug_num_rels, params.node_emb_dim+1, params.emb_dim, dropout=params.graph_dropout, JK=params.JK)

        self.node_emb = nn.Parameter(torch.Tensor(self.params.num_nodes, self.params.node_emb_dim))
        self.labels = labels
        nn.init.xavier_uniform_(self.node_emb, gain=nn.init.calculate_gain('relu'))

        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.params.emb_dim, 2*self.params.emb_dim), torch.nn.BatchNorm1d(2*self.params.emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*self.params.emb_dim, self.params.emb_dim))

        self.use_precomputed_emb = False

    def forward(self, data):
        g, node_ids = data
        if not self.use_precomputed_emb:
            self.graph_update(g)
        return self.get_graph_embedding(g, node_ids)

    def graph_update(self, g):
        g.ndata['feat'] = torch.cat([self.node_emb, self.labels.to(g.device).view(-1,1)], axis=-1)
        g.ndata['repr'] = self.gnn(g)

    def get_graph_embedding(self, g, node_ids):
        node_repr = g.ndata['repr'][node_ids]
        m_nodes = torch.sign(node_ids+1)
        node_repr = (node_repr*m_nodes.unsqueeze(2)).sum(dim=1)
        node_repr = self.mlp(node_repr)
        # node_repr = self.peer_fc(node_repr)
        # return g.ndata['node_emb'][node_ids]
        return node_repr