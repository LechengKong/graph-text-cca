import pickle as pkl
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch

from dataset import NodeTextLabelDataset

# def code_cleanup(code):
#     return code[:2]+code[3:5]

def code_cleanup(code):
    return code.strip().replace('.','')


def open_and_load_pickle(filename):
    open_file = open(filename, "rb")
    data = pkl.load(open_file)
    open_file.close()
    return data

def process_vocab(w2i, i2w, oov_char='^', padding_char=''):
    vocab_size = np.max(np.array(list(w2i.values())))
    i2w[vocab_size] = oov_char
    if vocab_size!= np.max(np.array(list(i2w.keys()))):
        print("dictionary mismatch")
        quit()
    vocab_size+=1
    w2i[padding_char]=vocab_size
    i2w[vocab_size] = padding_char

    padding_index = w2i[padding_char]
    vocab_size += 1
    return vocab_size, padding_index

def get_max_text_len(collection, ind):
    max_len = 0
    for data in collection:
        text = data[ind]
        if len(text)>max_len:
            max_len = len(text)
    return max_len

def sequence_max_len(sequences):
    max_len = 0
    for data in sequences:
        if len(data)>max_len:
            max_len = len(data)
    return max_len


def get_indexed_data(data, max_prc_len, max_ft_len, max_ct_len, e2n, w2i, padding_index):
    code_text = []
    # print(e2n)
    for icd, text, label in data:
        if len(text) == 0:
            continue
        nodes = [e2n[code_cleanup(c)] for c in icd]
        node_indices = np.zeros(max_prc_len, dtype=int)-1
        node_indices[:len(nodes)] = nodes
        text_indices = np.zeros(max_ft_len, dtype=int)+padding_index
        text_indices[:len(text)] = [w2i[w] for w in text]
        processed_row = [node_indices, text_indices, label]
        code_text.append(processed_row)
    return code_text

def value2index(data, map, max_len, padding_value=-1):
    updated_data = []
    for line in data:
        data = [map[v] for v in line]
        index = np.zeros(max_len, dtype=int)+padding_value
        index[:len(data)] = data
        updated_data.append(index)
    return updated_data


def data_split(code_mat, ratio=0.1):
    num_code_text = len(code_mat)
    perm = np.random.permutation(num_code_text)
    valid = int(num_code_text*ratio)
    valid_code_text = code_mat[perm[:valid]]
    train_code_text = code_mat[perm[valid:]]
    return train_code_text, valid_code_text


def prepare_ct_graph(c2d, num_nodes, max_ct_len, padding_index, e2n, w2i):
    
    emb_pick = np.zeros((num_nodes, max_ct_len), dtype=int)+padding_index
    for k,v in c2d.items():
        clcd = code_cleanup(k)
        node_idx = e2n[clcd]
        emb_pick[node_idx,:len(v)] = [w2i[w] for w in v]
    return emb_pick


def split_multiple_set(data, ratio):
    data_l = len(data[0])
    ratio = np.cumsum(ratio)
    interval = list((ratio*data_l).astype(int))
    interval.insert(0, 0)
    perm = np.random.permutation(data_l)
    splited_data = []
    for i in range(len(interval)-1):
        index_s = perm[interval[i]: interval[i+1]]
        splited_data.append([[v[j] for v in data] for j in index_s])
    return splited_data

def sample_by_index(data, index):
    keep_data = []
    for v in data:
        sampled_v = [v[i] for i in index]
        keep_data.append(sampled_v)
    return keep_data

def kfold_miss_node_split(unique_codes, fold, codes_ind, texts_ind, labels, g):
    perm = np.random.permutation(len(unique_codes))

    splits = np.array_split(perm, fold)

    data_col = []

    for sp in splits:
        droped_codes = unique_codes[sp]

        droped_codes_dict = set(droped_codes)
        if -1 in droped_codes_dict:
            droped_codes_dict.remove(-1)

        drop_ind = []
        for i,v in enumerate(codes_ind):
            for c in v:
                if c in droped_codes_dict:
                    drop_ind.append(i)
                    break
        drop_ind = np.array(drop_ind)
        full_arr = np.ones(len(codes_ind))
        full_arr[drop_ind] = 0
        keep_ind = full_arr.nonzero()[0]

        data = list(zip(codes_ind, texts_ind, labels))

        train_data = sample_by_index([data], keep_ind)[0]
        droped_data = sample_by_index([data], drop_ind)
        ind = np.arange(len(droped_data[0]))
        val_inter = int(len(ind)/2)
        val_ind = ind[:val_inter]
        test_ind = ind[val_inter:]
        val_data = sample_by_index(droped_data, val_ind)[0]
        test_data = sample_by_index(droped_data, test_ind)[0]



        train = NodeTextLabelDataset(train_data, g)
        valid = NodeTextLabelDataset(val_data, g)
        test = NodeTextLabelDataset(test_data, g)

        data_col.append([train, test, valid])

    return data_col

def kfold_miss_node_split_unseen(unique_codes, fold, codes_ind, texts_ind, labels, num_nodes, g):
    perm = np.random.permutation(len(unique_codes))

    splits = np.array_split(perm, fold)

    data_col = []

    for i, sp in enumerate(splits):
        droped_codes = unique_codes[sp]
        

        droped_codes_dict = set(droped_codes)
        # print(droped_codes_dict)
        if -1 in droped_codes_dict:
            droped_codes_dict.remove(-1)

        drop_ind = []
        for j,v in enumerate(codes_ind):
            for c in v:
                if c in droped_codes_dict:
                    drop_ind.append(j)
                    break
        drop_ind = np.array(drop_ind)
        full_arr = np.ones(len(codes_ind))
        full_arr[drop_ind] = 0
        keep_ind = full_arr.nonzero()[0]

        second_split = splits[int((i+1)%len(splits))]

        train_miss_nodes = unique_codes[second_split[:int(len(second_split)/2)]]
        miss_nodes_dict = set(train_miss_nodes)
        if -1 in miss_nodes_dict:
            miss_nodes_dict.remove(-1)

        miss_ind = []
        for v in keep_ind:
            for c in codes_ind[v]:
                if c in miss_nodes_dict:
                    miss_ind.append(v)
                    break
        miss_ind = np.array(miss_ind)
        full_arr[miss_ind] = 0
        part_keep_ind = full_arr.nonzero()[0]

        data = list(zip(codes_ind, texts_ind, labels))

        train_data = sample_by_index([data], part_keep_ind)[0]
        miss_data = sample_by_index([data], keep_ind)[0]
        droped_data = sample_by_index([data], drop_ind)

        ind = np.arange(len(droped_data[0]))
        val_ind = ind[:int(len(ind)/2)]
        test_ind = ind[int(len(ind)/2):]
        
        val_data = sample_by_index(droped_data, val_ind)[0]
        test_data = sample_by_index(droped_data, test_ind)[0]



        cca_train = NodeTextLabelDataset(train_data, g)
        task_train = NodeTextLabelDataset(miss_data, g)
        valid = NodeTextLabelDataset(val_data, g)
        test = NodeTextLabelDataset(test_data, g)
        print('cca_train:{},task_train:{},valid:{},test:{}'.format(len(cca_train),len(task_train), len(valid), len(test)))

        all_miss_nodes = list(miss_nodes_dict.union(droped_codes_dict))
        code_labels = torch.ones(num_nodes)
        code_labels[all_miss_nodes] = 0

        data_col.append([cca_train, task_train,test, valid, code_labels])

    return data_col