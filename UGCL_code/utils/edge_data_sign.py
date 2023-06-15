#############################################
# Copy and modify based on DiGCN and MagNET
# https://github.com/flyingtango/DiGCN
# https://github.com/matthew-hirn/magnet
#############################################

import torch
import numpy as np
import random
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.utils import to_undirected
import pickle


def load_directed_signed_graph_link2(root="./data"):
    g_train = root + ".pkl"
    with open(g_train, 'rb') as f:
        g_train = pickle.load(f)

    # get positive edges
    pos_src = g_train[list(g_train.keys())[0]][0]
    pos_dst = g_train[list(g_train.keys())[0]][1]
    pos_edges = [(i, j) for i, j in zip(pos_src, pos_dst)]

    # get negative edges
    neg_src = g_train[list(g_train.keys())[1]][0]
    neg_dst = g_train[list(g_train.keys())[1]][1]
    neg_edges = [(i, j) for i, j in zip(neg_src, neg_dst)]

    return pos_edges, neg_edges


def load_directed_signed_graph_link(root="./data"):
    file = root + ".pkl"
    with open(file, 'rb') as f:
        pos_edges, neg_edges = pickle.load(f)
    return pos_edges, neg_edges


def sub_adj(edge_index, prob, seed):
    sub_train, sub_test = train_test_split(edge_index, test_size=prob, random_state=seed)
    sub_train, sub_val = train_test_split(sub_train, test_size=0.2, random_state=seed)
    return sub_train.T, sub_val.T, sub_test.T


def label_pairs_gen(pos, neg):
    pairs = torch.cat((pos, neg), axis=-1)
    label = np.r_[np.ones(len(pos[0])), np.zeros(len(neg[0]))]
    return pairs, label


def generate_dataset_2class(pos_edge_index, neg_edge_index, splits=5, test_prob=0.2, instance_balancing=True, ratio=3):
    datasets = {}

    for i in range(splits):
        train_pos, val_pos, test_pos = sub_adj(pos_edge_index, prob=test_prob, seed=i * 10+1)
        train_neg, val_neg, test_neg = sub_adj(neg_edge_index, prob=test_prob, seed=i * 10+1)
        datasets[i] = {}

        train = torch.cat((train_pos, train_neg), axis=-1)
        datasets[i]['graph'] = train
        datasets[i]['undirected'] = to_undirected(train).detach().to('cpu').numpy().T
        rng = np.random.default_rng(i)

        datasets[i]['train'] = {}
        datasets[i]['train']['pos_edge'] = train_pos.detach().to('cpu').numpy().T
        datasets[i]['train']['neg_edge'] = train_neg.detach().to('cpu').numpy().T

        ############################################
        # training data
        if instance_balancing == True:
            idx = random.sample(range(train_pos.size(1)), int(train_neg.size(1)*ratio))
            train_pos = train_pos[:, idx]

        pairs, label = label_pairs_gen(train_pos, train_neg)
        perm = rng.permutation(len(pairs[0]))
        datasets[i]['train']['pairs'] = pairs[:, perm].detach().to('cpu').numpy().T
        datasets[i]['train']['label'] = label[perm]

        ############################################
        # validation data
        datasets[i]['validate'] = {}
        pairs, label = label_pairs_gen(val_pos, val_neg)
        perm = rng.permutation(len(pairs[0]))
        datasets[i]['validate']['pairs'] = pairs[:, perm].detach().to('cpu').numpy().T
        datasets[i]['validate']['label'] = label[perm]

        ############################################
        # test data
        datasets[i]['test'] = {}
        pairs, label = label_pairs_gen(test_pos, test_neg)
        perm = rng.permutation(len(pairs[0]))
        datasets[i]['test']['pairs'] = pairs[:, perm].detach().to('cpu').numpy().T
        datasets[i]['test']['label'] = label[perm]

    return datasets


def in_out_degree(edge_index, size):
    edge_index = edge_index.to('cpu')
    A = coo_matrix((np.ones(len(edge_index[0])), (edge_index[0, :], edge_index[1, :])), shape=(size, size),
                   dtype=np.float32).tocsr()
    out_degree = np.sum(A, axis=0).T
    in_degree = np.sum(A, axis=1)
    degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
    return degree


def link_prediction_evaluation(out_val, out_test, y_val, y_test):
    out = torch.exp(out_val).detach().to('cpu').numpy()
    y_val = y_val.detach().to('cpu').numpy()
    pred_label = np.argmax(out, axis=1)
    val_acc_full = accuracy_score(pred_label, y_val)
    out = out[y_val < 2, :2]
    y_val = y_val[y_val < 2]

    prob = out[:, 1] / (out[:, 0] + out[:, 1])
    prob = np.nan_to_num(prob, nan=0.5, posinf=0)
    val_auc = roc_auc_score(y_val, prob)
    pred_label = np.argmax(out, axis=1)
    val_acc = accuracy_score(pred_label, y_val)
    val_f1_macro = f1_score(pred_label, y_val, average='macro')
    val_f1_micro = f1_score(pred_label, y_val, average='micro')
    val_f1_binary = f1_score(pred_label, y_val, average='binary')


    out = torch.exp(out_test).detach().to('cpu').numpy()
    y_test = y_test.detach().to('cpu').numpy()
    pred_label = np.argmax(out, axis=1)
    test_acc_full = accuracy_score(pred_label, y_test)
    out = out[y_test < 2, :2]
    y_test = y_test[y_test < 2]

    prob = out[:, 1] / (out[:, 0] + out[:, 1])
    prob = np.nan_to_num(prob, nan=0.5, posinf=0)
    test_auc = roc_auc_score(y_test, prob)
    pred_label = np.argmax(out, axis=1)
    test_acc = accuracy_score(pred_label, y_test)
    test_f1_macro = f1_score(pred_label, y_test, average='macro')
    test_f1_micro = f1_score(pred_label, y_test, average='micro')
    test_f1_binary = f1_score(pred_label, y_test, average='binary')

    return [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro, val_f1_binary],
            [test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro, test_f1_binary]]

