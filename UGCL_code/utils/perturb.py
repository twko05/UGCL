import random
import numpy as np

def sign_perturb(pos_edges, neg_edges, ratio=0.1):
    n = len(pos_edges)
    idx = random.sample(list(range(n)), int(n*ratio))
    new_neg_edges = pos_edges[idx]
    survive = list(set(list(range(n))).difference(set(idx)))
    pos_edges = pos_edges[survive]

    n = len(neg_edges)
    idx = random.sample(list(range(n)), int(n * ratio))
    new_pos_edges = neg_edges[idx]
    survive = list(set(list(range(n))).difference(set(idx)))
    neg_edges = neg_edges[survive]

    pos_edges = np.append(pos_edges,new_pos_edges, axis=0)
    neg_edges = np.append(neg_edges,new_neg_edges, axis=0)

    return pos_edges, neg_edges


def direction_perturb(pos_edges, neg_edges, ratio=0.1):
    n = len(pos_edges)
    idx = random.sample(list(range(n)), int(n*ratio))
    new_pos_edges = pos_edges[idx]
    survive = list(set(list(range(n))).difference(set(idx)))
    pos_edges = pos_edges[survive]
    new_pos_edges = np.array([[i[1],i[0]] for i in new_pos_edges])
    pos_edges = np.append(pos_edges, new_pos_edges, axis=0)

    n = len(neg_edges)
    idx = random.sample(list(range(n)), int(n*ratio))
    new_neg_edges = neg_edges[idx]
    survive = list(set(list(range(n))).difference(set(idx)))
    neg_edges = neg_edges[survive]
    new_neg_edges = np.array([[i[1],i[0]] for i in new_neg_edges])
    neg_edges = np.append(neg_edges, new_neg_edges, axis=0)

    return pos_edges, neg_edges



def direction_perturb_node(pos_edges, neg_edges, ratio=0.1):
    pos_edges = pos_edges.T
    neg_edges = neg_edges.T

    n = len(pos_edges)
    idx = random.sample(list(range(n)), int(n*ratio))
    new_pos_edges = pos_edges[idx]
    survive = list(set(list(range(n))).difference(set(idx)))
    pos_edges = pos_edges[survive]
    new_pos_edges = np.array([[i[1],i[0]] for i in new_pos_edges])
    pos_edges = np.append(pos_edges, new_pos_edges, axis=0)

    n = len(neg_edges)
    idx = random.sample(list(range(n)), int(n*ratio))
    new_neg_edges = neg_edges[idx]
    survive = list(set(list(range(n))).difference(set(idx)))
    neg_edges = neg_edges[survive]
    new_neg_edges = np.array([[i[1],i[0]] for i in new_neg_edges])
    neg_edges = np.append(neg_edges, new_neg_edges, axis=0)

    return pos_edges.T, neg_edges.T

def composite_perturb(pos_edges, neg_edges, ratio=0.1):
    pos_edges, neg_edges = sign_perturb(pos_edges, neg_edges, ratio=ratio)
    pos_edges, neg_edges = direction_perturb(pos_edges, neg_edges, ratio=ratio)
    return pos_edges, neg_edges


def label_perturb(pos_edges, neg_edges, i):
    y_train = np.append(pos_edges,neg_edges, axis=0)
    label = [1]*len(pos_edges) + [0]*len(neg_edges)
    rng = np.random.default_rng(i)
    perm = rng.permutation(len(y_train))
    return y_train[perm], label[perm]

