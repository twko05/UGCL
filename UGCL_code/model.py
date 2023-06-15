import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from utils.hermitian import to_edge_dataset_sparse_sign


def process(mul_L_real, mul_L_imag, weight, X_real, X_imag):
    data = torch.spmm(mul_L_real, X_real)
    real = torch.matmul(data, weight)
    data = -1.0 * torch.spmm(mul_L_imag, X_imag)
    real += torch.matmul(data, weight)

    data = torch.spmm(mul_L_imag, X_real)
    imag = torch.matmul(data, weight)
    data = torch.spmm(mul_L_real, X_imag)
    imag += torch.matmul(data, weight)
    return torch.stack([real, imag])


class SDConv(nn.Module):
    def __init__(self, in_c, out_c, K, bias=True):
        super(SDConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, data):
        X_real, X_imag, L_norm_real, L_norm_imag = data[0], data[1], data[2], data[3]
        future = []
        for i in range(len(L_norm_real)):  # [K, B, N, D]
            future.append(torch.jit.fork(process, L_norm_real[i], L_norm_imag[i], self.weight[i], X_real, X_imag))
        result = []
        for i in range(len(L_norm_real)):
            result.append(torch.jit.wait(future[i]))
        result = torch.sum(torch.stack(result), dim=0)

        real = result[0]
        imag = result[1]
        return (real + self.bias, imag + self.bias, L_norm_real, L_norm_imag)


class complex_relu_layer(nn.Module):
    def __init__(self, ):
        super(complex_relu_layer, self).__init__()

    def complex_relu(self, real, img):
        mask = 1.0 * (real >= 0)
        return mask * real, mask * img

    def forward(self, data):
        real, img, L_norm_real, L_norm_imag = data[0], data[1], data[2], data[3]
        if img == None:
            img = real[1]
            real = real[0]

        real, img = self.complex_relu(real, img)
        return (real, img, L_norm_real, L_norm_imag)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class Encoder(nn.Module):
    def __init__(self, in_c, num_filter=2, K=1, label_dim=2, layer=2, dropout=False):
        super(Encoder, self).__init__()

        activation_func = complex_relu_layer
        chebs = [SDConv(in_c=in_c, out_c=num_filter, K=K)]
        chebs.append(activation_func())

        for i in range(1, layer):
            chebs.append(SDConv(in_c=num_filter, out_c=num_filter, K=K))
            chebs.append(activation_func())
        self.Chebs = torch.nn.Sequential(*chebs)
        self.linear = nn.Linear(num_filter * 4, num_filter)
        self.dropout = dropout


    def forward(self, real, imag, q, pos_edges, neg_edges, args, size, index):
        L = to_edge_dataset_sparse_sign(q, pos_edges, neg_edges, args.K,
                                        size, laplacian=True, norm=True, gcn_appr=False)
        L_img, L_real = [], []
        for ind_L in range(len(L)):
            L_img.append(sparse_mx_to_torch_sparse_tensor(L[ind_L].imag).to(args.cuda))
            L_real.append(sparse_mx_to_torch_sparse_tensor(L[ind_L].real).to(args.cuda))

        inputs = (real, imag, L_real, L_img)
        convs = self.Chebs(inputs)
        real, imag, _, _ = convs
        x = torch.cat((real[index[:, 0]], real[index[:, 1]], imag[index[:, 0]], imag[index[:, 1]]), dim=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.linear(x)
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,  num_label: int, tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.prediction_layer = torch.nn.Linear(num_proj_hidden*2, num_label)
        self.prediction_layer_sdgcn = torch.nn.Linear(num_proj_hidden, num_label)

    def forward(self, real, imag, q, pos_edges, neg_edges, args, size, index):
        return self.encoder(real, imag, q, pos_edges, neg_edges, args, size, index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batch_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        num_nodes = z1.size(0)
        idx = random.sample(list(range(num_nodes)),batch_size)
        z1, z2 = z1[idx], z2[idx]
        return self.semi_loss(z1,z2)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batch_semi_loss(h1, h2, batch_size)
            l2 = self.batch_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret

    def label_loss(self, z1, z2, y_label):
        x = torch.cat((z1, z2), dim=-1)
        x = self.prediction_layer(x)
        x = F.log_softmax(x, dim=1)

        loss = F.nll_loss(x, y_label)
        return loss

    def prediction(self, z1, z2):
        x = torch.cat((z1, z2), dim=-1)
        x = self.prediction_layer(x)
        x = F.log_softmax(x, dim=1)
        return x
