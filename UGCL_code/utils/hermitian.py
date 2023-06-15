#############################################
# Copy and modify based on DiGCN and MagNET
# https://github.com/flyingtango/DiGCN
# https://github.com/matthew-hirn/magnet
#############################################

import numpy as np
from scipy.sparse import coo_matrix


def cheb_poly_sparse(A, K):
    K += 1
    N = A.shape[0]  # [N, N]
    multi_order_laplacian = []
    multi_order_laplacian.append(coo_matrix((np.ones(N), (np.arange(N), np.arange(N))), shape=(N, N), dtype=np.float32))
    if K == 1:
        return multi_order_laplacian
    else:
        multi_order_laplacian.append(A)
        if K == 2:
            return multi_order_laplacian
        else:
            for k in range(2, K):
                multi_order_laplacian.append(2.0 * A.dot(multi_order_laplacian[k - 1]) - multi_order_laplacian[k - 2])
    return multi_order_laplacian


def hermitian_decomp_sparse(pos_edges, neg_edges, size, q=0.25, norm=True, laplacian=True, max_eigen=2,
                            gcn_appr=False, edge_weight=None):
    pos_row, pos_col = pos_edges[:,0], pos_edges[:,1]
    neg_row, neg_col = neg_edges[:,0], neg_edges[:,1]

    if edge_weight is None:
        A = coo_matrix((np.ones(len(pos_row) + len(neg_row)),
                        (
                            np.concatenate((pos_row, neg_row)),
                            np.concatenate((pos_col, neg_col)))
                        ),
                       shape=(size, size), dtype=np.float32)
    else:
        A = coo_matrix((edge_weight,
                        (
                            np.concatenate((pos_row, neg_row)),
                            np.concatenate((pos_col, neg_col)))
                        ),
                       shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    if gcn_appr:
        A += diag

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency

    if norm:
        d = np.array(A_sym.sum(axis=0))[0]  # out degree
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)

    if laplacian:

        phase_pos = coo_matrix((np.ones(len(pos_row)), (pos_row, pos_col)), shape=(size, size), dtype=np.float32)
        theta_pos = q * 1j * phase_pos
        theta_pos.data = np.exp(theta_pos.data)
        theta_pos_t = -q * 1j * phase_pos.T
        theta_pos_t.data = np.exp(theta_pos_t.data)

        phase_neg = coo_matrix((np.ones(len(neg_row)), (neg_row, neg_col)), shape=(size, size), dtype=np.float32)
        theta_neg = (np.pi + q) * 1j * phase_neg
        theta_neg.data = np.exp(theta_neg.data)
        theta_neg_t = (np.pi - q) * 1j * phase_neg.T
        theta_neg_t.data = np.exp(theta_neg_t.data)

        data = np.concatenate((theta_pos.data, theta_pos_t.data, theta_neg.data, theta_neg_t.data))
        theta_row = np.concatenate((theta_pos.row, theta_pos_t.row, theta_neg.row, theta_neg_t.row))
        theta_col = np.concatenate((theta_pos.col, theta_pos_t.col, theta_neg.col, theta_neg_t.col))

        phase = coo_matrix((data, (theta_row, theta_col)), shape=(size, size), dtype=np.complex64)
        Theta = phase

        if norm:
            D = diag
        else:
            d = np.sum(A_sym, axis=0)  # diag of degree array
            D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        L = D - Theta.multiply(A_sym)  # element-wise

    if norm:
        L = (2.0 / max_eigen) * L - diag
    return L


def to_edge_dataset_sparse_sign(q, pos_edges, neg_edges, K, size, laplacian=True, norm=True, max_eigen=2.0, gcn_appr=False):
    L = hermitian_decomp_sparse(pos_edges, neg_edges, size, q, norm=norm, laplacian=laplacian,
                                max_eigen=max_eigen, gcn_appr=gcn_appr)
    multi_order_laplacian = cheb_poly_sparse(L, K)

    return multi_order_laplacian
