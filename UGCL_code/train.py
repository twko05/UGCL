#############################################
# Copy and modify based on DiGCN and MagNET
# https://github.com/flyingtango/DiGCN
# https://github.com/matthew-hirn/magnet
#############################################
import torch.optim as optim
import os, argparse
from model import Encoder, Model
from utils.perturb import *
from utils.edge_data_sign import *
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="UGCL link sign prediction")
    parser.add_argument('--dataset', type=str, default='BitCoinAlpha', help='data set selection')
    parser.add_argument('--epochs', type=int, default=2000, help='training epochs')
    parser.add_argument('--num_filter', type=int, default=64, help='num of filters')
    parser.add_argument('--q', type=float, default=0.1, help='q value for the phase matrix')
    parser.add_argument('--K', type=int, default=1, help='K for cheb series')
    parser.add_argument('--layer', type=int, default=2, help='GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--num_class_link', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')
    parser.add_argument('--ensemble', type=int, default=5, help='number of ensemble model')
    parser.add_argument('--device', type=int, default=0, help='Select GPU idx')
    parser.add_argument('--ratio', type=int, default=3, help='pos_neg ratio')
    parser.add_argument('--loss_weight', type=float, default=0.2, help='contrastive_loss_weight')
    parser.add_argument('--batch_size', type=int, default=1024, help='contrastive_loss_batch')
    parser.add_argument('--graph', type=float, default=0.1, help='graph augment')
    parser.add_argument('--perturb_ratio', type=float, default=0.1, help='perturb ratio')
    parser.add_argument('--laplacian', type=int, default=1, help='q perturbing')
    parser.add_argument('--composite', type=int, default=0, help='composite perturbing')
    parser.add_argument('--sigma', type=float, default=0, help='q perturb sigma')
    parser.add_argument('--sign_only', type=float, default=0, help='ignore directions')
    return parser.parse_args()


def main(args):
    data_name = args.dataset

    dataset = load_directed_signed_graph_link2(root='data/'+data_name)
    print("Dataset Loaded "+data_name)
    print("# of Edges: ", len(dataset[0])+len(dataset[1]))

    if 'dataset' in locals():
        pos_edge, neg_edge = dataset
        pos_edge, neg_edge = torch.tensor(pos_edge).to(args.cuda), torch.tensor(neg_edge).to(args.cuda)

    p_max = torch.max(pos_edge).item()
    n_max = torch.max(neg_edge).item()
    size = torch.max(torch.tensor([p_max,n_max])).item() + 1
    datasets = generate_dataset_2class(pos_edge, neg_edge, splits = args.ensemble, test_prob = 0.20, ratio=args.ratio)
    results = np.zeros((args.ensemble, 2, 6))

    stop = 500
    if (data_name == 'Slashdot') or (data_name == 'Epinions'):
        stop = 300
    print('Stop Iteration: ', stop)

    for i in range(args.ensemble):
        edges = datasets[i]['graph']
        pos_edges = datasets[i]['train']['pos_edge']
        neg_edges = datasets[i]['train']['neg_edge']

        X_img = in_out_degree(edges, size).to(args.cuda)
        X_real = X_img.clone()

        encoder = Encoder(X_real.size(-1), K=args.K, label_dim=args.num_class_link,
                            layer=args.layer, num_filter=args.num_filter, dropout=args.dropout)
        model = Model(encoder, num_hidden=args.num_filter, num_proj_hidden=args.num_filter, num_label=args.num_class_link)
        model = model.to(args.cuda)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        y_train = torch.from_numpy(datasets[i]['train']['label']).long().to(args.cuda)
        y_val = torch.from_numpy(datasets[i]['validate']['label']).long().to(args.cuda)
        y_test = torch.from_numpy(datasets[i]['test']['label']).long().to(args.cuda)

        train_index = torch.from_numpy(datasets[i]['train']['pairs']).to(args.cuda)
        val_index = torch.from_numpy(datasets[i]['validate']['pairs']).to(args.cuda)
        test_index = torch.from_numpy(datasets[i]['test']['pairs']).to(args.cuda)

        #################################
        # Train/Validation/Test
        #################################
        best_test_err = 1000.0
        early_stopping = 0
        for epoch in range(args.epochs):
            if early_stopping > stop:
                break
            ####################
            # Train
            ####################
            model.train()

            ####################################################################################
            ### Augmenting
            if args.graph:
                pos_edges1, neg_edges1 = composite_perturb(pos_edges, neg_edges, ratio=args.perturb_ratio)
                pos_edges2, neg_edges2 = composite_perturb(pos_edges, neg_edges, ratio=args.perturb_ratio)
            else:
                pos_edges1, neg_edges1 = pos_edges, neg_edges
                pos_edges2, neg_edges2 = pos_edges, neg_edges

            if args.laplacian:
                q1, q2 = random.sample(np.arange(0,0.5,0.1).tolist(),2)
                q1, q2 = np.pi * q1 , np.pi * q2
            else:
                q1, q2 = args.q, args.q

            if args.composite:
                pos_edges1, neg_edges1 = composite_perturb(pos_edges, neg_edges, ratio=args.perturb_ratio)
                pos_edges2, neg_edges2 = pos_edges, neg_edges
                q2 = random.sample(np.arange(0,0.5,0.1).tolist(),1)[0]
                q1, q2 = args.q , np.pi * q2
            ### Augmenting
            ####################################################################################

            z1 = model(X_real, X_img, q1, pos_edges1, neg_edges1, args, size, train_index)
            z2 = model(X_real, X_img, q2, pos_edges2, neg_edges2, args, size, train_index)
            contrastive_loss = model.contrastive_loss(z1, z2, batch_size=args.batch_size)
            label_loss = model.label_loss(z1, z2, y_train)
            train_loss = args.loss_weight * contrastive_loss + label_loss

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            ####################
            # Validation
            ####################
            model.eval()
            z1 = model(X_real, X_img, q1, pos_edges, neg_edges, args, size, val_index)
            z2 = model(X_real, X_img, q2, pos_edges, neg_edges, args, size, val_index)
            val_loss = model.label_loss(z1, z2, y_val)

            ####################
            # Save weights
            ####################
            save_perform = val_loss.detach().item()
            if save_perform <= best_test_err:
                early_stopping = 0
                best_test_err = save_perform

                ####################
                # Test
                ####################
                z1 = model(X_real, X_img, q1, pos_edges, neg_edges, args, size, test_index)
                z2 = model(X_real, X_img, q2, pos_edges, neg_edges, args, size, test_index)
                out_test = model.prediction(z1, z2)

                [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro, val_f1_binary],
                 [test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro, test_f1_binary]] = \
                    link_prediction_evaluation(out_test, out_test, y_test, y_test)
            else:
                early_stopping += 1

        results[i] = [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro, val_f1_binary],
                      [test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro, test_f1_binary]]
        log_str = ('test_acc:{test_acc:.3f}, test_auc: {test_auc:.3f}, test_f1_macro: {test_f1_macro:.3f},'
                   ' test_f1_micro: {test_f1_micro:.3f}, test_f1_binary: {test_f1_binary:.3f}')
        log_str = log_str.format(test_acc=test_acc, test_auc=test_auc, test_f1_macro=test_f1_macro,
                                 test_f1_micro=test_f1_micro, test_f1_binary=test_f1_binary)
        print('Model:' + str(i) + ' ' + log_str)

    print(
        'Average Performance: test_acc:{:.3f}, test_auc: {:.3f}, test_f1_macro: {:.3f}, test_f1_micro: {:.3f}, test_f1_binary: {:.3f}'.format(
            np.mean(results[:, 1, 1]), np.mean(results[:, 1, 2]), np.mean(results[:, 1, 4]),
            np.mean(results[:, 1, 3]), np.mean(results[:, 1, 5])))
    return results


if __name__ == "__main__":
    args = parse_args()
    args.cuda = 'cuda:'+str(args.device)
    args.q = np.pi * args.q

    results = main(args)

