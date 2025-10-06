import argparse
import logging
import random
import yaml
from yaml import SafeLoader

import numpy as np
import torch
from sklearn.svm import LinearSVC, SVC
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from torch_scatter import scatter
from datasets import TUDataset, TUEvaluator
from unsupervised.embedding_evaluation import EmbeddingEvaluation
from unsupervised.encoder import TUEncoder
from unsupervised.learning import GInfoMinMax
from unsupervised.utils import initialize_edge_weight, initialize_node_features, set_tu_dataset_y_shape
from unsupervised.view_learner import ViewLearner
from feature_expansion import FeatureExpander
from datasets_tu import get_dataset

from time import perf_counter as t
import sys


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def run(args):
    logging.info("Running tu......")
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    setup_seed(args.seed)

    evaluator = TUEvaluator()

    if args.dataset in {'MUTAG', 'COLLAB', 'REDDIT-BINARY'}:
        my_transforms = Compose([initialize_node_features, initialize_edge_weight, set_tu_dataset_y_shape])
        dataset = TUDataset("./original_datasets/", args.dataset, transform=my_transforms)

    else:
        dataset = get_dataset(args.dataset, sparse=True, feat_str='deg+odeg100', root='../data')

    # g = torch.Generator()
    # g.manual_seed(args.seed)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=g)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    n_features = dataset.num_features
    logging.info('n_features: {}'.format(n_features))

    model = GInfoMinMax(
        TUEncoder(num_dataset_features=n_features, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
        args.emb_dim).to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    if args.downstream_classifier == "linear":
        ee = EmbeddingEvaluation(LinearSVC(dual=False, fit_intercept=True), evaluator, dataset.task_type, dataset.num_tasks, device, param_search=True)
    else:
        ee = EmbeddingEvaluation(SVC(), evaluator, dataset.task_type, dataset.num_tasks, device, param_search=True)

    model.eval()
    train_score, val_score, test_score = ee.kf_embedding_evaluation_autogcl(model.encoder, dataset)
    logging.info(
        "Before training Embedding Eval Scores: Train: {:.4f} Val: {:.4f} Test: {:.4f}".format(train_score, val_score, test_score))


    model_losses = []
    valid_curve = []
    test_curve = []
    train_curve = []
    start = t()
    num_T = 5   # Trigger learning

    for epoch in range(1, args.epochs + 1):
        model_loss_all = 0

        for batch in dataloader:
            # set up
            batch = batch.to(device)
            # Train model based on frequency filtering
            model.train()
            model.zero_grad()
            low_pass, middle_pass, high_pass, normal = model(batch.batch, batch.x, batch.edge_index, None, None)

            # 2.b. standard gradient descent formulation
            if epoch % num_T == 1:
                model_loss, quality = model.calc_loss(epoch, num_T, low_pass, middle_pass, high_pass, normal, model, lambda_reg=args.lambda_reg, gamma=args.gamma)
                model_loss_all += model_loss.item() * batch.num_graphs
                if quality > 0.8:  # Example threshold for "good quality" embeddings
                    new_lr = args.model_lr * 0.9  # Decrease learning rate
                else:
                    new_lr = args.model_lr * 1.1  # Increase learning rate
                new_lr = max(min(new_lr, args.max_lr), args.min_lr)
                for param_group in model_optimizer.param_groups:
                    param_group['lr'] = new_lr
            else:
                model_loss = model.calc_loss_reduce(epoch, num_T, low_pass, middle_pass, high_pass, normal, model, lambda_reg=args.lambda_reg, gamma=args.gamma)
                model_loss_all += model_loss.item() * batch.num_graphs
                
            model_loss.backward()
            model_optimizer.step()

        fin_model_loss = model_loss_all / len(dataloader)
        logging.info('Epoch {}, Model Loss {:.4f}'.format(epoch, fin_model_loss))
        model_losses.append(fin_model_loss)
        if epoch % args.eval_interval == 0:
            model.eval()
            train_score, val_score, test_score = ee.kf_embedding_evaluation_autogcl(model.encoder, dataset)

            logging.info(
                "Metric: {} Train: {:.4f} Val: {:.4f} Test: {:.4f}".format(evaluator.eval_metric, train_score, val_score, test_score))
            train_curve.append(train_score)
            valid_curve.append(val_score)
            test_curve.append(test_score)

    now = t()
    print("total time: ", now-start)
    memory_stats = torch.cuda.memory_stats(device)
    peak_memory = memory_stats["allocated_bytes.all.peak"]
    print(f"Peak Memory: {peak_memory / 1e6} MB")
    logging.info('total time: {}'.format(now-start))

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    logging.info('FinishedTraining!')
    logging.info('Dataset: {}'.format(args.dataset))
    logging.info('drop_ratio: {}'.format(args.drop_ratio))
    logging.info('BestEpoch: {}'.format(best_val_epoch))
    logging.info('BestTrainScore: {}'.format(best_train))
    logging.info('BestValidationScore: {}'.format(valid_curve[best_val_epoch]))
    logging.info('FinalTestScore: {}'.format(test_curve[best_val_epoch]))

    return valid_curve[best_val_epoch], test_curve[best_val_epoch]


def arg_parse():
    parser = argparse.ArgumentParser(description='A2C-GCL TU')

    parser.add_argument('--alpha', type=float, default=0.1, help='learning rate adjustment factor.')
    parser.add_argument('--lambda_reg', type=float, default=1e-2, help='Regularization.')
    parser.add_argument('--gamma', type=float, default=-0.6, help='margin coeeficient.')
    parser.add_argument('--dataset', type=str, default='REDDIT-MULTI-5K', help='Dataset') # NCI1, PROTEINS, MUTAG, DD, COLLAB, REDDIT-BINARY, REDDIT-MULTI-5K, IMDB-BINARY, IMDB-MULTI
    parser.add_argument('--model_lr', type=float, default=1e-5, help='Model Learning rate.')
    parser.add_argument('--max_lr', type=float, default=3e-3, help='Max Learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Min Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=4, help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='layerwise', help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.0, help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=400, help='Train Epochs')
    parser.add_argument('--eval_interval', type=int, default=5, help="eval epochs interval")
    parser.add_argument('--downstream_classifier', type=str, default="linear", help="Downstream classifier is linear or non-linear")
    
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    data_list = ['MUTAG']

    for dataset in data_list:
        res = []
        args.dataset = dataset
        root_logger = logging.getLogger()
        for h in root_logger.handlers:
            root_logger.removeHandler(h)
        log_file = args.dataset+'.log'
        logging.basicConfig(filename=log_file, level=logging.INFO, filemode='a')
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

        for i in range(5):
            val, test = run(args)
            res.append(test)
            args.seed += 1
        res_array = np.array(res)
        logging.info('Mean Testscore: {:.2f}±{:.2f}'.format( np.mean(res_array)*100, np.std(res_array)*100 ))
        print("dataset: ", dataset)
        print(np.mean(res_array),np.std(res_array))