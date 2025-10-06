import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from torch_scatter import scatter
from tqdm import tqdm

from datasets import MoleculeDataset
from transfer.learning import GInfoMinMax, ViewLearner
from transfer.model import GNN
from unsupervised.utils import initialize_edge_weight
import sys


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def compute_returns(next_value, rewards, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R
        returns.insert(0, R)
    return returns



def train(args, model, device, dataset, model_optimizer):
    dataset = dataset.shuffle()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)
    fin_model_loss = 0.0
    for epoch in tqdm(range(1, args.epochs)):
        logging.info('====epoch {}'.format(epoch))
        model_loss_all = 0
        for batch in dataloader:
            # set up
            batch = batch.to(device)
            # Train model based on frequency filtering
            model.train()
            model.zero_grad()
            low_pass, middle_pass, high_pass, normal = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)
            model_loss, quality = model.calc_loss(low_pass, middle_pass, high_pass, normal, model)
            model_loss_all += model_loss.item() * batch.num_graphs
            # 2.b. standard gradient descent formulation
            if quality > 0.8:  # Example threshold for "good quality" embeddings
                new_lr = args.model_lr * 0.9  # Decrease learning rate
            else:
                new_lr = args.model_lr * 1.1  # Increase learning rate
            new_lr = max(min(new_lr, args.max_lr), args.min_lr)
            for param_group in model_optimizer.param_groups:
                param_group['lr'] = new_lr

            model_loss.backward()
            model_optimizer.step()
        fin_model_loss = model_loss_all / len(dataloader)
        logging.info('Epoch {}, Model Loss {:.4f}'.format(epoch, fin_model_loss))
        if epoch % 1 == 0:
            torch.save(model.gnn.state_dict(), "./models_gf2l/chem/pretrain_gf2l_encoder_epoch_"+ str(epoch)+".pth")
        fin_model_loss = model_loss_all / len(dataloader)

    return fin_model_loss

def run(args):
    Path("./models_gf2l/chem").mkdir(parents=True, exist_ok=True)
    
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        root_logger.removeHandler(h)
    log_file = './gf2l_pretrain.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='a')
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    setup_seed(args.seed)

    my_transforms = Compose([initialize_edge_weight])
    dataset = MoleculeDataset("original_datasets/transfer/"+args.dataset, dataset=args.dataset,
                              transform=my_transforms)
    model = GInfoMinMax(
        GNN(num_layer=args.num_gc_layers, emb_dim=args.emb_dim, JK="last", drop_ratio=args.drop_ratio, gnn_type="gin"),
        proj_hidden_dim=args.emb_dim).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    model_loss = train(args, model, device, dataset, model_optimizer)




def arg_parse():
    parser = argparse.ArgumentParser(description='Transfer Learning GF2L Pretrain on ZINC 2M')

    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='Dataset')
    # parser.add_argument('--model_lr', type=float, default=1e-3,
    #                     help='Model Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=3,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--model_lr', type=float, default=1e-5, help='Model Learning rate.')
    parser.add_argument('--max_lr', type=float, default=3e-3, help='Max Learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Min Learning rate.')
    # parser.add_argument('--num_gc_layers', type=int, default=4,
    #                     help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='layerwise',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.0,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=11,
                        help='Train Epochs')
    parser.add_argument('--seed', type=int, default=2024)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    run(args)