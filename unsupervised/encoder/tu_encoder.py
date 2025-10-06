import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool

from unsupervised.convs.wgin_conv import WGINConv


class TUEncoder(torch.nn.Module):
    def __init__(self, num_dataset_features, emb_dim=300, num_gc_layers=5, drop_ratio=0.0, pooling_type="standard", is_infograph=False, filter_bands=(0.1, 1.0, 10.0)):
        super(TUEncoder, self).__init__()

        self.pooling_type = pooling_type
        self.emb_dim = emb_dim
        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.is_infograph = is_infograph
        self.filter_bands = filter_bands  # low-pass, middle-pass, high-pass

        self.out_node_dim = self.emb_dim
        if self.pooling_type == "standard":
            self.out_graph_dim = self.emb_dim
        elif self.pooling_type == "layerwise":
            self.out_graph_dim = self.emb_dim * self.num_gc_layers
        else:
            raise NotImplementedError

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            else:
                nn = Sequential(Linear(num_dataset_features, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            conv = WGINConv(nn)
            bn = torch.nn.BatchNorm1d(emb_dim)
            self.convs.append(conv)
            self.bns.append(bn)


    def forward(self, batch, x, edge_index, edge_attr=None, edge_weight=None):
        
        xs_total = []
        freq_filtered_x = torch.clone(x)
        for filter_type in ['low-pass', 'middle-pass', 'high-pass', 'normal']:
            xs = []
            x = freq_filtered_x
            for i in range(self.num_gc_layers):
                x = self.convs[i](x, edge_index, edge_weight, filter_type)
                x = self.bns[i](x)
                if i == self.num_gc_layers - 1:
                    # remove relu for the last layer
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
                xs.append(x)
            xs_total.append(xs)

        if self.pooling_type == "standard":
            xpool = global_add_pool(x, batch)
            return xpool, x
        elif self.pooling_type == "layerwise":
            xpool = [global_add_pool(x, batch) for x in xs_total[0]]
            low_pass = torch.cat(xpool, 1)
            xpool = [global_add_pool(x, batch) for x in xs_total[1]]
            middle_pass = torch.cat(xpool, 1)
            xpool = [global_add_pool(x, batch) for x in xs_total[2]]
            high_pass = torch.cat(xpool, 1)
            xpool = [global_add_pool(x, batch) for x in xs_total[3]]
            normal = torch.cat(xpool, 1)
            if self.is_infograph:
                return low_pass, middle_pass, high_pass, normal, torch.cat(xs, 1)
            else:
                return low_pass, middle_pass, high_pass, normal, x
        else:
            raise NotImplementedError


    def get_embeddings(self, loader, device, is_rand_label=False):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                if isinstance(data, list):
                    data = data[0].to(device)
                data = data.to(device)
                batch, x, edge_index = data.batch, data.x, data.edge_index
                edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _, _, _, _ = self.forward(batch, x, edge_index, edge_weight)

                ret.append(x.cpu().numpy())
                if is_rand_label:
                    y.append(data.rand_label.cpu().numpy())
                else:
                    y.append(data.y.cpu().numpy())

        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y
