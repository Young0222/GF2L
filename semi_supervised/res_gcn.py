from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv
# from gcn_conv import GCNConv

from IPython import embed

class ResGCN(nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, dataset, hidden, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0,
                 edge_norm=True):
        super().__init__()

        # print("GFN:", gfn)

        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.res_branch = res_branch
        self.collapse = collapse
        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout

        # GCNConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        if "xg" in dataset[0]:  # Utilize graph level features.
            self.use_xg = True
            self.bn1_xg = BatchNorm1d(dataset[0].xg.size(1))
            self.lin1_xg = Linear(dataset[0].xg.size(1), hidden)
            self.bn2_xg = BatchNorm1d(hidden)
            self.lin2_xg = Linear(hidden, hidden)
        else:
            self.use_xg = False

        hidden_in = dataset.num_features

        self.bn_feat = BatchNorm1d(hidden_in)
        feat_gfn = True  # set true so GCNConv is feat transform
        self.conv_feat = GCNConv(hidden_in, hidden)
        if "gating" in global_pool:
            self.gating = nn.Sequential(
                Linear(hidden, hidden),
                nn.ReLU(),
                Linear(hidden, 1),
                nn.Sigmoid())
        else:
            self.gating = None
        self.bns_conv = nn.ModuleList()
        self.convs = nn.ModuleList()
        if self.res_branch == "resnet":
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GCNConv(hidden, hidden))
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GCNConv(hidden, hidden))
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GCNConv(hidden, hidden))
        else:
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GCNConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = nn.ModuleList()
        self.lins = nn.ModuleList()
        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, dataset.num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0001)
        
        self.proj_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))

    def reset_parameters(self):
        raise NotImplemented(
            "This is prune to bugs (e.g. lead to training on test set in "
            "cross validation setting). Create a new model instance instead.")

    def forward(self, data, edge_weight):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        if self.res_branch == "BNConvReLU":
            return self.forward_BNConvReLU(x, edge_index, batch, edge_weight, xg)
        else:
            raise ValueError("Unknown res_branch %s" % self.res_branch)

    def forward_BNConvReLU(self, x, edge_index, batch, edge_weight, xg=None):
        # embed()
        # exit()
        # print("this forward")
        x_list = []
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index, edge_weight))
        x_initial = x.clone()
        for i, conv in enumerate(self.convs):   # low_pass
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index, edge_weight))
            x = x + x_ if self.conv_residual else x_
            if (i+1) == len(self.convs):
                x_list.append(x)
        x = x_initial
        for i, conv in enumerate(self.convs):   # middle_pass
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index, edge_weight))
            x_ = self.bns_conv[i](x_)
            x_ = F.relu(conv(x_, edge_index, edge_weight))
            x = x_ - x if self.conv_residual else x_
            if (i+1) == len(self.convs):
                x_list.append(x)
        x = x_initial
        for i, conv in enumerate(self.convs):   # high_pass
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index, edge_weight))
            x = x - x_ if self.conv_residual else x_
            if (i+1) == len(self.convs):
                x_list.append(x)
        x = x_initial
        for i, conv in enumerate(self.convs):   # high_pass
            x_ = self.bns_conv[i](x)
            x_ = F.relu(x_)
            x = x_ if self.conv_residual else x_
            if (i+1) == len(self.convs):
                x_list.append(x)
        z_list = []
        for x in x_list:
            gate = 1 if self.gating is None else self.gating(x)
            x = self.global_pool(x * gate, batch)
            x = x if xg is None else x + xg
            for i, lin in enumerate(self.lins):
                x_ = self.bns_fc[i](x)
                x_ = F.relu(lin(x_))
                x = x + x_ if self.fc_residual else x_
            x = self.bn_hidden(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin_class(x)
            z_list.append(x)
        return F.log_softmax(z_list[0], dim=-1), F.log_softmax(z_list[1], dim=-1), F.log_softmax(z_list[2], dim=-1), F.log_softmax(z_list[3], dim=-1)

    def forward_last_layers(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None
        
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        
        out1 = self.bn_hidden(x)
        if self.dropout > 0:
            out1 = F.dropout(out1, p=self.dropout, training=self.training)
        
        out2 = self.lin_class(out1)
        out3 = F.log_softmax(out2, dim=-1)
        return out1, out2, out3

    def forward_cl(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        return x
    
    def forward_graph_cl(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.proj_head(x)
        return x


    @staticmethod
    def calc_loss(low_pass, middle_pass, high_pass, normal, model, lambda_reg=1e-2):
        def reconstruction_loss(low_pass, middle_pass, high_pass, normal, margin=1.0):
            reconstructed_signal = (low_pass + middle_pass + high_pass) / 3
            reconstruction_loss = F.mse_loss(reconstructed_signal, normal)
            return F.relu(reconstruction_loss - margin)

        def cross_scale_consistency_loss(low_pass, middle_pass, high_pass, margin=-0.6):
            low_middle_cosine = F.cosine_similarity(low_pass, middle_pass, dim=-1)
            middle_high_cosine = F.cosine_similarity(middle_pass, high_pass, dim=-1)
            cross_scale_loss = - (torch.mean(low_middle_cosine) + torch.mean(middle_high_cosine))
            return F.relu(cross_scale_loss - margin)

        def indicator(z, t = 2):
            num_sample = 5
            num = z.shape[0]
            p = torch.ones(num)
            index = p.multinomial(num_samples=num_sample, replacement=True)
            z_sample = z[index]
            total_separation = -torch.pdist(z_sample, p=2).pow(2).mul(-t).exp().mean().log()
            return total_separation
            
        reconstruction_loss = reconstruction_loss(low_pass, middle_pass, high_pass, normal)
        cross_scale_loss = cross_scale_consistency_loss(low_pass, middle_pass, high_pass)
        l2_reg_loss = 0.0
        for param in model.parameters():
            l2_reg_loss += torch.norm(param, p=2) ** 2
        total_loss = reconstruction_loss + cross_scale_loss + lambda_reg * l2_reg_loss

        quality = (indicator(low_pass) + indicator(middle_pass) + indicator(high_pass)) / 3

        return total_loss, quality