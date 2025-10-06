import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F



class GInfoMinMax(torch.nn.Module):
    def __init__(self, gnn, proj_hidden_dim=300):
        super(GInfoMinMax, self).__init__()

        self.gnn = gnn
        self.pool = global_mean_pool
        self.input_proj_dim = gnn.emb_dim

        self.projection_head = Sequential(Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
                                       Linear(proj_hidden_dim, proj_hidden_dim))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))


    def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):
        # x = self.gnn(x, edge_index, edge_attr, edge_weight)
        # x = self.pool(x, batch)
        # x = self.projection_head(x)
        # return x

        low_pass = self.gnn(x, edge_index, edge_attr, edge_weight, 'low-pass')
        middle_pass = self.gnn(x, edge_index, edge_attr, edge_weight, 'middle-pass')
        high_pass = self.gnn(x, edge_index, edge_attr, edge_weight, 'high-pass')
        normal = self.gnn(x, edge_index, edge_attr, edge_weight, 'normal')

        low_pass = self.pool(low_pass, batch)
        low_pass = self.projection_head(low_pass)
        middle_pass = self.pool(middle_pass, batch)
        middle_pass = self.projection_head(middle_pass)
        high_pass = self.pool(high_pass, batch)
        high_pass = self.projection_head(high_pass)
        normal = self.pool(normal, batch)
        normal = self.projection_head(normal)

        return low_pass, middle_pass, high_pass, normal
        

    @staticmethod
    def calc_loss(low_pass, middle_pass, high_pass, normal, model, lambda_reg=1e-2, margin=1.0):
        def reconstruction_loss(low_pass, middle_pass, high_pass, normal):
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

