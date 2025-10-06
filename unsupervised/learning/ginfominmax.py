import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU


class GInfoMinMax(torch.nn.Module):
    def __init__(self, encoder, proj_hidden_dim=300):
        super(GInfoMinMax, self).__init__()

        self.encoder = encoder
        self.input_proj_dim = self.encoder.out_graph_dim

        self.proj_head = Sequential(Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
                                       Linear(proj_hidden_dim, proj_hidden_dim))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):

        low_pass, middle_pass, high_pass, normal, _ = self.encoder(batch, x, edge_index, edge_attr, edge_weight)
        low_pass = self.proj_head(low_pass)
        middle_pass = self.proj_head(middle_pass)
        high_pass = self.proj_head(high_pass)
        normal = self.proj_head(normal)
        # z shape -> Batch x proj_hidden_dim
        return low_pass, middle_pass, high_pass, normal

    @staticmethod
    def calc_loss_reduce(epoch, num_T, low_pass, middle_pass, high_pass, normal, model, 
                    lambda_reg=1e-2, gamma=-0.6, low_w=1.0, mid_w=1.0, high_w=1.0):
        def reconstruction_loss_raw(low_pass, middle_pass, high_pass, normal, low_w, mid_w, high_w):
            reconstructed_signal = (low_w*low_pass + mid_w*middle_pass + high_w*high_pass) / 3
            reconstruction_loss = F.mse_loss(reconstructed_signal, normal)
            return F.relu(reconstruction_loss - 1.0)

        def reconstruction_loss(low_pass, middle_pass, high_pass, normal, low_w, mid_w, high_w):
            """使用注意力机制融合频率嵌入进行重建 (用于 calc_loss_reduce)"""
            device = low_pass.device
            N, D = low_pass.shape
            concatenated = torch.cat([low_pass, middle_pass, high_pass], dim=-1) # [N, 3*D]

            part1 = torch.abs(concatenated[:, :D]).mean(dim=1, keepdim=True) # [N, 1]
            part2 = torch.abs(concatenated[:, D:2*D]).mean(dim=1, keepdim=True) # [N, 1]
            part3 = torch.abs(concatenated[:, 2*D:]).mean(dim=1, keepdim=True) # [N, 1]
            attention_scores = torch.cat([part1, part2, part3], dim=1) # [N, 3]
            attention_weights = F.softmax(attention_scores, dim=-1) # [N, 3]
            fused_signal = (attention_weights.unsqueeze(-1) *
                            torch.stack([low_pass, middle_pass, high_pass], dim=1)).sum(dim=1) # [N, D]

            reconstruction_loss_val = F.mse_loss(fused_signal, normal)
            return F.relu(reconstruction_loss_val - 1.0)

        def indicator(z, t = 2):
            num_sample = 5
            num = z.shape[0]
            p = torch.ones(num)
            index = p.multinomial(num_samples=num_sample, replacement=True)
            z_sample = z[index]
            total_separation = -torch.pdist(z_sample, p=2).pow(2).mul(-t).exp().mean().log()
            return total_separation
            
        reconstruction_loss = reconstruction_loss(low_pass, middle_pass, high_pass, normal, low_w, mid_w, high_w)
        l2_reg_loss = 0.0
        for param in model.parameters():
            l2_reg_loss += torch.norm(param, p=2) ** 2
        total_loss = reconstruction_loss + lambda_reg * l2_reg_loss

        return total_loss

    @staticmethod
    def calc_loss(epoch, num_T, low_pass, middle_pass, high_pass, normal, model, 
                    lambda_reg=1e-2, gamma=-0.6, low_w=1.0, mid_w=1.0, high_w=1.0):
        def reconstruction_loss_raw(low_pass, middle_pass, high_pass, normal, low_w, mid_w, high_w):
            reconstructed_signal = (low_w*low_pass + mid_w*middle_pass + high_w*high_pass) / 3
            reconstruction_loss = F.mse_loss(reconstructed_signal, normal)
            return F.relu(reconstruction_loss - 1.0)

        def cross_scale_consistency_loss(low_pass, middle_pass, high_pass, gamma):
            low_middle_cosine = F.cosine_similarity(low_pass, middle_pass, dim=-1)
            middle_high_cosine = F.cosine_similarity(middle_pass, high_pass, dim=-1)
            cross_scale_loss = - (torch.mean(low_middle_cosine) + torch.mean(middle_high_cosine))
            return F.relu(cross_scale_loss - gamma)

        def reconstruction_loss(low_pass, middle_pass, high_pass, normal, low_w, mid_w, high_w):
            device = low_pass.device
            N, D = low_pass.shape
            concatenated = torch.cat([low_pass, middle_pass, high_pass], dim=-1) # [N, 3*D]
            mean_abs = torch.abs(concatenated).mean(dim=1, keepdim=True) # [N, 1]

            part1 = torch.abs(concatenated[:, :D]).mean(dim=1, keepdim=True) # [N, 1]
            part2 = torch.abs(concatenated[:, D:2*D]).mean(dim=1, keepdim=True) # [N, 1]
            part3 = torch.abs(concatenated[:, 2*D:]).mean(dim=1, keepdim=True) # [N, 1]
            attention_scores = torch.cat([part1, part2, part3], dim=1) # [N, 3]
            attention_weights = F.softmax(attention_scores, dim=-1) # [N, 3]

            fused_signal = (attention_weights.unsqueeze(-1) *
                            torch.stack([low_pass, middle_pass, high_pass], dim=1)).sum(dim=1) # [N, D]
            reconstruction_loss_val = F.mse_loss(fused_signal, normal)

            return F.relu(reconstruction_loss_val - 1.0)

        def indicator(z, t = 2):
            num_sample = 5
            num = z.shape[0]
            p = torch.ones(num)
            index = p.multinomial(num_samples=num_sample, replacement=True)
            z_sample = z[index]
            total_separation = -torch.pdist(z_sample, p=2).pow(2).mul(-t).exp().mean().log()
            return total_separation
            
        reconstruction_loss = reconstruction_loss(low_pass, middle_pass, high_pass, normal, low_w, mid_w, high_w)
        cross_scale_loss = cross_scale_consistency_loss(low_pass, middle_pass, high_pass, gamma)
        l2_reg_loss = 0.0
        for param in model.parameters():
            l2_reg_loss += torch.norm(param, p=2) ** 2
        total_loss = reconstruction_loss + cross_scale_loss + lambda_reg * l2_reg_loss

        quality = (indicator(low_pass) + indicator(middle_pass) + indicator(high_pass)) / 3

        return total_loss, quality
