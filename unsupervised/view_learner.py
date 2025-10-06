import torch
from torch.nn import Sequential, Linear, ReLU


class ViewLearner(torch.nn.Module):
	def __init__(self, encoder, mlp_edge_model_dim=64):
		super(ViewLearner, self).__init__()

		self.encoder = encoder
		self.input_dim = self.encoder.out_node_dim
		self.x_aug_dim = 0

		self.mlp_edge_model = Sequential(
			Linear(self.input_dim * 2, mlp_edge_model_dim),
			ReLU(),
			Linear(mlp_edge_model_dim, 1)
		)

		self.mlp_x_aug_model = Sequential(
			Linear(self.input_dim * 2 + self.x_aug_dim, mlp_edge_model_dim),
			ReLU(),
			Linear(mlp_edge_model_dim, 1)
		)

		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, x_aug, batch, x, edge_index, edge_attr):

		_, node_emb = self.encoder(batch, x, edge_index, edge_attr)
		src, dst = edge_index[0], edge_index[1]
		emb_src = node_emb[src]
		emb_dst = node_emb[dst]
		edge_emb = torch.cat([emb_src, emb_dst], 1)
		self.x_aug_dim = x_aug.shape[1]
		# edge_logits = self.mlp_edge_model(edge_emb)
		edge_logits = self.mlp_x_aug_model(edge_emb)
		return edge_logits