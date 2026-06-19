import torch
from torch import nn

class FocalLossWithWeights(nn.Module):
	def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
		"""
		alpha: tensor of shape (n_classes,) — per-class pos weights
						computed from dataset statistics
		"""
		super().__init__()
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = reduction

	def forward(self, logits, targets):
		bce = nn.functional.binary_cross_entropy_with_logits(
				logits, targets,
				pos_weight=self.alpha,
				reduction='none'
		)
		probs = torch.sigmoid(logits)
		pt = targets * probs + (1 - targets) * (1 - probs)
		focal_weight = (1 - pt) ** self.gamma

		loss = focal_weight * bce

		if self.reduction == 'mean':
				return loss.mean()
		elif self.reduction == 'sum':
				return loss.sum()
		return loss
