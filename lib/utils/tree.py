class LeafNode:
	def __init__(self, index: int, sample_id: str, patient_id: str, visit_id: str, embedding):
		self.id = "leaf_node"
		self.index = index
		self.sample_uuid: str = sample_id
		self.patient_uuid: str = patient_id
		self.visit_uuid: str = visit_id
		self.embedding = embedding

class TreeNode:
	def __init__(self, index: int, embedding = None):
		self.id = "tree_node"
		self.index = index
		self.branches:list[TreeNode | LeafNode] = []
		self.embedding = embedding
		self.quotas = 0

	def append_child(self, node):
		self.branches.append(node)

	def assign_quota(self, quota: int):
		self.quotas = quota
