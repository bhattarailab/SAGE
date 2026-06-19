import torch
from torch.utils.data import Dataset
from PIL import Image

class SAGEMultiLabelDataset(Dataset):
	def __init__(self, df, transform=None, transform_y=None, target_type=torch.float32):
		self.images = [Image.open(path).convert("RGB") for path in df["file_path"].values]
		self.labels = df["labels"].values
		self.transform = transform
		self.transform_y = transform_y
		self.target_type = target_type

	def __len__(self):
		return len(self.images)
	
	def __getitem__(self, index):
		image = self.images[index]
		if self.transform:
			image = self.transform(image)

		label = self.labels[index]
		if self.transform_y:
			label = self.transform_y(label)

		label = torch.tensor(label, dtype=self.target_type)

		return image, label
