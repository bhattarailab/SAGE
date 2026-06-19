"""
Trains the resenet and vit on three datasets GastroVision, HyperKvasir, and SAGE for multilabel classification
"""
from argparse import ArgumentParser
import torch
import random
import os
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models
from dotenv import load_dotenv
from pathlib import Path
import wandb

from lib.trainer.sage import SAGEMultiLabelDataset
from lib.trainer.utils import train_one_epoch, evaluate
from lib.trainer.transforms import get_train_transform, get_test_transform
from lib.trainer.loss import FocalLossWithWeights

load_dotenv()

SEED = 0
BATCH_SIZE = 32
LR = 0.01
EPOCHS = 100
MODELS = ["resnet", "densenet", "hyperkvasir-resnet", "hyperkvasir-densenet"]
DATASETS = ["hyperkvasir", "SAGE"]

# setting the inital seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

def prepare_multilabel(dataset, model_name, device):
	mlb = MultiLabelBinarizer()

	def transform_y(label):
		return mlb.transform([label])[0]

	if dataset == "SAGE":
		dataset_df = pd.read_csv("./outputs/complete-dataset/metadata.csv")
		labels_df = pd.read_csv("./outputs/sample_labels.csv")

		# assign the file_path and split from sample_uuid in dataset
		labels_df = labels_df.merge(
    	dataset_df[["sample_uuid", "file_path", "split"]],
    	on="sample_uuid",
    	how="left"
		)
		labels_df["labels"] = labels_df["labels"].apply(ast.literal_eval)
		labels_df = labels_df[labels_df["labels"].str.len() > 0].reset_index(drop=True)
		train_df = labels_df[labels_df["split"] == "train"]
		test_df = labels_df[labels_df["split"] == "test"]

		mlb.fit(train_df["labels"])
		n_classes = len(mlb.classes_)

		train_transform, test_transform = get_train_transform(), get_test_transform()
	
		train_set = SAGEMultiLabelDataset(train_df, transform=train_transform, transform_y=transform_y)
		test_set = SAGEMultiLabelDataset(test_df, transform=test_transform ,transform_y=transform_y)
	elif dataset == "hyperkvasir":
		# hyperkvasir data download path
		labeled_images_base_path = Path("/scratch/noli1/labeled-images/")

		dataset_df = pd.read_csv(labeled_images_base_path/"image-labels.csv")
		dataset_df["file_path"] = dataset_df.apply(
			lambda row: os.path.join(labeled_images_base_path, "lower-gi-tract" if row["Organ"] == "Lower GI" else "upper-gi-tract", row["Classification"], row["Finding"], f"{row['Video file']}.jpg"), axis=1
		)
		dataset_df["labels"] = dataset_df["Finding"].apply(lambda x: [x])
		
		train_df, test_df = train_test_split(dataset_df, test_size=0.2, random_state=SEED, stratify=dataset_df["labels"])
		mlb.fit(train_df["labels"])
		n_classes = len(mlb.classes_)
		
		train_transform, test_transform = get_train_transform(), get_test_transform()

		train_set = SAGEMultiLabelDataset(df=train_df, transform=train_transform, transform_y=transform_y)
		test_set = SAGEMultiLabelDataset(df=test_df, transform=test_transform, transform_y=transform_y)

	return {
		"train_set": train_set,
		"test_set": test_set,
		"pos_weight": None,
		"n_classes": n_classes,
		"classes": mlb.classes_,
	}
	
def prepare_multiclass(dataset, model_name, device):
	le = LabelEncoder()

	def transform_y(class_name):
		return le.transform([class_name])[0]
	
	if dataset == "SAGE":
		dataset_df = pd.read_csv("./outputs/complete-dataset/metadata.csv")
		labels_df = pd.read_csv("./outputs/sample_labels.csv")

		# assign the file_path and split from sample_uuid in dataset
		labels_df = labels_df.merge(
    	dataset_df[["sample_uuid", "file_path", "split"]],
    	on="sample_uuid",
    	how="left"
		)
		labels_df["labels"] = labels_df["classes"]
		labels_df = labels_df[labels_df["classes"].str.len() > 0].reset_index(drop=True)
		train_df = labels_df[labels_df["split"] == "train"]
		test_df = labels_df[labels_df["split"] == "test"]

		le.fit(train_df["classes"])
		n_classes = len(le.classes_)
		
		train_transform, test_transform = get_train_transform(), get_test_transform()
	
		train_set = SAGEMultiLabelDataset(train_df, transform=train_transform,transform_y=transform_y, target_type=torch.long)
		test_set = SAGEMultiLabelDataset(test_df, transform=test_transform ,transform_y=transform_y, target_type=torch.long)
	elif dataset == "hyperkvasir":
		# hyperkvasir data download path
		labeled_images_base_path = Path("/scratch/noli1/labeled-images/")

		dataset_df = pd.read_csv(labeled_images_base_path/"image-labels.csv")
		dataset_df["file_path"] = dataset_df.apply(
			lambda row: os.path.join(labeled_images_base_path, "lower-gi-tract" if row["Organ"] == "Lower GI" else "upper-gi-tract", row["Classification"], row["Finding"], f"{row['Video file']}.jpg"), axis=1
		)
		dataset_df["labels"] = dataset_df["Finding"]
		
		train_df, test_df = train_test_split(dataset_df, test_size=0.2, random_state=SEED, stratify=dataset_df["labels"])
		le.fit(train_df["labels"])
		n_classes = len(le.classes_)
		
		train_transform, test_transform = get_train_transform(), get_test_transform()

		train_set = SAGEMultiLabelDataset(df=train_df, transform=train_transform, transform_y=transform_y, target_type=torch.long)
		test_set = SAGEMultiLabelDataset(df=test_df, transform=test_transform, transform_y=transform_y, target_type=torch.long)

	return {
		"train_set": train_set,
		"test_set": test_set,
		"pos_weight": None,
		"n_classes": n_classes,
		"classes": le.classes_,
	}

def main(model_name, run, dataset, device, lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS, multiclass=False):
	if multiclass:
		items = prepare_multiclass(dataset=dataset, model_name=model_name, device=device)
	else:
		items = prepare_multilabel(dataset=dataset, model_name=model_name, device=device)

	train_set, test_set, pos_weight, n_classes, classes = items["train_set"], items["test_set"], items["pos_weight"], items["n_classes"], items["classes"]
	
	print(f"Loaded dataset {dataset} with train = {len(train_set)} and test = {len(test_set)}")
	print(f"Classes: {classes}")
	print(f"Number of classes = {n_classes}")

	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

	if model_name == "resnet":
		model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
		model.fc = nn.Linear(model.fc.in_features, n_classes)
	elif model_name == "densenet":
		model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
		model.classifier = nn.Linear(model.classifier.in_features, n_classes)
	elif model_name == "hyperkvasir-resnet":
		model = models.resnet50()
		model.fc = nn.Linear(model.fc.in_features, 23)
		model.load_state_dict(torch.load("./outputs/checkpoints/hyperkvasir-resnet.pth"))
		model.fc = nn.Linear(model.fc.in_features, n_classes)
	elif model_name == "hyperkvasir-densenet":
		model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
		model.classifier = nn.Linear(model.classifier.in_features, 23)
		model.load_state_dict(torch.load("./outputs/checkpoints/hyperkvasir-densenet.pth"))
		model.classifier = nn.Linear(model.classifier.in_features, n_classes)

	model = model.to(device)
	if not multiclass:
		criterion = FocalLossWithWeights()
	else:
		criterion = nn.CrossEntropyLoss(weight=pos_weight)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

	# training loop
	for epoch in range(epochs):
		train_loss=train_one_epoch(model=model, loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
		scheduler.step()

		train_metric = evaluate(model=model, loader=train_loader, device=device, multiclass=multiclass)
		test_metric = evaluate(model=model, loader=test_loader, device=device, multiclass=multiclass)

		run.log({
			"train": train_metric,
			"train_loss": train_loss,
			"test": test_metric,
		})
		print(f"Completed {epoch}")
		
	torch.save(model.state_dict(), f"./outputs/checkpoints/{dataset}-{model_name}.pth")
	
	run.finish()

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--model", required=True, choices=MODELS)
	parser.add_argument("--dataset", required=True, choices=DATASETS)
	parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
	parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
	parser.add_argument("--lr", default=LR, type=float)
	parser.add_argument("--epochs", default=EPOCHS, type=int)
	parser.add_argument("--multi-class", action="store_true")
	parser.add_argument("--name", required=True)

	args = parser.parse_args()

	run = wandb.init(
		name=args.name,
		entity="xxx",
		project="SAGE Validation",
		config={
			"learning_rate": args.lr,
			"model": args.model,
			"dataset": args.dataset,
			"multiclass": args.multi_class,
		}
	)

	main(
		model_name=args.model,
		dataset=args.dataset,
		device=args.device,
		batch_size=args.batch_size,
		lr=args.lr,
		epochs=args.epochs,
		multiclass=args.multi_class,
		run=run
	)