"""
Trains the model by converting hyperkvasir to our label sapce and test it on our dataset
"""
from argparse import ArgumentParser
import torch
import random
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models
from dotenv import load_dotenv
from pathlib import Path

from lib.trainer.sage import sageMultiLabelDataset
from lib.trainer.utils import train_one_epoch, evaluate
from lib.trainer.transforms import get_train_transform, get_test_transform

load_dotenv()

SEED = 0
BATCH_SIZE = 32
LR = 0.01
EPOCHS = 100
MODELS = ["resnet", "densenet", "hyperkvasir-resnet"]
DATASETS = ["hyperkvasir", "sage"]

# setting the inital seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

def train(model, train_set, test_set, device, n_classes, lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS):
  print(f"Loaded dataset with train = {len(train_set)} and test = {len(test_set)}")
  print(f"Number of classes = {n_classes}")

  # create the dataloaser
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

  criterion = nn.CrossEntropyLoss()

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

  # training loop
  best_f1 = 0.0
  for epoch in range(epochs):
    train_loss = train_one_epoch(model=model, loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
    scheduler.step()

    train_metric = evaluate(model=model, loader=train_loader, device=device, multiclass=True)
    test_metric = evaluate(model=model, loader=test_loader, device=device, multiclass=True)

    print(f"Epoch = {epoch+1}: train_loss={train_loss:.6f}, train f1={train_metric['f1']:.6f}, test_f1={test_metric['f1']:.6f}")
    if test_metric["f1"] > best_f1:
      torch.save(model.state_dict(), "./outputs/temp.pth")
      best_f1 = test_metric["f1"]

  test_metric = evaluate(model=model, loader=test_loader, device=device, multiclass=True)
  print("\n\nTraining Complete")
  print(test_metric)

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--model", required=True, choices=MODELS)
  parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
  parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
  parser.add_argument("--lr", default=LR, type=float)
  parser.add_argument("--epochs", default=EPOCHS, type=int)

  args = parser.parse_args()

  # conversion of hyperkvasir label sapce to ours
  HYPERKVASIR_2_sage = {
    "cecum": None,
    "ileum": None,
    "retroflex-rectum": "rectum-retroflexion",
    "hemorrhoids": None,
    "polyps": "polyp",
    "ulcerative-colitis-grade-0-1": None,
    "ulcerative-colitis-grade-1": None,
    "ulcerative-colitis-grade-1-2": None,
    "ulcerative-colitis-grade-2": None,
    "ulcerative-colitis-grade-2-3": None,
    "ulcerative-colitis-grade-3": None,
    "bbps-0-1": None,
    "bbps-2-3": None,
    "impacted-stool": None,
    "dyed-lifted-polyps": None,
    "dyed-resection-margins": None,
    "pylorus": "pylorus-normal",
    "retroflex-stomach": "fundus-retroflexion",
    "z-line": "z-line-normal",
    "barretts": None,
    "barretts-short-segment": None,
    "esophagitis-a": "esophagitis",
    "esophagitis-b-d": "esophagitis",
  }

  # loading the hyperkvasir dataset with re-relabelling
  labeled_images_base_path = Path("/scratch/noli1/labeled-images/")
  hyperkvasir_dataset_df = pd.read_csv(labeled_images_base_path/"image-labels.csv")
  hyperkvasir_dataset_df["file_path"] = hyperkvasir_dataset_df.apply(
    lambda row: os.path.join(labeled_images_base_path, "lower-gi-tract" if row["Organ"] == "Lower GI" else "upper-gi-tract", row["Classification"], row["Finding"], f"{row['Video file']}.jpg"), axis=1
  )
  hyperkvasir_dataset_df["labels"] = hyperkvasir_dataset_df["Finding"].apply(lambda x: HYPERKVASIR_2_sage[x])

  # remove those entries with all lables no
  hyperkvasir_dataset_df = hyperkvasir_dataset_df[~hyperkvasir_dataset_df["labels"].isna()]
  
  # encoding the label
  le = LabelEncoder()
  le.fit(hyperkvasir_dataset_df["labels"])
  n_classes = len(le.classes_)
  
  def transform_y(label):
    return le.transform([label])[0]
  
  # dataset splitting
  train_df, test_df = train_test_split(hyperkvasir_dataset_df, test_size=0.2, random_state=SEED, stratify=hyperkvasir_dataset_df["labels"])
  
  train_transform, test_transform = get_train_transform(), get_test_transform()

  # prepare the tran and test set for hyperkavasir
  hyperkvasir_train_set = sageMultiLabelDataset(df=train_df, transform=train_transform, transform_y=transform_y, target_type=torch.long)
  hyperkvasir_test_set = sageMultiLabelDataset(df=test_df, transform=test_transform, transform_y=transform_y, target_type=torch.long)

  # prepare sage dataset
  sage_dataset_df = pd.read_csv("./outputs/complete-dataset/metadata.csv")
  sage_labels_df = pd.read_csv("./outputs/sample_labels.csv")

  # assign the file_path and split from sample_uuid in dataset
  sage_labels_df = sage_labels_df.merge(
    sage_dataset_df[["sample_uuid", "file_path", "split"]],
    on="sample_uuid",
    how="left"
  )
  sage_labels_df = sage_labels_df[sage_labels_df["classes"].isin(le.classes_)]
  sage_labels_df["labels"] = sage_labels_df["classes"]
  train_df = sage_labels_df[sage_labels_df["split"] == "train"]
  test_df = sage_labels_df[sage_labels_df["split"] == "test"]

  sage_train_set = sageMultiLabelDataset(train_df, transform=train_transform,transform_y=transform_y, target_type=torch.long)
  sage_test_set = sageMultiLabelDataset(test_df, transform=test_transform ,transform_y=transform_y, target_type=torch.long)

  # prepare the model
  if args.model== "resnet":
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
  elif args.model== "densenet":
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, n_classes)

  model = model.to(args.device)

  # train on hyperkvasir and test on hyperkvasir
  train(model=model, train_set=hyperkvasir_train_set, test_set=hyperkvasir_test_set, device=args.device, n_classes=n_classes, lr=args.lr, batch_size=args.batch_size, epochs=args.epochs)

  model.load_state_dict(torch.load("./outputs/temp.pth"))

  sage_test_loader = DataLoader(sage_test_set, batch_size=args.batch_size, shuffle=False)
  metrics = evaluate(model, sage_test_loader, args.device, multiclass=True)
  print("/n/SAGE Performance Metrics")
  print(metrics)
