import torch
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
def train_one_epoch(model, loader, criterion, optimizer, device):
	model.train()
	total_loss = 0

	for images, labels in loader:
		images, labels = images.to(device), labels.to(device)

		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()

	return total_loss/len(loader)

def evaluate(model, loader, device, multiclass=False):
	model.eval()
	all_probs, all_labels, all_preds = [], [], []

	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device)
			logits = model(images)

			if multiclass:
				probs = torch.softmax(logits, dim=1).cpu().numpy()
				preds = np.argmax(probs, axis=1)

				all_probs.append(probs)
				all_preds.append(preds)
				all_labels.append(labels.numpy())

			else:
				probs = torch.sigmoid(logits).cpu().numpy()

				all_probs.append(probs)
				all_labels.append(labels.numpy())

	if multiclass:
		all_probs = np.concatenate(all_probs)
		all_labels = np.concatenate(all_labels)
		all_preds = np.concatenate(all_preds)

		metrics = {
				"f1": f1_score(all_labels, all_preds, average="weighted"),
				"accuracy": accuracy_score(all_labels, all_preds)
		}

	else:
		all_probs = np.vstack(all_probs)
		all_labels = np.vstack(all_labels)

		metrics = {
				"mAP": average_precision_score(all_labels, all_probs, average="macro"),
				"mAP_weighted": average_precision_score(all_labels, all_probs, average="weighted"),
				"mAP_micro": average_precision_score(all_labels, all_probs, average="micro"),
		}

	return metrics