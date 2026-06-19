from torchvision import transforms

def get_train_transform():
	return transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.RandomHorizontalFlip(),
		transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
		transforms.RandomGrayscale(p=0.1),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

def get_test_transform():
	return transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])