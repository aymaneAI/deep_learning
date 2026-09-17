import os
from torchvision import datasets
from torch.utils.data import DataLoader
# Set num_workers hyperparameter
NUM_WORKERS = os.cpu_count()
def create_dataloaders(train_dir,
                       test_dir,
                       transform,
                       batch_size,
                       num_workers = NUM_WORKERS):
  # Use ImageFolder to create datasets
  train_data = datasets.ImageFolder(root = train_dir, transform = transform)
  test_data  = datasets.ImageFolder(root = test_dir, transform = transform)
  # Get the class names
  class_names = train_data.classes
  # Turn the datasets into dataloaders
  train_dataloader = DataLoader(dataset = train_data,
                                batch_size = batch_size,
                                num_workers = NUM_WORKERS,
                                shuffle = True)
  test_dataloader = DataLoader(dataset = test_data,
                               batch_size = batch_size,
                               num_workers = NUM_WORKERS,
                               shuffle = False)
  return train_dataloader, test_dataloader, class_names
