import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage  # Ensure this is imported
from sklearn.model_selection import train_test_split



class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.to_pil = ToPILImage()  # Initialize ToPILImage here

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = np.array(image)  # Ensure image is a NumPy array
        image = self.to_pil(image)  # Convert to PIL Image
        image = image.convert('RGB')  # Convert to RGB

        if self.transform:
            image = self.transform(image)

        return image, label

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())   

        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)/255.0
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)   
    
def _get_cur_file_path():
    return f'{os.path.dirname(os.path.abspath(__file__))}'
 
def get_loaders(batch_size, norm_mean= (0.229, 0.224, 0.225), norm_std = (0.485, 0.456, 0.406)):
    cur_dir = _get_cur_file_path()
    training_images_filepath = join(cur_dir, 'train-images-idx3-ubyte')
    training_labels_filepath = join(cur_dir, 'train-labels-idx1-ubyte')
    test_images_filepath = join(cur_dir, 't10k-images-idx3-ubyte')
    test_labels_filepath = join(cur_dir, 't10k-labels-idx1-ubyte')
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    
    num_test_samples = len(x_test)
    num_val_samples = int(num_test_samples * 0.5)
    num_test_samples = num_test_samples - num_val_samples

    # Create validation and test datasets
    # x_val, y_val = x_test[:num_val_samples], y_test[:num_val_samples]
    # x_test, y_test = x_test[num_val_samples:], y_test[num_val_samples:]
    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=0.5, stratify=y_test, random_state=42
    )
    

    # Define the transformation for training images
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # Define the transformation for validation images
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # Create custom datasets with transformations
    train_dataset = MNISTDataset(x_train, y_train, transform=train_transform)
    train_eval_dataset = MNISTDataset(x_train, y_train, transform=val_transform)
    val_dataset = MNISTDataset(x_val, y_val, transform=val_transform)
    test_dataset = MNISTDataset(x_test, y_test, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, train_eval_loader, val_loader, test_loader
    
    
