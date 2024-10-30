from get_loaders import get_loaders

import argparse
import torch
import numpy as np
import os
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim


# Set a random seed for reproducibility
seed_value = 42
torch.manual_seed(seed_value)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 256)  # Adjust the dimensions based on your input size
        self.fc2 = nn.Linear(256, 9)  # Assuming 9 output classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 14 * 14)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def shuffle_data_and_labels(all_data, all_labels):
    # Generate a random permutation of indices
    indices = torch.randperm(all_data.size(0))

    # Shuffle both data and labels using the same indices
    shuffled_data = all_data[indices]
    shuffled_labels = all_labels[indices]

    return shuffled_data, shuffled_labels

            
            
def create_large_tensor(train_loader_at_eval, val_loader, test_loader):
    # Empty lists to store the data and labels
    data_list = []
    label_list = []
    index_list = []

    # Function to concatenate the data from a given loader
    def append_data(loader):
        for batch in loader:
            images, labels = batch
            data_list.append(images)
            label_list.append(labels)

    # Concatenate data from all loaders
    append_data(train_loader_at_eval)
    index_list.append(len(label_list))
    append_data(val_loader)
    index_list.append(len(label_list))
    append_data(test_loader)

    # Concatenate all data and labels into large tensors
    all_data = torch.cat(data_list, dim=0)
    all_labels = torch.cat(label_list, dim=0)

    print(f"Total data size: {all_data.size()}")
    print(f"Total label size: {all_labels.size()}")
    shuffled_data, shuffled_labels = shuffle_data_and_labels(all_data, all_labels)
    return shuffled_data, shuffled_labels


def train(model, images, labels, num_epochs, batch_size, target_acc):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Set the model to training mode
    model.train()
    
    # Calculate the number of batches
    num_samples = len(images)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division to cover all data
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(num_batches):
            # Create batches of images and labels
            start = i * batch_size
            end = min(start + batch_size, num_samples)  # Make sure we don't go out of range
            images_batch = images[start:end]
            labels_batch = labels[start:end]

            # Move batches to device (CPU or GPU)
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)
            labels_batch = labels_batch.squeeze()  # Remove extra dimensions if necessary


            # Forward pass
            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy for the current batch
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

        # Calculate overall accuracy for the epoch
        accuracy = 100 * correct / total
        
        # Print epoch details
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Early stopping condition based on target accuracy
        if accuracy >= target_acc:
            print(f"Model reached target accuracy of {target_acc}, stopping training.")
            break


def predict_and_save(model, loader, loader_name, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient calculation for prediction
        for batch in loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass to get the outputs
            outputs = model(images)

            # Get the predicted class (index of the max log-probability)
            _, preds = torch.max(outputs, 1)

            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())  # Moving to CPU for easier storage
            all_labels.extend(labels.cpu().numpy()) 

    # Convert to NumPy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels).squeeze()
    print(f"{loader_name} accuracy : {sum(all_labels==all_preds)/len(all_preds)}")

    # Save predictions and true labels as CSV
    # results = pd.DataFrame({'TrueLabel': all_labels, 'PredictedLabel': all_preds})
    # results.to_csv(f'predictions_{loader_name}.csv', index=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    outputdir = f'{current_dir}/output'
    os.makedirs(outputdir, exist_ok=True) 
    np.save(f"{outputdir}/{loader_name}_noisy_labels.npy", all_preds)
    np.save(f"{outputdir}/{loader_name}_labels.npy", all_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create feature map for pathMnist')
    parser.add_argument('--data_flag',
                        default='pathmnist',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--resize',
                        help='resize images of size 28x28 to 224x224',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='convert the grayscale image to RGB',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_name',
                        default='resnet18',
                        help='choose backbone from resnet18, resnet50',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    
    parser.add_argument('--acc',
                        default=60,
                        help='desired accuracy of noisy labels',
                        type=int)
    
    args = parser.parse_args()
    data_flag = args.data_flag
    batch_size = args.batch_size
    download = args.download
    model_name = args.model_name
    resize = args.resize
    as_rgb = args.as_rgb
    run = args.run
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = ConvNet()
    model = model.to(device)
    train_loader,train_loader_at_eval, val_loader, test_loader = get_loaders(model_name=model_name, as_rgb=as_rgb, batch_size=batch_size, data_flag= data_flag, download=download, resize=resize)
    
    x, labels = create_large_tensor(train_loader_at_eval, val_loader, test_loader)
    
    # x_tf = torch_to_tf(x)
    # labels_tf = torch_to_tf(labels)
    
    train(model, x, labels, 10, 128, args.acc)
    
    predict_and_save(model, train_loader_at_eval, 'train', device)
    predict_and_save(model, val_loader, 'valid', device)
    predict_and_save(model, test_loader, 'test', device)
    
    
    
