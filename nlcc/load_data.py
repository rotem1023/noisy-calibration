import os
import numpy as np
import torch
from pseudo_labels_maker import generate_pseudo_labels

class InputData:   
    '''
    Class that holds the input data for calibration
    '''

    def __init__(self, data_type, logits, noisy_labels, pseudo_labels, labels, transtion_matrix, n_classes):
        self.data_type = data_type
        # logits of the model
        self.logits = logits
        # original noisy labels
        self.noisy_labels = noisy_labels
        # pseudo labels generated in the first stage of the algorithm
        self.pseudo_labels = pseudo_labels
        # true labels used for evaluation
        self.labels = labels
        # tranistion matrix from noisy labels to labels
        self.transtion_matrix  = transtion_matrix
        self.n_classes = n_classes
        
def _get_cur_file_path():
    return f'{os.path.dirname(os.path.abspath(__file__))}'


def _get_data_dir():
    return f'{_get_cur_file_path()}/data'


def _read_torch_file_as_np(filepath):
    data = torch.load(filepath)

    # If the tensor is on the GPU, move it to the CPU before converting to NumPy
    if data.is_cuda:
        data = data.cpu()

    # Convert to NumPy array
    data_np = data.numpy()
    return data_np

def _create_transition_matrix(n_classes, labels, noisy_labels):
    """
    Create a normalized transition matrix from noisy labels and true labels
    :param n_classes: number of classes
    :param labels: true labels
    :param noisy_labels: noisy labels
    :return: normalized confusion matrix
    """
    cm = np.zeros((n_classes, n_classes))

    labels_np = labels
    noisy_labels_np = noisy_labels
    # Populate the confusion matrix
    for i in range(labels_np.shape[0]):
        cm[int(labels_np[i])][int(noisy_labels_np[i])] += 1

    # Normalize each row
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = cm / row_sums
    # Handle cases where row_sums might be zero to avoid division by zero
    cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)

    return cm_normalized

def _load_data(dataset, data_type, twenty_two = True):
    data_dir = _get_data_dir()
    dataset_dir = f'{data_dir}/{dataset}'
    data_type_dir = f'{dataset_dir}/{data_type}'
    labels = np.load(f'{data_type_dir}/{data_type}_labels.npy')
    noisy_labels = np.load(f'{data_type_dir}/{data_type}_noisy_labels.npy')
    # logits = _read_torch_file_as_np(f'{data_type_dir}/{data_type}_logits.pt')
    logits = np.load(f'{data_type_dir}/{data_type}_logits.npy')
    feature_map_extension = '22k' if twenty_two else '1k'
    features_map = np.load(f'{data_type_dir}/{data_type}_features_map_{feature_map_extension}.npy') 
    n_classes = len(torch.unique(torch.from_numpy(labels)))
    pseudo_labels = generate_pseudo_labels(noisy_labels, features_map, n_classes)
    tranistion_matrix = _create_transition_matrix(n_classes, labels, noisy_labels)
    print(f'{data_type} acc noisy labels: {sum(labels==noisy_labels)/len(labels)}')
    print(f'{data_type} acc pseudo labels: {sum(labels==pseudo_labels)/len(labels)}')
    labels_check = np.squeeze(np.load(f'{data_type_dir}/{data_type}_labels_check.npy'))
    print(f'{data_type} check: {sum(labels==labels_check)/len(labels)}')
    return InputData(data_type = data_type, logits=torch.from_numpy(logits), noisy_labels=torch.from_numpy(noisy_labels),
                     pseudo_labels=pseudo_labels, labels=torch.from_numpy(labels), transtion_matrix=tranistion_matrix,
                     n_classes= n_classes)
    
    
    

def load_test_data(dataset, twenty_two = True):
    return _load_data(dataset, 'test', twenty_two)

def load_valid_data(dataset, twenty_two = True):
    return _load_data(dataset, 'valid', twenty_two)

# if __name__ == '__main__':
#     test_data = load_test_data('mnist-10')
#     valid_data = load_valid_data('mnist-10')
    
    