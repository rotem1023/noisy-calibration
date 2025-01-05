import os
import numpy as np
import pandas as pd
import torch
from pseudo_labels_maker import *
import pickle


class InputData:   
    '''
    Class that holds the input data for calibration
    '''

    def __init__(self, data_type, logits, noisy_labels, pseudo_labels, labels, transtion_matrix, n_classes, dist_closest_center, relevant_indexes, confidence_pl):
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
        self.dist_closest_center = dist_closest_center

        self.relevant_indexes = relevant_indexes
        self.confidence_pl = confidence_pl


class NoisyInputData:
    '''
    Class that holds the input data for calibration
    '''

    def __init__(self, data_type, logits, noisy_labels, labels, transition_matrix, opt_transition_matrix, n_classes, relevant_indexes, confidence_pl):
        self.data_type = data_type
        # logits of the model
        self.logits = logits
        # original noisy labels
        self.noisy_labels = noisy_labels

        # true labels used for evaluation
        self.labels = labels
        # tranistion matrix from noisy labels to labels
        self.transition_matrix = transition_matrix
        self.opt_transition_matrix = opt_transition_matrix
        self.n_classes = n_classes

        self.relevant_indexes = relevant_indexes
        self.confidence_pl = confidence_pl
        
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


def tmp(labels, noisy_labels, labels_count, k):
    indexes = np.where(labels_count > k)[0]
    noisy_labels = noisy_labels[indexes]
    labels = labels[indexes]
    return (labels == noisy_labels).sum() / len(labels)


def read_pickle_file(file_path):
    """
    Reads a pickle file and returns its content.

    Parameters:
    file_path (str): Path to the pickle file.

    Returns:
    object: The content of the pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def _load_data(dataset, model, accuracy, data_type):
    data_dir = _get_data_dir()
    dataset_dir = f'{data_dir}/{dataset}'
    dataset_acc_dir = f'{dataset_dir}/{accuracy}'
    data = read_pickle_file(f"{dataset_acc_dir}/model_{model}_noisy_training_{data_type}_acc_{accuracy}.pickle")
    labels = data['labels']
    if 'logits' in data:
        logits = data['logits']
    else:
        logits = data['loguts']
    preds = data['preds']

    noisy_labels = data['noisyLabels']
    transition_matrix = data['matrix']
    n_classes= len(transition_matrix)
    opt_transition_matrix = _create_transition_matrix(n_classes, labels, noisy_labels)
    features_map = np.load(f'{dataset_dir}/features_map/{data_type}_features_map_22k.npy')
    relevant_indexes = generate_strong_pl_indexes(features_map, noisy_labels)
    conf_pl = generate_pseudo_labels_confidence(features_map, noisy_labels)

    print(f"accuracy noisy_labels  = {sum(noisy_labels == labels)/len(labels)}")
    print(f"accuracy model= {sum(preds == labels)/len(labels)}")
    return NoisyInputData(data_type = data_type, logits=torch.from_numpy(logits), noisy_labels=torch.from_numpy(noisy_labels),
                     labels=labels, transition_matrix=transition_matrix, opt_transition_matrix=opt_transition_matrix,
                     n_classes= n_classes, relevant_indexes=relevant_indexes, confidence_pl=conf_pl)



# def _load_data(dataset, data_type, twenty_two = True):
#     data_dir = _get_data_dir()
#     dataset_dir = f'{data_dir}/{dataset}'
#     data_type_dir = f'{dataset_dir}/{data_type}'
#     labels = np.load(f'{data_type_dir}/{data_type}_labels.npy')
#     noisy_labels = np.load(f'{data_type_dir}/{data_type}_noisy_labels.npy')
#     logits = _read_torch_file_as_np(f'{data_type_dir}/{data_type}_logits.pt')
#     # logits = np.load(f'{data_type_dir}/{data_type}_logits.npy')
#     predictions = np.argmax(logits, axis=1)
#     feature_map_extension = '22k' if twenty_two else '1k'
#     features_map = np.load(f'{data_type_dir}/{data_type}_features_map_{feature_map_extension}.npy')
#     n_classes = len(torch.unique(torch.from_numpy(labels)))
#     pseudo_labels, dist_closest_center = generate_pseudo_labels(predictions, features_map, n_classes)
#     # pseudo_labels = generate_pseudo_labels_kmeans(predictions, features_map, n_classes)
#     tranistion_matrix = _create_transition_matrix(n_classes, labels, noisy_labels)
#     print(f'{dataset} {data_type} acc noisy labels: {sum(labels==noisy_labels)/len(labels)}')
#     print(f'{dataset} {data_type} acc pseudo labels: {sum(labels==pseudo_labels)/len(labels)}')
#     print(f'{dataset} {data_type} acc preds: {sum(labels==predictions)/len(labels)}')
#     labels_check = np.squeeze(np.load(f'{data_type_dir}/{data_type}_labels_check.npy'))
#     print(f'{dataset} {data_type} check: {sum(labels==labels_check)/len(labels)}')
#     relevant_indexes = generate_strong_pl_indexes(features_map, noisy_labels)
#     conf_pl = generate_pseudo_labels_confidence(features_map, noisy_labels)
#     print(f'{dataset} {data_type} acc relevant indexes: {sum(labels[relevant_indexes]==noisy_labels[relevant_indexes])/len(relevant_indexes)}')
#     return InputData(data_type = data_type, logits=torch.from_numpy(logits), noisy_labels=torch.from_numpy(noisy_labels),
#                      pseudo_labels=pseudo_labels, labels=torch.from_numpy(labels), transtion_matrix=tranistion_matrix,
#                      n_classes= n_classes, dist_closest_center=dist_closest_center, relevant_indexes=relevant_indexes, confidence_pl=conf_pl)



def load_test_data(dataset, model, accuracy):
    return _load_data(dataset, model, accuracy, 'test')

def load_valid_data(dataset, model, accuracy):
    return _load_data(dataset, model, accuracy, 'valid')

# if __name__ == '__main__':
#     test_data = load_test_data('mnist-10')
#     valid_data = load_valid_data('mnist-10')
    
    