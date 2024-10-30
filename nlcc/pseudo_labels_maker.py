import torch
import numpy as np
from scipy.spatial import distance


def _normalize_features(all_fea):
    all_fea = torch.from_numpy(all_fea)
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    return all_fea.numpy()

def _calc_center_k(k, softmaxes, features):
    '''
    Calculate the center of class k
    :param k: class index
    :param softmaxes: model softmaxes
    :param features: values of feature extractor
    :return: center of class k
    '''
    k_softmaxes = softmaxes[:, k]
    adapted_features = k_softmaxes.unsqueeze(1) * features
    sum_adapted_features = adapted_features.sum(dim=0)
    sum_k_softmaxes = k_softmaxes.sum().item()
    if sum_k_softmaxes ==0:
        print(f"Warning: can't find noisy labels for class: {k}")
        sum_k_softmaxes = 1
    return sum_adapted_features / sum_k_softmaxes


def _create_centers(softmaxes, features, n_classes):
    '''
    Create the centers of the classes
    :param softmaxes: softmaxes of the model
    :param features: values of feature extractor
    :param n_classes: number of classes
    :return: centers of the classes
    '''
    centers = []
    for k in range(n_classes):
        centers.append(_calc_center_k(k, softmaxes, features))
    return torch.from_numpy(np.array(centers))


def _calc_dist_center(features, centers):
    '''
    Calculate the distance between each example and the centers
    :param features: values of feature extractor
    :param centers: centers of the classes
    :return: distance between each example and the centers
    '''
    def calc_distance(row):
        def calc_distance_from_center(center):
            output = distance.cosine(row, center)
            if np.isnan(output):
                return 2
            return output
        return np.apply_along_axis(calc_distance_from_center, axis=1, arr=centers)

    distances = np.apply_along_axis(calc_distance, axis=1, arr=features)
    return torch.from_numpy(distances)



def generate_pseudo_labels(noisy_labels, features, n_classes):
    features = _normalize_features(features)
    # Calculate the softmaxes of the model
    softmaxes = np.zeros((noisy_labels.size, n_classes))
    softmaxes[np.arange(noisy_labels.size), noisy_labels] = 1
    softmaxes = torch.from_numpy(softmaxes)
    
    # softmaxes = _create_one_hot_encoding(torch.from_numpy(noisy_labels))
    # Calculate the centers of the classes
    centers = _create_centers(softmaxes, features, n_classes)
    # Calculate the distance between each example and the centers
    dist = _calc_dist_center(features, centers)
    # Assign pseudo labels to the examples
    pseudo_labels = torch.argmin(dist, dim=1)
    return pseudo_labels
