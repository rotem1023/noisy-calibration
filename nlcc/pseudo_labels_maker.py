import torch
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans


def _normalize_unique_counts(labels):
    results = []
    labels, counts = np.unique(labels, return_counts=True)
    i = 0
    cur_index = 0
    while(i<= labels[-1]):
        if i== labels[cur_index]:
            results.append(counts[cur_index])
            cur_index +=1
        else:
            results.append(0)
        i+=1
    return np.array(results)


def normalize_features(all_fea):
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


def _extract_i_th_distance(dist, i):
    if i < 1 or i > dist.size(1):
        raise ValueError("i must be between 1 and the number of centers (inclusive)")

    _, indices = torch.topk(dist, k=i, dim=1, largest=False)
    pseudo_labels = indices[:, -1]  # Select the i-th smallest distance label for each point
    dist_i = dist.gather(1, indices[:, -1].unsqueeze(1)).squeeze(1)
    return dist_i


def _calc_pairwize_distance(features):
    features = torch.from_numpy(features)
    diffs = features.unsqueeze(1) - features.unsqueeze(0)

    # Compute squared distances and take the square root to get Euclidean distances
    distances = torch.norm(diffs, dim=-1)
    return distances

def _calc_pairwize_distance(features):
    features = torch.from_numpy(features)

    # Compute squared norms for all rows
    norms = torch.sum(features**2, dim=1, keepdim=True)  # Shape: (N, 1)

    # Compute pairwise squared distances
    squared_distances = norms + norms.T - 2 * features @ features.T

    # Avoid numerical instability (e.g., due to negative values from precision issues)
    squared_distances = torch.clamp(squared_distances, min=0.0)

    # Compute Euclidean distances
    distances = torch.sqrt(squared_distances)
    return distances

def _find_agree_close_neighbors(distances, labels):
    n = len(labels)
    unique_counts = np.unique(labels, return_counts=True)[1]
    sorted_distances, sorted_indices = torch.sort(distances, dim=1)
    # Count closest neighbors with the same label
    same_label_counts = []
    for i in range(len(labels)):
        # Get the sorted indices and labels for neighbors
        neighbor_indices = sorted_indices[i, 1:]  # Exclude self (index 0)
        neighbor_labels = labels[neighbor_indices]

        # Count neighbors with the same label as the current vector
        count = 0
        for label in neighbor_labels:
            if label == labels[i]:  # Check if the label matches
                count += 1
            else:
                break  # Stop when a different label is encountered
        same_label_counts.append(count* (1-(unique_counts[labels[i]]/n)))

    same_label_counts = torch.tensor(same_label_counts)
    return same_label_counts.numpy()


def _find_agree_from_close_neighbors(distances, labels):
    n = len(labels)-1 # exclude self
    unique_counts = _normalize_unique_counts(labels)
    sorted_distances, sorted_indices = torch.sort(distances, dim=1)
    # Count closest neighbors with the same label
    same_label_counts = []
    for i in range(len(labels)):
        # Get the sorted indices and labels for neighbors
        neighbor_indices = sorted_indices[i, 1:]  # Exclude self (index 0)
        neighbor_labels = labels[neighbor_indices]
        current_label = labels[i]

        # Count neighbors with the same label as the current vector
        count = 0
        for j in range(unique_counts[current_label]):
            if neighbor_labels[j] == current_label:  # Check if the label matches
                count += 1

        cur_n = unique_counts[current_label]-1 # exclude self
        conf = count/(cur_n)
        # in case the class are very imbalanced
        if cur_n/n > 0.5:
            minus = (cur_n - (n-cur_n))/n
            conf = conf - minus
            scale = 1-minus
            conf = conf/scale
        same_label_counts.append(conf)

    same_label_counts = torch.tensor(same_label_counts)
    return same_label_counts.numpy()

def generate_agree_close_neighbors(features, labels):
    distances = _calc_pairwize_distance(features)
    closest_neighbors_agree_counts = _find_agree_close_neighbors(distances, labels)
    k = np.percentile(closest_neighbors_agree_counts, 50)
    indexes = np.where(closest_neighbors_agree_counts > k)[0]
    return torch.from_numpy(indexes)


def generate_strong_pl_indexes(features, labels):
    distances = _calc_pairwize_distance(features)
    arr =  _find_agree_from_close_neighbors(distances, labels)
    k = np.percentile(arr, 50)
    indexes = np.where(arr > k)[0]
    return torch.from_numpy(indexes)

def generate_noisy_labels_confidence(features, labels):
    distances = _calc_pairwize_distance(features)
    return _find_agree_from_close_neighbors(distances, labels)




def generate_pseudo_labels(noisy_labels, features, n_classes):
    features = normalize_features(features)
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


def generate_pseudo_labels_kmeans(noisy_labels, features, n_classes):
    features = normalize_features(features)
    softmaxes = np.zeros((noisy_labels.size, n_classes))
    softmaxes[np.arange(noisy_labels.size), noisy_labels] = 1
    softmaxes = torch.from_numpy(softmaxes)
    # Calculate the centers of the classes
    centers = _create_centers(softmaxes, features, n_classes)
    kmeans = KMeans(n_clusters=n_classes, init=centers, n_init=1, random_state=0, max_iter=1)

    # Fit the model to the data
    kmeans.fit(features)
    return torch.from_numpy(kmeans.labels_)
