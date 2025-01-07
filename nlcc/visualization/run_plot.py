import sys
sys.path.append('..')
from nlcc.load_data import load_test_data, load_valid_data
import os
import plot
from nlcc.calibration_loss import CalibrationLoss
import numpy as np
import torch
from torch.nn import functional as F


def _get_cur_file_path():
    return f'{os.path.dirname(os.path.abspath(__file__))}'

def _get_data_dir():
    return f'{_get_cur_file_path()}/../data'

def _get_datasets_names():
    data_dir = _get_data_dir()
    return [f for f in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{f}')]

def _create_y_tilde_from_transition_matrix(P, y):
    n = len(y)
    classes = np.arange(len(P))
    output = np.zeros(n)
    for i in range(n):
        new_label = np.random.choice(classes, p=P[y[i]])
        output[i] = new_label
    return torch.from_numpy(output).int()

class InputData:
    def __init__(self, dataset_name, accuracy, model_name):
        self.dataset_name = dataset_name
        self.accuracy = accuracy
        self.model_name = model_name
        self.valid_input_data = load_valid_data(dataset_name, model_name, accuracy)
        self.test_input_data = load_test_data(dataset_name, model_name, accuracy)


def _get_all_datasets():
    results= []
    dataset_names= ['bloodmnist']
    acc = [85]
    models = ['vgg']
    for dataset in dataset_names:
        for a in acc:
            for model in models:
                results.append(InputData(dataset, a, model))
    return results



def cretae_tsne_plots_for_dataset(dataset):
    plot_name = "logits"

    x_valid = dataset.valid_input_data.logits
    y_valid = dataset.valid_input_data.labels
    plot.create_tsne_plot(dataset.dataset_name, x_valid, y_valid, plot_name)
    x_test = dataset.test_input_data.logits
    y_test = dataset.test_input_data.labels
    plot.create_tsne_plot(dataset.dataset_name, x_test, y_test, plot_name, test = True)

    plot_name = "pseudo_labels"
    y_valid = dataset.valid_input_data.pseudo_labels
    plot.create_tsne_plot(dataset.dataset_name, x_valid, y_valid, plot_name)
    y_test = dataset.test_input_data.pseudo_labels
    plot.create_tsne_plot(dataset.dataset_name, x_test, y_test, plot_name, test = True)

    plot_name = "noisy_labels"
    y_valid = dataset.valid_input_data.noisy_labels
    plot.create_tsne_plot(dataset.dataset_name, x_valid, y_valid, plot_name)
    y_test = dataset.test_input_data.noisy_labels
    plot.create_tsne_plot(dataset.dataset_name, x_test, y_test, plot_name, test = True)


def run_calibration(dataset, syn_pl_valid, syn_pl_test):
    calib_test = CalibrationLoss(LOGIT=True, adaECE=adaEce, n_bins=n_bins, true_labels=dataset.test_input_data.labels)
    loss_test = calib_test.forward(dataset.test_input_data.logits, dataset.test_input_data.labels, num_classes=dataset.test_input_data.n_classes)

    calib_valid = CalibrationLoss(LOGIT=True, adaECE=adaEce, n_bins=n_bins, true_labels=dataset.valid_input_data.labels)
    loss_valid = calib_valid.forward(dataset.valid_input_data.logits, dataset.valid_input_data.labels, num_classes=dataset.valid_input_data.n_classes)

    calib_test_noisy = CalibrationLoss(LOGIT=True, adaECE=adaEce, n_bins=n_bins, true_labels=dataset.test_input_data.labels)
    loss_test_noisy = calib_test_noisy.forward(dataset.test_input_data.logits, dataset.test_input_data.noisy_labels, num_classes=dataset.test_input_data.n_classes)

    calib_valid_noisy = CalibrationLoss(LOGIT=True, adaECE=adaEce, n_bins=n_bins, true_labels=dataset.valid_input_data.labels)
    loss_valid_noisy = calib_valid_noisy.forward(dataset.valid_input_data.logits, dataset.valid_input_data.noisy_labels, num_classes=dataset.valid_input_data.n_classes)

    calib_valid_syn = CalibrationLoss(LOGIT=True, adaECE=adaEce, n_bins=n_bins, true_labels=dataset.valid_input_data.labels)
    loss_valid_syn = calib_valid_syn.forward(dataset.valid_input_data.logits, syn_pl_valid, num_classes=dataset.valid_input_data.n_classes)

    calib_test_syn = CalibrationLoss(LOGIT=True, adaECE=adaEce, n_bins=n_bins, true_labels=dataset.test_input_data.labels)
    loss_test_syn = calib_test_syn.forward(dataset.test_input_data.logits, syn_pl_test, num_classes=dataset.test_input_data.n_classes)

    return calib_test.stats, calib_valid.stats, calib_test_noisy.stats, calib_valid_noisy.stats, calib_valid_syn.stats, calib_test_syn.stats


def _histedges_equalN(x, n_bins):
        npt = len(x)
        return np.interp(np.linspace(0, npt, n_bins + 1),
                         np.arange(npt),
                         np.sort(x))

def estimate_acc_conf(dataset):
    results = []
    noisy_results = []
    noisy_results_con = []
    noisy_results_selected = []
    input_data = dataset.valid_input_data
    softmaxes = F.softmax(input_data.logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    confidences[confidences == 1] = 0.999999
    correctness = predictions.eq(dataset.valid_input_data.noisy_labels)
    true_correctness =predictions.eq(dataset.valid_input_data.labels)

    probs = torch.from_numpy(dataset.valid_input_data.confidence_pl)

    # Sampling (Bernoulli random variable for each element)
    selected = torch.rand(len(probs)) < probs

    # Get the indexes of selected elements
    selected_indexes = torch.where(selected)[0]

    selected2 = torch.rand(len(probs)) < probs

    selected_indexes2 = torch.where(selected2)[0]
    selected_indexes = torch.tensor(list(set(selected_indexes.tolist()) & set(selected_indexes2.tolist())))



    n, bin_boundaries = np.histogram(confidences.cpu().detach(),
                                     _histedges_equalN(confidences.cpu().detach(), n_bins=15))
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    print(f"total nl acc for selected indexes{sum(dataset.valid_input_data.noisy_labels[selected_indexes]==dataset.valid_input_data.labels[selected_indexes])/len(selected_indexes)}")
    for i in range(len(bin_lowers)):
        bin_lower = bin_lowers[i]
        bin_upper = bin_uppers[i]
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            correctness_in_bin = correctness[in_bin]
            pl_conf_in_bin = dataset.valid_input_data.confidence_pl[in_bin]
            noisy_acc_in_bin = correctness[in_bin].float().mean()

            noisy_accuracy_in_bin_conf = (correctness_in_bin*pl_conf_in_bin).sum() / pl_conf_in_bin.sum()

            correctness_selected = correctness[selected_indexes]
            in_bin_selected = in_bin[selected_indexes]
            correctness_in_bin_and_selected = correctness_selected[in_bin_selected]
            accuracy_in_bin_selected = correctness_in_bin_and_selected.sum() / in_bin_selected.sum()

            true_accuracy_in_bin = true_correctness[in_bin].float().mean()

            noisy_labels_selected = dataset.valid_input_data.noisy_labels[selected_indexes][in_bin_selected]
            labels_selected = dataset.valid_input_data.labels[selected_indexes][in_bin_selected]
            acc_noisy_labels = sum(noisy_labels_selected==labels_selected)/len(labels_selected)
            print(acc_noisy_labels.item())

            results.append(round(100*true_accuracy_in_bin.item(),2))
            noisy_results_con.append(round(100*noisy_accuracy_in_bin_conf.item(),2))
            noisy_results_selected.append(round(100*accuracy_in_bin_selected.item(),2))
            noisy_results.append(round(100*noisy_acc_in_bin.item(),2))
    return noisy_results, noisy_results_con, noisy_results_selected, results



if __name__ == '__main__':
    n_bins = 15
    adaEce = True
    datasets = _get_all_datasets()
    for dataset in datasets:
        print(f"Creating  plots for {dataset.dataset_name}")

        valid_syn_labels = _create_y_tilde_from_transition_matrix(dataset.valid_input_data.opt_transition_matrix, dataset.valid_input_data.labels.to(torch.int))
        test_syn_labels = _create_y_tilde_from_transition_matrix(dataset.test_input_data.opt_transition_matrix, dataset.test_input_data.labels.to(torch.int))


        noisy_acc, noisy_acc_conf, noisy_acc_selected, acc  = estimate_acc_conf(dataset)
        # cretae_tsne_plots_for_dataset(dataset)
        calib_test_stats, calib_valid_stats, calib_test_noisy_stats, calib_valid_noisy_stats, calib_valid_syn_stats, calib_test_syn_stats= run_calibration(dataset, valid_syn_labels, test_syn_labels)

        # plot.plot_dist_from_closest_center(calib_valid_stats, dataset.valid_input_data, dataset.dataset_name, "dist_from_center")
        # plot.plot_dist_from_closest_center(calib_test_stats, dataset.test_input_data, dataset.dataset_name, "dist_from_center", test=True)


    # plot.plot_agreement_explained_over_bins_v2(dataset.dataset_name, calib_test_stats, dataset.dataset_name, 'test', test = True)
    #     plot.plot_agreement_explained_over_bins_v2(dataset.dataset_name, calib_valid_stats, dataset.dataset_name, 'valid')
    #     plot.plot_agreement_explained_over_bins_v2(dataset.dataset_name, calib_test_noisy_stats, dataset.dataset_name, 'test_noisy', test=True)
    #     plot.plot_agreement_explained_over_bins_v2(dataset.dataset_name, calib_valid_noisy_stats, dataset.dataset_name, 'valid_noisy')
    #     plot.plot_agreement_explained_over_bins_v2(dataset.dataset_name, calib_test_syn_stats, dataset.dataset_name, 'test_syn', test = True)
    #     plot.plot_agreement_explained_over_bins_v2(dataset.dataset_name, calib_valid_syn_stats, dataset.dataset_name, 'valid_syn')
        #
        plot.plot_avg_noise_pl_over_bins(calib_valid_stats, dataset.valid_input_data,valid_syn_labels, dataset, "accuracy_estimation")
        plot.plot_avg_noise_pl_over_bins(calib_test_stats, dataset.test_input_data,test_syn_labels, dataset, "accuracy_estimation", test=True)
        # plot.plot_avg_acc_pl_over_bins(calib_valid_stats, dataset.valid_input_data,valid_syn_labels, dataset.dataset_name, "accuracy_pseudo_labels")
        # plot.plot_avg_acc_pl_over_bins(calib_test_stats, dataset.test_input_data,test_syn_labels, dataset.dataset_name, "accuracy_pseudo_labels", test=True)


    
    