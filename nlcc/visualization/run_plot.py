import sys
sys.path.append('..')
from nlcc.load_data import load_test_data, load_valid_data
import os
import plot
from nlcc.calibration_loss import CalibrationLoss
import numpy as np
import torch


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
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.valid_input_data = load_valid_data(dataset_name)
        self.test_input_data = load_test_data(dataset_name)


def _get_all_datasets():
    dataset_names= _get_datasets_names()
    return [InputData(dataset_name) for dataset_name in dataset_names]



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
    loss_test = calib_test.forward(dataset.test_input_data.logits, dataset.test_input_data.pseudo_labels, num_classes=dataset.test_input_data.n_classes)

    calib_valid = CalibrationLoss(LOGIT=True, adaECE=adaEce, n_bins=n_bins, true_labels=dataset.valid_input_data.labels)
    loss_valid = calib_valid.forward(dataset.valid_input_data.logits, dataset.valid_input_data.pseudo_labels, num_classes=dataset.valid_input_data.n_classes)

    calib_test_noisy = CalibrationLoss(LOGIT=True, adaECE=adaEce, n_bins=n_bins, true_labels=dataset.test_input_data.labels)
    loss_test_noisy = calib_test_noisy.forward(dataset.test_input_data.logits, dataset.test_input_data.noisy_labels, num_classes=dataset.test_input_data.n_classes)

    calib_valid_noisy = CalibrationLoss(LOGIT=True, adaECE=adaEce, n_bins=n_bins, true_labels=dataset.valid_input_data.labels)
    loss_valid_noisy = calib_valid_noisy.forward(dataset.valid_input_data.logits, dataset.valid_input_data.noisy_labels, num_classes=dataset.valid_input_data.n_classes)

    calib_valid_syn = CalibrationLoss(LOGIT=True, adaECE=adaEce, n_bins=n_bins, true_labels=dataset.valid_input_data.labels)
    loss_valid_syn = calib_valid_syn.forward(dataset.valid_input_data.logits, syn_pl_valid, num_classes=dataset.valid_input_data.n_classes)

    calib_test_syn = CalibrationLoss(LOGIT=True, adaECE=adaEce, n_bins=n_bins, true_labels=dataset.test_input_data.labels)
    loss_test_syn = calib_test_syn.forward(dataset.test_input_data.logits, syn_pl_test, num_classes=dataset.test_input_data.n_classes)

    return calib_test.stats, calib_valid.stats, calib_test_noisy.stats, calib_valid_noisy.stats, calib_valid_syn.stats, calib_test_syn.stats




if __name__ == '__main__':
    n_bins = 15
    adaEce = True
    datasets = _get_all_datasets()
    for dataset in datasets:
        print(f"Creating  plots for {dataset.dataset_name}")

        valid_syn_labels = _create_y_tilde_from_transition_matrix(dataset.valid_input_data.transtion_matrix, dataset.valid_input_data.labels.to(torch.int))
        test_syn_labels = _create_y_tilde_from_transition_matrix(dataset.test_input_data.transtion_matrix, dataset.test_input_data.labels.to(torch.int))

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
        plot.plot_avg_noise_pl_over_bins(calib_valid_stats, dataset.valid_input_data,valid_syn_labels, dataset.dataset_name, "accuracy_pseudo_labels")
        plot.plot_avg_noise_pl_over_bins(calib_test_stats, dataset.test_input_data,test_syn_labels, dataset.dataset_name, "accuracy_pseudo_labels", test=True)



    
    