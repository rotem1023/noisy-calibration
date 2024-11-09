import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import os
import seaborn as sns
import numpy as np
import torch


def _get_cur_file_path():
    return f'{os.path.dirname(os.path.abspath(__file__))}'

def _get_plots_dir():
    return f'{_get_cur_file_path()}/plots'

def _get_dataset_plots_dir(dataset, test):
    plots_dir  = _get_plots_dir()
    os.makedirs(plots_dir, exist_ok=True)
    dataset_dir = f'{plots_dir}/{dataset}'
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_dir = f'{dataset_dir}/{_get_set_name(test)}'
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir


def _get_set_name(test):
    return 'test' if test else 'valid'

def _add_axes_labels(ax, x_label, y_label):
    ax.set_xlabel(x_label, fontsize=26)
    ax.set_ylabel(y_label, fontsize=26)
    ax.legend(fontsize=20)

def _add_ax_plot(ax, x, y, label, color):
    ax.plot(x, y, label=label, color=color, linewidth=3)

def create_tsne_plot(dataset, x, y, plot_name, test = False):
    tsne = TSNE(n_components=2, random_state=0)
    x_tsne = tsne.fit_transform(x)
    
    df = pd.DataFrame(x_tsne, columns = ['TSNE1', 'TSNE2'])
    df['label'] = y
    
    plt.figure(figsize=(10,6))
    scater =plt.scatter(df['TSNE1'], df['TSNE2'], c = df['label'], cmap='viridis')
    plt.colorbar(scater, label='label')
    
    plots_dir = _get_dataset_plots_dir(dataset, test)
    
    plt.savefig(f"{plots_dir}/tsne_{plot_name}.png", format='png', dpi = 300)
    plt.show()
    
    plt.close()


def plot_agreement_explained_over_bins_v2(dataset, stats, data_set, plot_name, test = False):
    '''
    Plot prediction agreement with true and pseudo labels over bins of confidence.
    Plot prediction agreement only with true labels over bins of confidence.
    Plot prediction agreement only with pseudo labels over bins of confidence.
    :param stats: dictionary holds the statistics from the calibration process
    :param data_set: data set name
    :param plot_name: name of the plot to save
    :param synthetic: is the pseudo labels are real or synthetic using the true transition matrix or the estimated one
    :param year: the year of the data set
    :return:
    '''
    all_agree = 'all_agree'
    pl_agree = 'pl_agree'
    true_agree = 'true_agree'
    all_agree_dic = stats[all_agree]
    pl_agree_dic = stats[pl_agree]
    true_agree_dic = stats[true_agree]
    confidence_dic = stats['confidence']
    bins = sorted(list(all_agree_dic.keys()))
    # get the agreement values for each bin
    all_agree_values = [round(100 * all_agree_dic[bin]) for bin in bins]
    pl_agree_values = [round(100 * pl_agree_dic[bin]) for bin in bins]
    true_agree_values = [round(100 * true_agree_dic[bin]) for bin in bins]
    confidence_values = [round(100 * confidence_dic[bin]) for bin in bins]

    width = 0.25  # Width of each bar
    group_spacing = 0.4  # Space between each group of bars

    # Adjust x positions to include spacing between groups
    x = np.arange(len(bins)) * (3 * width + group_spacing)

    # Set a Seaborn style for better aesthetics
    sns.set_style("whitegrid")
    color_palette = "Paired"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plots with appropriate spacing
    ax.bar(x, pl_agree_values, width, label=r'$\hat{y} = \tilde{y} \neq y$',
           color=sns.color_palette(color_palette)[6])
    ax.bar(x + width, true_agree_values, width, label=r'$\hat{y} = y \neq \tilde{y}$',
           color=sns.color_palette(color_palette)[3])
    ax.bar(x + 2 * width, all_agree_values, width, label=r'$\hat{y} = \tilde{y} = y$',
           color=sns.color_palette(color_palette)[4])

    # Adding labels to the axes
    _add_axes_labels(ax, 'Confidence Bins (lowest to highest)', 'Agreement (%)')

    # Set xticks to the middle of the grouped bars
    ax.set_xticks(x + width)  # Center xticks between the groups of bars
    ax.set_xticklabels(bins)
    # ax.set_xticklabels(confidence_values, fontsize = 18)
    ax.set_ylim(0, 100)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    dir_to_save = _get_dataset_plots_dir(dataset, test)

    plt.savefig(f"{dir_to_save}/{plot_name}_{data_set}_{len(bins)}_bins.png", dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    plt.close(fig)


def plot_avg_noise_pl_over_bins(stats, input_data, syn_pl, data_set, plot_name, test = False):
    '''
    The name of the function is misleading.
    The function plots the accuracy of the pseudo labels and the source model predictions, as well as synthetic pseudo labels over bins of confidence.
    :param stats: dictionary holds the statistics from the calibration process
    :param input_data: raw data from the data set
    :param syn_pl: true if to plot the synthetic pseudo labels
    :param data_set: name of the data set
    :param plot_name: name of the plot to save
    :param year: year of the data set
    :return:
    '''
    y= input_data.labels
    noisy_labels = input_data.noisy_labels
    y_tilda = input_data.pseudo_labels
    prediction = torch.argmax(input_data.logits, dim=1)
    index = stats['indexes']
    bins = sorted(list(index.unique().int()))
    noises = []
    noisy_labels_noises = []
    syn_noises = []
    prediction_noises = []
    prediction_noises_noisy = []
    for i in range(len(bins)):
        bin_index = index == bins[i]

        prediction_bin = prediction[bin_index]
        noisy_labels_bin = noisy_labels[bin_index]
        y_tilda_bin = y_tilda[bin_index]
        y_bin = y[bin_index]



        noise = 100*(sum(y_tilda_bin == y_bin) / len(y_tilda_bin))
        noises.append(round(noise.item()))
        noisy_label_noise = 100*(sum(noisy_labels_bin == y_bin) / len(noisy_labels_bin))
        noisy_labels_noises.append(round(noisy_label_noise.item()))
        preds_noise = 100*(sum(prediction_bin == y_bin) / len(prediction_bin))
        prediction_noises.append(round(preds_noise.item()))
        preds_noise_noisy = 100*(sum(prediction_bin == y_tilda_bin) / len(prediction_bin))
        prediction_noises_noisy.append(round(preds_noise_noisy.item()))


        if syn_pl is not None:
            syn_pl_bin = syn_pl[bin_index]
            syn_noise = 100*(sum(syn_pl_bin == y_bin) / len(syn_pl_bin))
            syn_noises.append(round(syn_noise.item()))

    # Set up bar plot parameters
    bar_width = 0.35  # Adjusted width of the bars
    spacing = 0.1  # Space between different x labels
    x = np.arange(len(bins))

    sns.set_style("whitegrid")
    color_palette = "Paired"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    if syn_pl is not None:
        _add_ax_plot(ax,x , syn_noises, color=sns.color_palette(color_palette)[8], label='Synthetic')
    _add_ax_plot(ax, x, noises, label='Enhanced PL', color=sns.color_palette(color_palette)[4])
    _add_ax_plot(ax,x , noisy_labels_noises, color=sns.color_palette(color_palette)[6], label='PL')
    _add_ax_plot(ax,x , prediction_noises, color=sns.color_palette(color_palette)[3], label='Preds')
    _add_ax_plot(ax,x , prediction_noises_noisy, color=sns.color_palette(color_palette)[2], label='Preds Noisy')


    # Label the plot
    _add_axes_labels(ax, 'Confidence Bins (lowest to highest)', 'Accuracy')
    # ax.set_title(f'{_create_title(data_set)} Pseudo Labels Noise over {len(bins)} Bins')

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Set y-axis limits
    ax.set_ylim(0, 100)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    plt.tight_layout()

    # Save the plot
    filename = f"{plot_name}_{data_set}_{len(bins)}_bins.png"
    dir_to_save = _get_dataset_plots_dir(data_set, test)
    plt.savefig(f"{dir_to_save}/{filename}", dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()
    plt.close(fig)


def plot_dist_from_closest_center(stats, input_data, data_set, plot_name, test = False):
    '''
    The name of the function is misleading.
    The function plots the accuracy of the pseudo labels and the source model predictions, as well as synthetic pseudo labels over bins of confidence.
    :param stats: dictionary holds the statistics from the calibration process
    :param input_data: raw data from the data set
    :param syn_pl: true if to plot the synthetic pseudo labels
    :param data_set: name of the data set
    :param plot_name: name of the plot to save
    :param year: year of the data set
    :return:
    '''
    y= input_data.labels
    dist_from_center = input_data.dist_closest_center
    y_tilda = input_data.pseudo_labels
    index = stats['indexes']

    correct = y == y_tilda

    bins = sorted(list(index.unique().int()))

    for i in range(len(bins)):
        bin_index = index == bins[i]
        dist_bin = dist_from_center[bin_index]
        correct_bin = correct[bin_index]

        # for each bin create a plot of the distance from the center of the class, color the plot based on the correctness of the pseudo label
        sns.set_style("whitegrid")
        color_palette = "Paired"

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(correct_bin, dist_bin, c=correct_bin, cmap='viridis')

        filename = f"{plot_name}_{data_set}_{len(bins)}_bins_{i}_bin.png"
        dir_to_save = _get_dataset_plots_dir(data_set, test)
        plt.savefig(f"{dir_to_save}/{filename}", dpi=300, bbox_inches='tight')

        # Show the plot
        plt.show()
        plt.close(fig)


