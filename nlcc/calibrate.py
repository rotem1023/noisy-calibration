import argparse

import numpy as np

from load_data import load_test_data, load_valid_data
from execute_calibration_methods import run_calibration_methods
import json
import os


def run(data_set, model, accuracy, n_bins, adaECE_calib, adaECE_eval):
    # Load data
    valid_input_data = load_valid_data(dataset=data_set, model=model, accuracy= accuracy)
    test_input_data = load_test_data(dataset=data_set, model=model, accuracy= accuracy)

    # Run calibration methods
    Ts, losses = run_calibration_methods(valid_input_data, test_input_data, n_bins, adaECE_calib, adaECE_eval)

    loss_st = "adaECE" if adaECE_eval else "ECE"

    for key in Ts.keys():
        losses[loss_st][key] = round(100*losses[loss_st][key], 2)
        Ts[key] = round(Ts[key], 2)



    print(f"Data set: {data_set}")
    print(f"losses: {losses}")
    print(f"T: {Ts}")

    # create final dict
    results = {'T': Ts}
    results.update(losses)

    output_dir = f'{os.path.dirname(os.path.abspath(__file__))}/outputs'
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = f'{output_dir}/{data_set}'
    os.makedirs(dataset_dir, exist_ok=True)
    acc_dir = f'{dataset_dir}/{accuracy}'
    os.makedirs(acc_dir, exist_ok=True)
    with open(f'{acc_dir}/model_{model}_{loss_st}_calibration_results.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate model using pseudo labels')
    parser.add_argument('--dataset', type=str, required=False, help='Dataset name', default='bloodmnist')
    parser.add_argument('--acc', type=int, required=False, help='noisy labels accuracy', default=90)
    parser.add_argument('--model', type=str, required=False, help='model (resnet50, vgg, densenet121)', default='vgg')

    parser.add_argument('--n_bins', type=int, default=15, help='Number of bins for ECE')
    parser.add_argument('--adaECE_calib', type=bool, default=True, help='Use adaptive ECE to find the best temperature')
    # parser.add_argument('--adaECE_eval', type=bool, default=True, help='Use adaptive ECE to evaluate the temperature')

    args = parser.parse_args()
    data_set = args.dataset
    accuracy = args.acc
    model= args.model
    n_bins = args.n_bins
    adaECE_calib = args.adaECE_calib
    # adaECE_eval = args.adaECE_eval
    adaECE_eval  = args.adaECE_calib

    datasets = ['pathmnist']
    models = [ 'densenet121']
    accuracies = [95]
    for data_set in datasets:
        for model in models:
            for accuracy in accuracies:
                print(f"run calibration on dataset: {data_set}, model: {model}, accuracy: {accuracy}, bins: {n_bins}, val loss: {adaECE_calib}, test loss: {adaECE_eval}")
                run(data_set=data_set, model= model, accuracy=accuracy, n_bins=n_bins,adaECE_calib=adaECE_calib, adaECE_eval=adaECE_eval)
