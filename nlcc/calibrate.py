import argparse

import numpy as np

from load_data import load_test_data, load_valid_data
from execute_calibration_methods import run_calibration_methods
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate model using pseudo labels')
    parser.add_argument('--dataset', type=str, required=False, help='Dataset name', default='bloodmnist')
    parser.add_argument('--acc', type=int, required=False, help='noisy labels accuracy', default=85)
    parser.add_argument('--model', type=str, required=False, help='model (resenet50, vgg, densenet121)', default='resnet50')

    parser.add_argument('--n_bins', type=int, default=15, help='Number of bins for ECE')
    parser.add_argument('--adaECE_calib', type=bool, default=True, help='Use adaptive ECE to find the best temperature')
    parser.add_argument('--adaECE_eval', type=bool, default=True, help='Use adaptive ECE to evaluate the temperature')

    args = parser.parse_args()
    data_set = args.dataset
    accuracy = args.acc
    model= args.model
    n_bins = args.n_bins
    adaECE_calib = args.adaECE_calib
    adaECE_eval = args.adaECE_eval


    # Load data
    valid_input_data = load_valid_data(dataset=data_set, model=model, accuracy= accuracy)
    test_input_data = load_test_data(dataset=data_set, model=model, accuracy= accuracy)
    # Run calibration methods
    Ts, losses = run_calibration_methods(valid_input_data, test_input_data, n_bins, adaECE_calib, adaECE_eval)
    for key in losses.keys():
        losses[key] = round(100*losses[key], 2)
        Ts[key] = round(Ts[key], 2)

    loss_st = "adaECE" if adaECE_eval else "ECE"


    print(f"Data set: {data_set}")
    print(f"losses: {losses}")
    print(f"T: {Ts}")

    results = {'T': Ts, loss_st: losses}
    output_dir = f'{os.path.dirname(os.path.abspath(__file__))}/outputs'
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/{data_set}_calibration_results.json', 'w') as f:
        json.dump(results, f)
