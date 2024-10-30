import numpy as np

from calibration_loss import CalibrationLoss
from find_temp import FindTemp
import torch
from enum import Enum


class CalibrationMethodName(Enum):
    '''
    all calibration methods names we tested
    '''
    Uncalibrated = 'Uncalibrated'
    NoisyTS = 'Noisy-TS'
    SMPL = 'SMPL'
    NLCC = 'NLCC'
    NTS = 'NTS'
    TsClean = 'TS-Clean'







def run_calibration_methods(valid_input_data, test_input_data, n_bins, adaECE_calib, adaECE_eval):
    output_t = {}
    output_loss = {}

    # ece loss used to evaluate calibration
    ece_loss = CalibrationLoss(LOGIT=True, adaECE=adaECE_eval, n_bins=n_bins, true_labels= valid_input_data.labels.clone().detach())
    calib_model = FindTemp(n_classes=valid_input_data.n_classes, n_bins=n_bins, LOGIT=True, adaECE=adaECE_calib)

    # No calibration
    output_loss[CalibrationMethodName.Uncalibrated.value] = calc_calibration_loss_with_temp(ece_loss, test_input_data, 1)
    output_t[CalibrationMethodName.Uncalibrated.value] = 1
    print("finish no calibration")

    # Ts Clean
    T = calib_model.find_best_T(valid_input_data.logits.clone().detach(), valid_input_data.labels.clone().detach()).item()
    output_loss[CalibrationMethodName.TsClean.value] = calc_calibration_loss_with_temp(ece_loss, test_input_data, T)
    output_t[CalibrationMethodName.TsClean.value] = T
    print("finish Ts clean")
    
    # Noisy TS
    T = calib_model.find_best_T(valid_input_data.logits.clone().detach(), valid_input_data.noisy_labels.clone().detach()).item()
    output_loss[CalibrationMethodName.NoisyTS.value] = calc_calibration_loss_with_temp(ece_loss, test_input_data, T)
    output_t[CalibrationMethodName.NoisyTS.value] = T
    print("finish noisy ts")

    # NTS
    if valid_input_data.transtion_matrix is not None:
        T = calib_model.find_best_T_with_transition_matrix(valid_input_data.logits.clone().detach(),
                                                           valid_input_data.pseudo_labels.clone().detach(),
                                                           valid_input_data.transtion_matrix).item()
        output_loss[CalibrationMethodName.NTS.value] = calc_calibration_loss_with_temp(ece_loss, test_input_data, T)
        output_t[CalibrationMethodName.NTS.value] = T
        print("finish nts")
        
    # nlcc
    T =  calib_model.find_best_T(valid_input_data.logits.clone().detach(), valid_input_data.pseudo_labels.clone().detach(), true_labels=valid_input_data.labels.clone().detach()).item()
    output_loss[CalibrationMethodName.NLCC.value] = calc_calibration_loss_with_temp(ece_loss, test_input_data, T)
    output_t[CalibrationMethodName.NLCC.value] = T
    print("finish nlcc")
    
    return output_t, output_loss




def _calc_losses_for_T(Ts, input_data, ece_loss):
    n = len(Ts)
    losses = []
    for i in range(n):
        losses.append(calc_calibration_loss_with_temp(ece_loss, input_data, Ts[i]))
    output_T = 0
    output_loss = 0
    for i in range(n):
        output_T += Ts[i] / n
        output_loss += losses[i] / n
    return output_T, output_loss


def calc_calibration_loss_with_temp(ece_loss, input_data, T):
    return ece_loss.forward(input_data.logits.clone().detach() / T, input_data.labels,
                            num_classes=input_data.n_classes).item()
