import numpy as np

from calibration_loss import CalibrationLoss
from find_temp import FindTemp
import torch
from enum import Enum
from constants.loss_name import LossName

class CalibrationMethodName(Enum):
    '''
    all calibration methods names we tested
    '''
    Uncalibrated = 'Uncalibrated'
    NoisyTS = 'Noisy-TS'
    SMPL = 'SMPL'
    NTS = 'NTS'
    NTS_OPT = 'NTS*'
    TsClean = 'TS-Clean'
    NLCCConf = 'NLCCConf'
    NLCCRand = 'NLCCRand'
    NLCC = 'NLCC'








def run_calibration_methods(valid_input_data, test_input_data, n_bins, adaECE_calib, adaECE_eval):
    output_t = {}
    if adaECE_eval:
        output_loss = {LossName.adaECE.value: {}, LossName.adaNLL.value: {}, LossName.adaBS.value: {}, LossName.adaSCE.value: {}}
    else:
        output_loss = {LossName.ECE.value: {}, LossName.NLL.value: {}, LossName.BS.value: {}, LossName.SCE.value: {}}

    # ece loss used to evaluate calibration
    ece_loss = CalibrationLoss(LOGIT=True, adaECE=adaECE_eval, n_bins=n_bins)
    calib_model = FindTemp(n_classes=valid_input_data.n_classes, n_bins=n_bins, LOGIT=True, adaECE=adaECE_calib)

    # No calibration
    calc_calibration_losses_with_temp(CalibrationMethodName.Uncalibrated, output_loss, ece_loss, input_data=test_input_data, T=1, adaECE=adaECE_eval)
    # output_loss[CalibrationMethodName.Uncalibrated.value] = calc_calibration_loss_with_temp(ece_loss, test_input_data, 1)
    output_t[CalibrationMethodName.Uncalibrated.value] = 1
    print("finish no calibration")

    # Ts Clean
    T = calib_model.find_best_T(valid_input_data.logits.clone().detach(), valid_input_data.labels.clone().detach()).item()
    # output_loss[CalibrationMethodName.TsClean.value] = calc_calibration_loss_with_temp(ece_loss, test_input_data, T)
    calc_calibration_losses_with_temp(CalibrationMethodName.TsClean, output_loss, ece_loss, input_data=test_input_data, T=T, adaECE=adaECE_eval)
    output_t[CalibrationMethodName.TsClean.value] = T
    print("finish Ts clean")
    
    # Noisy TS
    T = calib_model.find_best_T(valid_input_data.logits.clone().detach(), valid_input_data.noisy_labels.clone().detach(), true_labels=valid_input_data.labels.clone().detach()).item()
    # output_loss[CalibrationMethodName.NoisyTS.value] = calc_calibration_loss_with_temp(ece_loss, test_input_data, T)
    calc_calibration_losses_with_temp(CalibrationMethodName.NoisyTS, output_loss, ece_loss, input_data=test_input_data, T=T, adaECE=adaECE_eval)
    output_t[CalibrationMethodName.NoisyTS.value] = T
    print("finish noisy ts")

    # NTS
    if test_input_data.transition_matrix is not None:
        T = calib_model.find_best_T_with_transition_matrix(valid_input_data.logits.clone().detach(),
                                                           valid_input_data.noisy_labels.clone().detach(),
                                                           valid_input_data.transition_matrix).item()
        # output_loss[CalibrationMethodName.NTS.value] = calc_calibration_loss_with_temp(ece_loss, test_input_data, T)
        calc_calibration_losses_with_temp(CalibrationMethodName.NTS, output_loss, ece_loss, input_data=test_input_data, T=T, adaECE=adaECE_eval)
        output_t[CalibrationMethodName.NTS.value] = T
        print("finish nts")

    # NTS opt
    if valid_input_data.opt_transition_matrix is not None:
        T = calib_model.find_best_T_with_transition_matrix(valid_input_data.logits.clone().detach(),
                                                           valid_input_data.noisy_labels.clone().detach(),
                                                           valid_input_data.opt_transition_matrix).item()
        # output_loss[CalibrationMethodName.NTS_OPT.value] = calc_calibration_loss_with_temp(ece_loss, test_input_data, T)
        calc_calibration_losses_with_temp(CalibrationMethodName.NTS_OPT, output_loss, ece_loss, input_data=test_input_data, T=T, adaECE=adaECE_eval)
        output_t[CalibrationMethodName.NTS_OPT.value] = T
        print("finish nts opt")
        
    # nlcc
    T =  calib_model.find_best_T_with_indexes(valid_input_data.logits.clone().detach(), valid_input_data.noisy_labels.clone().detach(), relevant_indexes=valid_input_data.relevant_indexes, true_labels = valid_input_data.labels).item()
    # output_loss[CalibrationMethodName.NLCC.value] = calc_calibration_loss_with_temp(ece_loss, test_input_data, T)
    calc_calibration_losses_with_temp(CalibrationMethodName.NLCC, output_loss, ece_loss, input_data=test_input_data, T=T, adaECE=adaECE_eval)
    output_t[CalibrationMethodName.NLCC.value] = T
    print("finish nlcc")

    # NLCC conf
    T = calib_model.find_best_T_with_pl_confidence(valid_input_data.logits.clone().detach(), valid_input_data.noisy_labels.clone().detach(), valid_input_data.confidence_pl, true_labels=valid_input_data.labels).item()
    # output_loss[CalibrationMethodName.NLCCConf.value] = calc_calibration_loss_with_temp(ece_loss, test_input_data, T)
    calc_calibration_losses_with_temp(CalibrationMethodName.NLCCConf, output_loss, ece_loss, input_data=test_input_data, T=T, adaECE=adaECE_eval)
    output_t[CalibrationMethodName.NLCCConf.value] = T
    print("finish nlcc conf")

    # Nlcc Rand
    Ts = []
    for i in range(10):
        Ts.append(calib_model.find_best_T_with_randomenss(valid_input_data.logits.clone().detach(), valid_input_data.noisy_labels.clone().detach(), valid_input_data.confidence_pl).item())
    T = sum(Ts)/len(Ts)
    # output_loss[CalibrationMethodName.NLCCRand.value] = calc_calibration_loss_with_temp(ece_loss, test_input_data, T)
    calc_calibration_losses_with_temp(CalibrationMethodName.NLCCRand, output_loss, ece_loss, input_data=test_input_data, T=T, adaECE=adaECE_eval)
    output_t[CalibrationMethodName.NLCCRand.value] = T
    
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

def calc_calibration_losses_with_temp(method, loss_dic, ece_loss, input_data, T, adaECE):
    loss = ece_loss.forward(input_data.logits.clone().detach() / T, input_data.labels,
                     num_classes=input_data.n_classes).item()
    nll_loss  = ece_loss.nll_forward(input_data.logits.clone().detach() / T, input_data.labels).item()
    bs_loss = ece_loss.bs_forward(input_data.logits.clone().detach() / T, input_data.labels, num_classes=input_data.n_classes).item()
    sce_loss = ece_loss.sce_forward(input_data.logits.clone().detach() / T, input_data.labels, num_classes=input_data.n_classes).item()
    if adaECE:
        loss_dic[LossName.adaECE.value][method.value] = loss
        loss_dic[LossName.adaNLL.value][method.value] = nll_loss
        loss_dic[LossName.adaBS.value][method.value] = bs_loss
        loss_dic[LossName.adaSCE.value][method.value] = sce_loss
    else:
        loss_dic[LossName.ECE.value][method.value] = loss
        loss_dic[LossName.NLL.value][method.value] = nll_loss
        loss_dic[LossName.BS.value][method.value] = bs_loss
        loss_dic[LossName.SCE.value][method.value] = sce_loss



