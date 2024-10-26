import torch
from torch import nn
from calibration_loss import CalibrationLoss
from scipy import optimize
import numpy as np
from torch.nn import functional as F
from scipy.optimize import minimize


ranges = (slice(1, 10, 0.05),)


class FindTemp(nn.Module):

    def __init__(self, n_classes=10, n_bins=15, LOGIT=True, adaECE=False):
        super(FindTemp, self).__init__()
        self.adaECE = adaECE
        self.n_classes = n_classes
        self.n_bins = n_bins
        self.LOGIT = LOGIT

    def find_best_T(self, logits, labels, true_labels = None):
        ece_loss = CalibrationLoss(adaECE=self.adaECE, n_bins=self.n_bins, LOGIT=self.LOGIT, true_labels = true_labels)

        def eval(x):
            "x ==> temperature T"
            x = torch.from_numpy(x)
            if (x < 0):
                return 1
            scaled_logits = logits.float() / x
            return ece_loss.forward(scaled_logits, labels, self.n_classes)

        return self._calc_optimal_T(eval)

    def find_best_T_with_noise(self, logits, labels, epsilon):
        assert epsilon is not None, "epsilon should be provided"

        ece_loss = CalibrationLoss(adaECE=self.adaECE, n_bins=self.n_bins, LOGIT=self.LOGIT)

        def eval(x):
            "x ==> temperature T"
            x = torch.from_numpy(x)
            if (x < 0):
                return 1
            scaled_logits = logits.float() / x
            return ece_loss.forward(scaled_logits, labels, self.n_classes, epsilon=epsilon)

        return self._calc_optimal_T(eval)

    def find_best_T_with_transition_matrix(self, logits, labels, transition_matrix):
        assert transition_matrix is not None, "transition_matrix should be provided"

        ece_loss = CalibrationLoss(adaECE=self.adaECE, n_bins=self.n_bins, LOGIT=self.LOGIT)

        def eval(x):
            "x ==> temperature T"
            x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            if (x < 0):
                return 1
            output = ece_loss.forward(scaled_logits, labels, self.n_classes, transition_matrix=transition_matrix)
            return output

        return self._calc_optimal_T(eval)

    def _calc_optimal_T(self, eval_func):
        global ranges
        optimal_parameter = optimize.brute(eval_func, ranges, finish=optimize.fmin)
        return optimal_parameter[0]




class TransCal(nn.Module):
    def __init__(self, bias_term=True, variance_term=True):
        super(TransCal, self).__init__()
        self.bias_term = bias_term
        self.variance_term = variance_term

    def find_best_T(self, logits, weight, error, source_confidence):
        def eval(x):
            "x ==> temperature T"

            scaled_logits = logits / x[0]

            "x[1] ==> learnable meta parameter \lambda"
            if self.bias_term:
                controled_weight = weight ** x[1]
            else:
                controled_weight = weight

            ## 1. confidence
            max_L = np.max(scaled_logits, axis=1, keepdims=True)
            exp_L = np.exp(scaled_logits - max_L)
            softmaxes = exp_L / np.sum(exp_L, axis=1, keepdims=True)
            confidences = np.max(softmaxes, axis=1)
            confidence = np.mean(confidences)
            ## 2. accuracy
            if self.variance_term:
                weighted_error = controled_weight * error
                cov_1 = np.cov(np.concatenate((weighted_error, controled_weight), axis=1), rowvar=False)[0][1]
                var_w = np.var(controled_weight, ddof=1).item()
                eta_1 = - cov_1 / (var_w)

                cv_weighted_error = weighted_error + eta_1 * (controled_weight - 1)
                correctness = 1 - error
                cov_2 = np.cov(np.concatenate((cv_weighted_error, correctness), axis=1), rowvar=False)[0][1]
                var_r = np.var(correctness, ddof=1)
                eta_2 = - cov_2 / (var_r)

                target_risk = np.mean(weighted_error) + eta_1 * np.mean(controled_weight) - eta_1 \
                              + eta_2 * np.mean(correctness) - eta_2 * source_confidence
                estimated_acc = 1.0 - target_risk
            else:
                weighted_error = controled_weight * error
                target_risk = np.mean(weighted_error)
                estimated_acc = 1.0 - target_risk

                ## 3. ECE on bin_size = 1 for optimizing.
            ## Note that: We still utilize a bin_size of 15 while evaluating,
            ## following the protocal of Guo et al. (On Calibration of Modern Neural Networks)
            loss = np.abs(confidence - estimated_acc)
            return loss

        # return best_tmp
        bnds = ((1.0, None), (0.01, 1.0))
        optimal_parameter = minimize(eval, np.array([2.0, 0.5]), method='SLSQP', bounds=bnds)
        return optimal_parameter.x[0].item()



class CPCS(nn.Module):
    def __init__(self):
        super(CPCS, self).__init__()

    def find_best_T(self, logits, labels, weight):
        def eval(x):
            "x ==> temperature T"
            if type(x) != np.float64:
                x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            softmaxes = F.softmax(scaled_logits, dim=1)

            ## Transform to onehot encoded labels
            labels_onehot = torch.FloatTensor(scaled_logits.shape[0], scaled_logits.shape[1])
            labels_onehot.zero_()
            labels_onehot.scatter_(1, labels.long().view(len(labels), 1), 1)
            brier_score = torch.sum((softmaxes - labels_onehot) ** 2, dim=1,keepdim = True)
            loss = torch.mean(brier_score * weight)
            return loss

        global ranges

        optimal_parameter = optimize.fmin(eval, 2.0, disp=False)

        return optimal_parameter[0]



class UTDC(nn.Module):
    def __init__(self, n_bins, acc_fix, adeECE):
        super(UTDC, self).__init__()
        self.n_bins = n_bins
        self.acc_fix = acc_fix
        self.adeECE = adeECE

    def find_best_T(self, logits, source_logits, source_labels):
        ece_criterion = CalibrationLoss(adaECE=self.adeECE, n_bins=self.n_bins, LOGIT=True)
        acc_list = self.get_scaled_accuracy_of_bins(source_logits, source_labels)
        def eval(x):
            "x ==> temperature T"
            if type(x) != np.float64:
                x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            loss = ece_criterion.forward_given_acc(scaled_logits, acc_list)
            return loss

        global ranges
        optimal_parameter = optimize.brute(eval, ranges, finish=optimize.fmin)

        return optimal_parameter[0]

    def get_scaled_accuracy_of_bins(self, logits, labels):
        def histedges_equalN(nbins, x):
            npt = len(x)
            return np.interp(np.linspace(0, npt, nbins + 1),
                             np.arange(npt),
                             np.sort(x))


        accuracy_list = []
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        correctness = predictions.eq(labels)

        if self.adeECE:
            n, bin_boundaries = np.histogram(confidences.cpu().detach(), histedges_equalN(self.n_bins, confidences.cpu().detach()))
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
        else:
            bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]


        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = correctness[in_bin].float().mean()
                accuracy_list.append(accuracy_in_bin.item()*self.acc_fix)
            else:
                accuracy_list.append(0)

        return accuracy_list