import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
import numpy as np






class CalibrationLoss(nn.Module):
    def __init__(self, n_bins=15, LOGIT=True, adaECE=False, true_labels= None):
        super(CalibrationLoss, self).__init__()
        self.nbins = n_bins
        self.LOGIT = LOGIT
        self.adaECE = adaECE
        self.true_labels = true_labels
        self.stats = {}

    def forward(self, logits, labels, num_classes=10, epsilon=None, transition_matrix=None):
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        correctness = predictions.eq(labels)
        confidences[confidences == 1] = 0.999999
        bin_lowers, bin_uppers = self._claculate_bin_boundaries(confidences)

        ece = torch.zeros(1, device=logits.device)

        for i in range(len(bin_lowers)):
            bin_lower = bin_lowers[i]
            bin_upper = bin_uppers[i]

            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0 and (self.adaECE or in_bin.sum() > 20):
                accuracy_in_bin = self._calculate_accuracy_in_bin(in_bin, correctness, num_classes, predictions, labels,
                                                                  epsilon, transition_matrix)
                if self.true_labels is not None:
                    true_acc_in_bin = self._calculate_accuracy_in_bin(in_bin, predictions.eq(self.true_labels), num_classes, predictions, self.true_labels,
                                                                  None, None)
                avg_confidence_in_bin = confidences[in_bin].mean()
                cur_ece = torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                ece += cur_ece
        return ece


    def forward_given_acc(self, logits, acc):
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        confidences[confidences == 1] = 0.999999
        ece = torch.zeros(1, device=logits.device)
        bin_lowers, bin_uppers = self._claculate_bin_boundaries(confidences)


        for idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if (prop_in_bin.item() > 0):
                if acc[idx] is not None:
                    accuracy_in_bin = acc[idx]
                    accuracy_in_bin = self._normalize_acc(accuracy_in_bin)
                    avg_confidence_in_bin = confidences[in_bin].mean().float()

                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    def _claculate_bin_boundaries(self, confidences):
        if self.adaECE:
            n, bin_boundaries = np.histogram(confidences.cpu().detach(),
                                             self._histedges_equalN(confidences.cpu().detach()))
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
        else:
            bin_boundaries = torch.linspace(0, 1, self.nbins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
        return bin_lowers, bin_uppers

    def _histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                         np.arange(npt),
                         np.sort(x))

    def _calculate_accuracy_in_bin(self, in_bin, correctness, num_classes, predictions, labels, epsilon=None,
                                   transition_matrix=None):
        # we assume that at most one of epsilon and transition_matrix is not None
        assert (epsilon is None) or (
                    transition_matrix is None), "Only one of epsilon and transition_matrix should be not None"

        if epsilon is not None:  # noisy-lables
            accuracy_in_bin = correctness[in_bin].float().mean()
            accuracy_in_bin = self._normalize_acc(accuracy_in_bin)

            # -- Fixing the noisy-accuracy -- #
            accuracy_in_bin = (accuracy_in_bin - (epsilon / (num_classes - 1))) / (
                        1 - epsilon - (epsilon / (num_classes - 1)))
        elif transition_matrix is not None:  # transition-matrix
            predictions_in_bin = predictions[in_bin]
            labels_in_bin = labels[in_bin]
            # Compute confusion matrix in bin
            M = confusion_matrix(predictions_in_bin, labels_in_bin, labels=np.arange(num_classes), normalize='all')

            # Compute "fixed" accuracy in bin -  A = M * inv(P)
            p_inv = np.linalg.inv(transition_matrix)
            accuracy_in_bin = np.trace(M * p_inv)

            accuracy_in_bin = self._normalize_acc(accuracy_in_bin)
        else:
            accuracy_in_bin = correctness[in_bin].float().mean()
            accuracy_in_bin = self._normalize_acc(accuracy_in_bin)
        return accuracy_in_bin

    def _normalize_acc(self, accuracy_in_bin):
        accuracy_in_bin = min(accuracy_in_bin, 0.99)
        accuracy_in_bin = max(accuracy_in_bin, 0.01)
        return accuracy_in_bin
