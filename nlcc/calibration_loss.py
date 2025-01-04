import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
import numpy as np


def add_stats(bin_num, all_agreement, y_y_hat_agreement, y_tilda_y_hat_agreement,
              y_y_tilda_agreement_all, y_hat_y_tilda_agreement_all, y_y_hat_agreement_all, confidence, dic):
    all_agree = 'all_agree'
    pl_agree = 'pl_agree'
    true_agree = 'true_agree'
    y_y_hat_agreement_all_st = 'y_y_hat_agreement'
    y_hat_y_tilda_agreement_all_st = 'y_hat_y_tilda_agreement'
    y_y_tilda_agreement_all_st = 'y_y_tilda_agreement'
    confidence_st = 'confidence'
    if all_agree not in dic:
        dic[all_agree] = {}
    if pl_agree not in dic:
        dic[pl_agree] = {}
    if true_agree not in dic:
        dic[true_agree] = {}
    if y_y_hat_agreement_all_st not in dic:
        dic[y_y_hat_agreement_all_st] = {}
    if y_hat_y_tilda_agreement_all_st not in dic:
        dic[y_hat_y_tilda_agreement_all_st] = {}
    if y_y_tilda_agreement_all_st not in dic:
        dic[y_y_tilda_agreement_all_st] = {}
    if confidence_st not in dic:
        dic[confidence_st] = {}
    dic[all_agree][bin_num] = all_agreement
    dic[pl_agree][bin_num] = y_tilda_y_hat_agreement
    dic[true_agree][bin_num] = y_y_hat_agreement
    dic[y_y_hat_agreement_all_st][bin_num] = y_y_hat_agreement_all
    dic[y_hat_y_tilda_agreement_all_st][bin_num] = y_hat_y_tilda_agreement_all
    dic[y_y_tilda_agreement_all_st][bin_num] = y_y_tilda_agreement_all
    dic[confidence_st][bin_num] = confidence


def calc_accuracy(bin_num, predictions, true_labels, labels, in_bin, confidence, dic):
    if bin_num == 0:
        dic['indexes'] = torch.zeros(len(true_labels))
    else:
        dic['indexes'] = torch.where(in_bin, (bin_num * torch.ones(len(true_labels)).to(torch.int)), dic['indexes'])
    predictions_in_bin = predictions[in_bin]
    labels_in_bin = labels[in_bin]
    true_labels_in_bin = true_labels[in_bin]
    only_y_hat_y_tilda_agree = 0
    only_y_hat_y_agree = 0
    all_agree = 0
    n = len(predictions_in_bin)
    for i in range(n):
        if (predictions_in_bin[i] == labels_in_bin[i]) and (predictions_in_bin[i] != true_labels_in_bin[i]):
            only_y_hat_y_tilda_agree += 1
        if (predictions_in_bin[i] != labels_in_bin[i]) and (predictions_in_bin[i] == true_labels_in_bin[i]):
            only_y_hat_y_agree += 1
        if (predictions_in_bin[i] == labels_in_bin[i]) and (predictions_in_bin[i] == true_labels_in_bin[i]):
            all_agree += 1
    # print(f"all agree: {all_agree}, agree with pl: {only_y_hat_y_tilda_agree}, agree with true label: {only_y_hat_y_agree}, total examples in bin: {len(predictions_in_bin)}, ratio: {abs(only_y_hat_y_tilda_agree - only_y_hat_y_agree) / len(predictions_in_bin)}")
    y_hat_y_tilda_agreement = sum(predictions_in_bin == labels_in_bin).item() / n
    y_hat_y_agreement = sum(predictions_in_bin == true_labels_in_bin).item() / n
    y_tilda_y_agreement = sum(labels_in_bin == true_labels_in_bin).item() / n
    add_stats(bin_num, all_agree / n, only_y_hat_y_agree / n, only_y_hat_y_tilda_agree / n, y_tilda_y_agreement,
              y_hat_y_tilda_agreement, y_hat_y_agreement, confidence.item(), dic)


def _print_stats_with_indexes(relevant_indexes_in_bin, true_labels, noisy_labels, predictions, sorted_indices_in_bin,
                              correctness, i, est_acc_in_bin, conf_in_bin):
    relevant_true_labels = true_labels[relevant_indexes_in_bin]
    relevant_noisy_labels = noisy_labels[relevant_indexes_in_bin]
    true_labels_in_bin = true_labels[sorted_indices_in_bin]
    predictions_in_bin = predictions[sorted_indices_in_bin]
    correctness_in_bin = predictions_in_bin.eq(true_labels_in_bin)
    accuracy_in_bin = correctness_in_bin.float().mean()
    relevant_noisy_acc = (relevant_noisy_labels == relevant_true_labels).float().mean()

    # Create a mask for elements not in indices_to_exclude
    mask = torch.ones(noisy_labels.size(0), dtype=torch.bool)
    mask[relevant_indexes_in_bin] = False
    mask = mask[sorted_indices_in_bin]

    # Filter the data using the mask
    filtered_data = correctness_in_bin[mask]
    not_noisy_acc = (filtered_data).float().mean()

    true_acc_in_relevant_indexes = (relevant_true_labels == predictions[relevant_indexes_in_bin]).float().mean()
    true_acc_not_in_relevant_indexes = (true_labels_in_bin[mask] == predictions_in_bin[mask]).float().mean()


    # print(f"true accuracy in bin {i} is {accuracy_in_bin}, relevant pl accuracy is: {relevant_noisy_acc}")
    print(
        f"bin {i} has {len(relevant_indexes_in_bin)} relevant indexes, estimate accuracy is {est_acc_in_bin}, true accuracy: {accuracy_in_bin}, pl accuracy: {relevant_noisy_acc}, other index acc est: {not_noisy_acc}, confidence is {conf_in_bin}, true acc in relevant indexes: {true_acc_in_relevant_indexes}, true acc not in relevant indexes: {true_acc_not_in_relevant_indexes}")


class CalibrationLoss(nn.Module):
    def __init__(self, n_bins=15, LOGIT=True, adaECE=False, true_labels=None):
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

                avg_confidence_in_bin = confidences[in_bin].mean()
                cur_ece = torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                ece += cur_ece
                if self.true_labels is not None:
                    true_acc_in_bin = self._calculate_accuracy_in_bin(in_bin, predictions.eq(self.true_labels), num_classes, predictions, self.true_labels,
                                                              None, None)
                    calc_accuracy(i, predictions, self.true_labels, labels, in_bin, avg_confidence_in_bin, self.stats)
        return ece

    # def forward(self, logits, labels, num_classes=10, epsilon=None, transition_matrix=None):
    #     if self.LOGIT:
    #         softmaxes = F.softmax(logits, dim=1)
    #     else:
    #         softmaxes = logits
    #     confidences, predictions = torch.max(softmaxes, 1)
    #     correctness = predictions.eq(labels)
    #     confidences[confidences == 1] = 0.999999
    #
    #     ece = torch.zeros(1, device=logits.device)
    #
    #     sorted_indices = torch.sort(confidences).indices
    #     sorted_confidences = confidences[sorted_indices]
    #     sorted_predictions = predictions[sorted_indices]
    #     sorted_correctness = correctness[sorted_indices]
    #     sorted_labels = labels[sorted_indices]
    #     bin_indexes = np.linspace(0, len(sorted_indices), self.nbins + 1).astype(int)
    #     bin_lowers = bin_indexes[:-1]
    #     bin_uppers = bin_indexes[1:]
    #
    #     indexes = torch.zeros(len(labels))
    #     for i in range(len(bin_lowers)):
    #         bin_lower = bin_lowers[i]
    #         bin_upper = bin_uppers[i]
    #
    #         # Calculated |confidence - accuracy| in each bin
    #         in_bin = np.zeros(len(sorted_indices), dtype=bool)
    #         in_bin[bin_lower:bin_upper] = True
    #
    #         in_bin = torch.from_numpy(in_bin)
    #         prop_in_bin = in_bin.float().mean()
    #
    #         if prop_in_bin.item() > 0 and (self.adaECE or in_bin.sum() > 20):
    #             accuracy_in_bin = self._calculate_accuracy_in_bin(in_bin, sorted_correctness, num_classes,
    #                                                               sorted_predictions, sorted_labels,
    #                                                               epsilon, transition_matrix)
    #
    #             avg_confidence_in_bin = sorted_confidences[in_bin].mean()
    #             if transition_matrix is not None:
    #                 print(f"NTS: bin {i} has {in_bin.sum()} examples, confidence is {avg_confidence_in_bin}, accuracy is {accuracy_in_bin}")
    #             cur_ece = torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    #             ece += cur_ece
    #             if self.true_labels is not None:
    #                 sorted_true_labels = self.true_labels[sorted_indices]
    #                 true_acc_in_bin = self._calculate_accuracy_in_bin(in_bin, sorted_predictions.eq(sorted_true_labels),
    #                                                                   num_classes, predictions, self.true_labels,
    #                                                                   None, None)
    #                 calc_accuracy(i, sorted_predictions, sorted_true_labels, sorted_labels, in_bin,
    #                               avg_confidence_in_bin, self.stats)
    #                 indexes_in_bin = sorted_indices[in_bin]
    #                 indexes[indexes_in_bin] = i
    #         self.stats['indexes'] = indexes
    #     return ece

    def forward_with_indexes(self, logits, labels, relevant_indexes, num_classes=10):
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits
        if self.true_labels is not None:
            print(f'acc relevant indexes: {sum(labels[relevant_indexes] == self.true_labels[relevant_indexes]) / len(relevant_indexes)}')
        confidences, predictions = torch.max(softmaxes, 1)
        correctness = predictions.eq(labels)
        confidences[confidences == 1] = 0.999999

        ece = torch.zeros(1, device=logits.device)

        sorted_indices = torch.sort(confidences).indices
        bin_indexes = np.linspace(0, len(sorted_indices), self.nbins + 1).astype(int)
        bin_lowers = bin_indexes[:-1]
        bin_uppers = bin_indexes[1:]

        indexes = torch.zeros(len(labels))
        for i in range(len(bin_lowers)):
            bin_lower = bin_lowers[i]
            bin_upper = bin_uppers[i]

            # Calculated |confidence - accuracy| in each bin
            in_bin = np.zeros(len(sorted_indices), dtype=bool)
            in_bin[bin_lower:bin_upper] = True

            sorted_indices_in_bin = sorted_indices[in_bin]
            relevant_indexes_in_bin = sorted_indices_in_bin[torch.isin(sorted_indices_in_bin, relevant_indexes)]
            not_relevant_indexes_in_bin = sorted_indices_in_bin[~torch.isin(sorted_indices_in_bin, relevant_indexes)]
            other_acc_in_bin = correctness[not_relevant_indexes_in_bin].float().mean()
            conf_in_bin = confidences[sorted_indices_in_bin].mean()
            acc_in_bin = correctness[relevant_indexes_in_bin].sum() / len(relevant_indexes_in_bin)
            cur_ece = torch.abs(conf_in_bin - acc_in_bin) * (sum(in_bin) / len(correctness))
            ece += cur_ece
            if (self.true_labels is not None):
                _print_stats_with_indexes(relevant_indexes_in_bin, self.true_labels, labels, predictions,
                                          sorted_indices_in_bin, correctness, i, acc_in_bin, conf_in_bin)
            self.stats['indexes'] = indexes
        return ece

    def forward_with_confidence(self, logits, labels, pl_conf, num_classes):
        '''
        calaculate weighted ece loss
        :param logits:
        :param labels:
        :param pl_conf:
        :param num_classes:
        :return:
        '''
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
                correctness_in_bin = correctness[in_bin]
                pl_conf_in_bin = pl_conf[in_bin]
                avg_confidence_in_bin = confidences[in_bin].mean()

                accuracy_in_bin = (correctness_in_bin*pl_conf_in_bin).sum() / pl_conf_in_bin.sum()

                cur_ece = torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                ece += cur_ece
                if self.true_labels is not None:
                    calc_accuracy(i, predictions, self.true_labels, labels, in_bin, avg_confidence_in_bin, self.stats)
        return ece


    # def forward_with_confidence(self, logits, labels, pl_conf):
    #     if self.LOGIT:
    #         softmaxes = F.softmax(logits, dim=1)
    #     else:
    #         softmaxes = logits
    #     confidences, predictions = torch.max(softmaxes, 1)
    #     correctness = predictions.eq(labels)
    #     confidences[confidences == 1] = 0.999999
    #
    #     ece = torch.zeros(1, device=logits.device)
    #
    #     sorted_indices = torch.sort(confidences).indices
    #     bin_indexes = np.linspace(0, len(sorted_indices), self.nbins + 1).astype(int)
    #     bin_lowers = bin_indexes[:-1]
    #     bin_uppers = bin_indexes[1:]
    #
    #     indexes = torch.zeros(len(labels))
    #     for i in range(len(bin_lowers)):
    #         bin_lower = bin_lowers[i]
    #         bin_upper = bin_uppers[i]
    #
    #         # Calculated |confidence - accuracy| in each bin
    #         in_bin = np.zeros(len(sorted_indices), dtype=bool)
    #         in_bin[bin_lower:bin_upper] = True
    #
    #         sorted_indices_in_bin = sorted_indices[in_bin]
    #         conf_in_bin = confidences[sorted_indices_in_bin].mean()
    #         correctness_in_bin = correctness[sorted_indices_in_bin]
    #         pl_conf_in_bin = pl_conf[sorted_indices_in_bin]
    #         acc_in_bin = (correctness_in_bin*pl_conf_in_bin).sum() / pl_conf_in_bin.sum()
    #         cur_ece = torch.abs(conf_in_bin - acc_in_bin) * (sum(in_bin) / len(correctness))
    #         ece += cur_ece
    #         self.stats['indexes'] = indexes
    #     return ece


    def forward_with_selected_indexes(self, logits, labels, selected_indexes, num_classes):
        ''''
        calculate ece using only part of the data set
        '''
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
                correctness_in_bin = correctness[in_bin]

                correctness_selected = correctness[selected_indexes]
                in_bin_selected = in_bin[selected_indexes]
                correctness_in_bin_and_selected = correctness_selected[in_bin_selected]
                accuracy_in_bin = correctness_in_bin_and_selected.sum() / in_bin_selected.sum()

                avg_confidence_in_bin = confidences[in_bin].mean()
                cur_ece = torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                ece += cur_ece
        return ece

    # def forward_with_selected_indexes(self, logits, labels, selected_indexes, num_classes=10):
    #     if self.LOGIT:
    #         softmaxes = F.softmax(logits, dim=1)
    #     else:
    #         softmaxes = logits
    #     confidences, predictions = torch.max(softmaxes, 1)
    #     correctness = predictions.eq(labels)
    #     confidences[confidences == 1] = 0.999999
    #
    #     ece = torch.zeros(1, device=logits.device)
    #
    #     sorted_indices = torch.sort(confidences).indices
    #     bin_indexes = np.linspace(0, len(sorted_indices), self.nbins + 1).astype(int)
    #     bin_lowers = bin_indexes[:-1]
    #     bin_uppers = bin_indexes[1:]
    #
    #     for i in range(len(bin_lowers)):
    #         bin_lower = bin_lowers[i]
    #         bin_upper = bin_uppers[i]
    #
    #         # Calculated |confidence - accuracy| in each bin
    #         in_bin = np.zeros(len(sorted_indices), dtype=bool)
    #         in_bin[bin_lower:bin_upper] = True
    #
    #         sorted_indices_in_bin = sorted_indices[in_bin]
    #         relevant_indexes_in_bin = sorted_indices_in_bin[torch.isin(sorted_indices_in_bin, selected_indexes)]
    #         not_relevant_indexes_in_bin = sorted_indices_in_bin[~torch.isin(sorted_indices_in_bin, selected_indexes)]
    #         other_acc_in_bin = correctness[not_relevant_indexes_in_bin].float().mean()
    #         conf_in_bin = confidences[sorted_indices_in_bin].mean()
    #         acc_in_bin = correctness[relevant_indexes_in_bin].sum() / len(relevant_indexes_in_bin)
    #         cur_ece = torch.abs(conf_in_bin - acc_in_bin) * (sum(in_bin) / len(correctness))
    #         ece += cur_ece
    #     return ece

    def _create_estimate_acc_function(self, bin_lowers, bin_uppers, sorted_indices, correctness):
        lowest_acc = self._estimate_bin(bin_lowers, bin_uppers, sorted_indices, correctness, 0)
        highest_acc = self._estimate_bin(bin_lowers, bin_uppers, sorted_indices, correctness, len(bin_lowers) - 1)
        b = lowest_acc
        a = (highest_acc - lowest_acc) / (len(bin_lowers) - 1)
        return lambda i: a * i + b

    def _estimate_bin(self, bin_lowers, bin_uppers, sorted_indices, correctness, i):
        bin_lower = bin_lowers[i]
        bin_upper = bin_uppers[i]

        # Calculated |confidence - accuracy| in each bin
        in_bin = np.zeros(len(sorted_indices), dtype=bool)
        in_bin[bin_lower:bin_upper] = True
        in_bin = torch.from_numpy(in_bin)
        accuracy_in_bin = correctness[in_bin].float().mean()
        accuracy_in_bin = self._normalize_acc(accuracy_in_bin)
        return accuracy_in_bin

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
            # bin_boundaries = self._calc_equal_bin_bounds(confidences.cpu().detach())
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

    def _calc_equal_bin_bounds(self, confidences):
        sorted_confidences = np.sort(confidences)
        npt = len(sorted_confidences)
        index = np.linspace(0, npt, self.nbins + 1).astype(int)[:-1]
        extracted_values = sorted_confidences[index]
        extracted_values[0] = 0
        extracted_values = np.append(extracted_values, 1)
        return extracted_values

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
