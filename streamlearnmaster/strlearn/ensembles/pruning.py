from builtins import range

import numpy as np
from sklearn import metrics

PRUNING_CRITERION = ('accuracy')

class OneOffPruner(object):
    def __init__(self, ensemble_support_matrix, y, pruning_criterion='accuracy'):
        self.pruning_criterion = pruning_criterion
        self.ensemble_support_matrix = ensemble_support_matrix
        self.y = y

        best_permutation = self.accuracy()

        self.best_permutation = best_permutation

    def accuracy(self):
        """
        Accuracy pruning.
        """
        candidates_no = self.ensemble_support_matrix.shape[0]

        loser = 0
        best_accuracy = 0.

        for cid in range(candidates_no):
            weights = np.array(
                [0 if i == cid else 1 for i in range(candidates_no)])
            weighted_support = self.ensemble_support_matrix * \
                weights[:, np.newaxis, np.newaxis]
            acumulated_weighted_support = np.sum(weighted_support, axis=0)
            decisions = np.argmax(acumulated_weighted_support, axis=1)
            accuracy = metrics.accuracy_score(self.y, decisions)
            if accuracy > best_accuracy:
                loser = cid
                best_accuracy = accuracy

        best_permutation = list(range(candidates_no))
        best_permutation.pop(loser)

        return best_permutation
