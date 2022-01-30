import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

##############################################################

from streamlearnmaster.strlearn.streams import StreamGenerator
import random
import strlearn as sl
from sklearn.naive_bayes import GaussianNB
import uuid
import os

counter = 0

def saveChart(filename, clfs, clf_names, metrics, metrics_names, evaluator):
    fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
    for m, metric in enumerate(metrics):
        #print(metrics_names[m])
        ax.set_title(metrics_names[m])
        ax.set_ylim(0, 1)
        for i, clf in enumerate(clfs):
            ax.plot(evaluator.scores[i, :, m], label=clf_names[i])
            plt.ylabel("Metric")
            plt.xlabel("Chunk")
            ax.legend()
    plt.savefig("results_OOB_UOB/" + filename + ".png")

modes = [0,1,2,3]
myWeights = [[0.05, 0.95], [0.10, 0.90], [0.15, 0.85]]
streamsData = []
clf_names = [
            "OOB",
            "UOB"
        ]
metrics_names = [
            "G-mean"
        ]

for i in range(3):
    for j in range(10):
        randomState = random.randint(10,10000)

        stream = StreamGenerator(n_chunks=200,
                                        chunk_size=500,
                                        n_classes=2,
                                        n_drifts=0,
                                        n_features=10,
                                        random_state=randomState,
                                        weights=myWeights[i])
        streamsData.append((stream.X, stream.y))
        stream.rstate = randomState
        stream.mode = 0

        print(randomState, myWeights[i], i, j)

        
        clfs = [
            sl.ensembles.OOB(base_estimator=GaussianNB()),
            sl.ensembles.UOB(base_estimator=GaussianNB())
        ]
        metrics = [
            sl.metrics.geometric_mean_score_1
        ]
        evaluator = sl.evaluators.TestThenTrain(metrics)
        evaluator.process(stream, clfs)
        name = f"{i}_{j}_{randomState}_{0}_{myWeights[i][0]}_{myWeights[i][1]}_{counter}"
        counter += 1
        saveChart(name, clfs, clf_names, metrics, metrics_names, evaluator)
        np.save("scores_OOB_UOB/" + name + "_scores", evaluator.scores)

exit()