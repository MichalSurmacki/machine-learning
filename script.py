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
    plt.savefig("results/" + filename + ".png")

modes = [0,1,2,3]
myWeights = [[0.05, 0.95], [0.10, 0.90], [0.15, 0.85]]
streamsData = []
clf_names = [
            "AWE",
            "KMC",
            "NIE"
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
        xxx = stream.X
        yyy = stream.y

        print(randomState, myWeights[i], i, j)

        
        clfs = [
            sl.ensembles.AWE(base_estimator=GaussianNB()),
            sl.ensembles.KMC(base_estimator=GaussianNB()),
            sl.ensembles.NIE(base_estimator=GaussianNB())
        ]
        metrics = [
            sl.metrics.geometric_mean_score_1
        ]
        evaluator = sl.evaluators.TestThenTrain(metrics)
        evaluator.process(stream, clfs)
        name = f"{i}_{j}_{randomState}_{0}_{myWeights[i][0]}_{myWeights[i][1]}_{counter}"
        counter += 1
        saveChart(name, clfs, clf_names, metrics, metrics_names, evaluator)
        np.save("scores/" + name + "_scores", evaluator.scores)

        stream = StreamGenerator(n_chunks=200,
                                        chunk_size=500,
                                        n_classes=2,
                                        n_drifts=0,
                                        n_features=10,
                                        random_state=randomState,
                                        weights=myWeights[i])
        stream.X = xxx
        stream.y = yyy
        stream.rstate = randomState
        stream.mode = 1
        clfs = [
            sl.ensembles.AWE(base_estimator=GaussianNB()),
            sl.ensembles.KMC(base_estimator=GaussianNB()),
            sl.ensembles.NIE(base_estimator=GaussianNB())
        ]
        metrics = [
            sl.metrics.geometric_mean_score_1
        ]
        evaluator = sl.evaluators.TestThenTrain(metrics)
        evaluator.process(stream, clfs)
        name = f"{i}_{j}_{randomState}_{1}_{myWeights[i][0]}_{myWeights[i][1]}_{counter}"
        counter += 1
        saveChart(name, clfs, clf_names, metrics, metrics_names, evaluator)
        np.save("scores/" + name + "_scores", evaluator.scores)


        stream = StreamGenerator(n_chunks=200,
                                        chunk_size=500,
                                        n_classes=2,
                                        n_drifts=0,
                                        n_features=10,
                                        random_state=randomState,
                                        weights=myWeights[i])
        stream.X = xxx
        stream.y = yyy
        stream.rstate = randomState
        stream.mode = 2
        clfs = [
            sl.ensembles.AWE(base_estimator=GaussianNB()),
            sl.ensembles.KMC(base_estimator=GaussianNB()),
            sl.ensembles.NIE(base_estimator=GaussianNB())
        ]
        metrics = [
            sl.metrics.geometric_mean_score_1
        ]
        evaluator = sl.evaluators.TestThenTrain(metrics)
        evaluator.process(stream, clfs)
        name = f"{i}_{j}_{randomState}_{2}_{myWeights[i][0]}_{myWeights[i][1]}_{counter}"
        counter += 1
        saveChart(name, clfs, clf_names, metrics, metrics_names, evaluator)
        np.save("scores/" + name + "_scores", evaluator.scores)


        stream = StreamGenerator(n_chunks=200,
                                        chunk_size=500,
                                        n_classes=2,
                                        n_drifts=0,
                                        n_features=10,
                                        random_state=randomState,
                                        weights=myWeights[i])
        stream.X = xxx
        stream.y = yyy
        stream.rstate = randomState
        stream.mode = 3
        clfs = [
            sl.ensembles.AWE(base_estimator=GaussianNB()),
            sl.ensembles.KMC(base_estimator=GaussianNB()),
            sl.ensembles.NIE(base_estimator=GaussianNB())
        ]
        metrics = [
            sl.metrics.geometric_mean_score_1
        ]
        evaluator = sl.evaluators.TestThenTrain(metrics)
        evaluator.process(stream, clfs)
        name = f"{i}_{j}_{randomState}_{3}_{myWeights[i][0]}_{myWeights[i][1]}_{counter}"
        counter += 1
        saveChart(name, clfs, clf_names, metrics, metrics_names, evaluator)
        np.save("scores/" + name + "_scores", evaluator.scores)

exit()




stream.rstate = 1234

stream.mode = 0

#dla kazdego strumienia robic to w pretli




evaluator = sl.evaluators.TestThenTrain(metrics)
evaluator.process(stream, clfs)

fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
for m, metric in enumerate(metrics):
    ax[m].set_title(metrics_names[m])
    ax[m].set_ylim(0, 1)
    for i, clf in enumerate(clfs):
        ax[m].plot(evaluator.scores[i, :, m], label=clf_names[i])
    plt.ylabel("Metric")
    plt.xlabel("Chunk")
    ax[m].legend()
plt.savefig("normal")

stream.reset()
stream.mode = 1

evaluator = sl.evaluators.TestThenTrain(metrics)
evaluator.process(stream, clfs)

fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
for m, metric in enumerate(metrics):
    ax[m].set_title(metrics_names[m])
    ax[m].set_ylim(0, 1)
    for i, clf in enumerate(clfs):
        ax[m].plot(evaluator.scores[i, :, m], label=clf_names[i])
    plt.ylabel("Metric")
    plt.xlabel("Chunk")
    ax[m].legend()
plt.savefig("over")
