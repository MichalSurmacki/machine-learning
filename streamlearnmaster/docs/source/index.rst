######################################
Welcome to stream-learn documentation!
######################################

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   quickstart

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User guide

   streams_guide
   evaluators_guide
   ensembles_guide
   classifiers_guide

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   streams_api
   evaluators_api
   ensembles_api
   classifiers_api
   utils_api
   metrics_api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Other informations

   about
   cite

.. image:: plots/hello.gif

The ``stream-learn`` module is a set of tools necessary for processing data streams using ``scikit-learn`` estimators. The batch processing approach is used here, where the dataset is passed to the classifier in smaller, consecutive subsets called `chunks`. The module consists of five sub-modules:

- `streams <streams.html>`_ - containing a data stream generator that allows obtaining both stationary and dynamic distributions in accordance with various types of concept drift (also in the field of a priori probability, i.e. dynamically unbalanced data) and a parser of the standard ARFF file format.
- `evaluators <evaluators.html>`_ - containing classes for running experiments on stream data in accordance with the Test-Then-Train and Prequential methodology.
- `classifiers <classifiers.html>`_ - containing sample stream classifiers,
- `ensembles <ensembles.html>`_ - containing standard team models of stream data classification,
- `metrics <evaluators.html>`_ - containing typical classification quality metrics in data streams.

You can read more about each module in the User Guide.

`Getting started <quickstart.html>`_
------------------------------------

A brief description of the installation process and basic usage of the module in a simple experiment.

`API Documentation <api.html>`_
-------------------------------

Precise API description of all the classes and functions implemented in the module.

`Examples <auto_examples/index.html>`_
--------------------------------------

A set of examples illustrating the use of all module elements.

See the `README <https://github.com/w4k2/stream-learn/blob/master/README.md>`_
for more information.
