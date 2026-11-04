"""
Multiclass MLP classification: Iris and MNIST.

Trains a multilayer perceptron (sklearn MLPClassifier) with one hidden layer,
softmax output, and one-hot target encoding. Hidden activations: tanh / ReLU.
Solvers: adam, sgd, lbfgs.

The task is split into two parts:

Part 1 — Iris: 3 species from 4 flower measurements.
Part 2 — MNIST: handwritten digits 0-9; each 28x28 image flattened to a
784-dim input vector.

Hyperparameters are selected with GridSearchCV over:
  - hidden_layer_sizes: (20,), (40,), …, (100,)
  - learning_rate_init: 0.1, 0.01, 0.001
  - activation: tanh, relu
  - solver: adam, lbfgs, sgd

Results for both datasets are reported with confusion matrices.
"""