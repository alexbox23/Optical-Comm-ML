# 210A-project
SVM for optical communication over dispersive fiber

The goal of this project is to use support vector machines to classify a distorted signal into a digital signal.

Outline of Algorithm:

The time-domain signal is partitioned into sets of samples based on the bit rate. The SVM is trained on the samples after they have been cast as 16-dimensional feature vectors, where each dimension is a sample within the clock cycle.