# 210A-project
SVM for optical communication over dispersive fiber

The goal of this project is to use support vector machines to classify a distorted signal into a digital signal.

Adapted from https://www.mathworks.com/matlabcentral/fileexchange/63158-support-vector-machine

various kernels: https://www.mathworks.com/matlabcentral/fileexchange/63033-svm-using-various-kernels

nonlinear classification: https://www.mathworks.com/matlabcentral/fileexchange/63024-svm-for-nonlinear-classification

Outline of Algorithm:

The time-domain signal is fed into the SVM, where each time sample is an instance of a 1D feature.
The SVM will return the classification for each sample (0 or 1 in the binary case).
A digital string of data is created by taking the class that is the majority within each clock cycle.

TODO

- understand and clean code, consolidate into a single matlab file

- label binary dataset and test code

- implement multi-class and test on PAM4 dataset

- test on more datasets found online (time permitting)