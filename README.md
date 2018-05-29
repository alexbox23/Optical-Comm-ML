# 210A-project
SVM for optical communication over dispersive fiber

The goal of this project is to use support vector machines to classify a distorted signal into a digital signal.

reference code: https://www.mathworks.com/matlabcentral/fileexchange/63158-support-vector-machine

various kernels: https://www.mathworks.com/matlabcentral/fileexchange/63033-svm-using-various-kernels

nonlinear classification: https://www.mathworks.com/matlabcentral/fileexchange/63024-svm-for-nonlinear-classification

Outline of Algorithm:

The time-domain signal is partitioned into sets of samples based on the bit rate. The SVM is trained on the samples after they have been cast as 2D features. The y-axis is the signal value and the x-axis is the time from the beginning of the clock cycle.
The SVM will return the classification for each sample (0 or 1 in the binary case). A digital string of data is created using the majority of the classes within each clock cycle.

TODO

- implement basic SVM

- experiment with soft-margin and non-linear kernels

- implement multi-class and test on PAM4 dataset

- test on more datasets found online (time permitting)