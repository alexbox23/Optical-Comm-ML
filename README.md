# Optical-Comm-ML
Machine learning for optical communication over a dispersive fiber

The goal of this project is to use machine learning techniques to decode a distorted signal received through a simulated optical fiber.\
We examine two encodings: binary (NRZ) and 4-bit (4-PAM)

Maximum Likelihood Estimator (Matlab)\
We assume each time sample within a clock cycle has a Gaussian distribution.\
An input sample is scaled by the variance to find the deviation from the mean of each symbol, and the prediction is given by the nearest symbol.

Convolutional Neural Network (Python)\
A CNN typically used for 2D image classification is modified to classify 1D waveforms.\
Instead of 2D convolutions, we use 1D dilated causal convolutions.

Support Vector Machine (Matlab)\
The time-domain signal is partitioned into sets of samples based on the bit rate.\
The SVM is trained on the samples after they have been cast as 16-dimensional feature vectors, where each dimension is a sample within the clock cycle.

Dependencies\
>Matlab: None\
>Python: Python 3.6, Numpy 1.14, TensorFlow 1.6

Feature Extraction files\
auto_labeler.m - run program to generate label file from TX file\
data_parser.m - function to parse RX and label files into training and test arrays

Maximum Likelihood Estimator files\
MLbin.m - runs max likelihood on binary data\
MLpam.m - runs max likelihood on 4-PAM data

Convolutional Neural Net files\
dilated_neural_net.py - defines helper functions for training the CNN\
test_nn_binary.py - run to train CNN to classify binary data\
test_nn_pam4.py - run to train CNN to classify 4-PAM data\
To run visualizations, use the following command (alternatively python -m tensorboard.main)\
>tensorboard --logdir=/tmp/dilated_cnn_model_binary\
or\
>tensorboard --logdir=/tmp/dilated_cnn_model_pam4\
Once TensorBoard is running, navigate your web browser to localhost:6006 to view the TensorBoard.


Support Vector Machine files\
SVM_train.m - function to train an SVM hyperplane using training array and hyperparameters\
SVM_test.m - function to test an SVM hyperplane using test array and hyperparameters\
binary_SVM.m - run SVM for Binary RX data\
svmp4.m - run SVM for 4-PAM RX data

svmp4_helper.m - Calls svmp4.m N times and returns averages\
binary_SVM_helper.m - Calls binary_SVM.m N times and returns averages\
hard_margin_main.m - Collects and plots data for binary and 4-PAM SVM\
test_regularizer.m - run program to test soft-margin SVM\
test_desynch.m - run program to test the effects of desynchronization

Gaussian Radial Basis Kernel files\
kernelbin_SVM.m -this function contains the RBF kernel code for binary data training and testing\
kernelpam_SVM.m -this function contains the RBF kernel code for 4PAM data training and testing\
kernelfindparam.m -m file that deploys the binary kernel. Oneshot variable determines if a tuning sweep is to be performed using variable vectors, otherwise the function will be run once with our default tuned values.\
kernelfindparampam.m  -m file that deploys the 4PAM kernel.\

