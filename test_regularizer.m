% test file for regularizer on binary SVM
% We run train multiple SVMs with different regularization penalties
% (lambda) to see the effect of soft-margin. The norm of the hyperplane 
% weight vector is plotted against lambda.

train_length = 128;
learning_rate = 10;
tolerance = 0.1;
slowdown = true;
class_pos = [1];

% parse data into training and test sets
file_RX = 'data/data_Binary_NRZ_RX(small).csv';
file_labels = 'data/labels_Binary_NRZ_TX.csv';
shuffle = true;
[training_set, test_set] = data_parser(file_RX, file_labels, train_length, shuffle);

regularizers = 0:1e-6:3e-4;
missed_data = zeros(length(regularizers), 1);
loss_data = zeros(length(regularizers), 1);
epoch_data = zeros(length(regularizers), 1);
w_data = zeros(length(regularizers), 1);

for n=1:length(regularizers)
    % train SVM with regularizer
    reg_pen = regularizers(n);
    [epoch, loss, w, b]=SVM_train(training_set, class_pos, learning_rate, tolerance, reg_pen, slowdown);
    epoch_data(n) = epoch;
    w_data(n) = norm(w);
    
    % test SVM performance 
    [avg_loss, misclass]=SVM_test(test_set, class_pos, w, b, reg_pen);
    loss_data(n) = avg_loss;
    missed_data(n) = misclass;
end

% plot data
hold on
title('Soft-margin SVM Training Results vs. Regularizer (Binary)')
xlabel('Regularizer \lambda')
xlim([0 regularizers(end)])
yyaxis right
ylabel('Loss')
plot(regularizers, loss_data)
yyaxis left
ylabel('||w||')
plot(regularizers, w_data)