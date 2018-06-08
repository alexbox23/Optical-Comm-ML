% test file for desynchronization
% We run train SVMs and test their performance when the RX data is offset.
% Missed symbols for both Binary and PAM4 SVMs are plotted against the
% number of samples by which the receiver is lagging behind.

train_length = 1024;
reg_pen = 0;
learning_rate = 10;
tolerance = 0.01;
slowdown = true;

offset_data = 0:16;
missed_bin = zeros(length(offset_data),1);
missed_pam = zeros(length(offset_data),1);

shuffle = false;

% parse binary data into training and test sets
file_RX = 'data/data_Binary_NRZ_RX(small).csv';
file_labels = 'data/labels_Binary_NRZ_TX.csv';
[training_set_bin, test_set_bin] = data_parser(file_RX, file_labels, train_length, shuffle);
% train binary SVM
class_pos = [1];
[epoch, loss, w_bin, b_bin]=SVM_train(training_set_bin, class_pos, learning_rate, tolerance, reg_pen, slowdown);

% parse pam4 data into training and test sets
file_RX = 'data/data_PAM4_RX(small).csv';
file_labels = 'data/labels_PAM4_TX.csv';
[training_set_pam, test_set_pam] = data_parser(file_RX, file_labels, train_length, shuffle);
% folding transformation for pam4 data
data_mean = mean(vertcat(training_set_pam(:,2), test_set_pam(:,2)));
training_set_pam(:,2) = abs(training_set_pam(:,2) - data_mean*ones(train_length,1));
test_set_pam(:,2) = abs(test_set_pam(:,2) - data_mean*ones(length(test_set_pam(:,2)),1));
% train pam4 SVM
class_pos = [0 2];
[epoch, loss, w_pam, b_pam]=SVM_train(training_set_pam, class_pos, learning_rate, tolerance, reg_pen, slowdown);

offset_bin = test_set_bin;
offset_pam = test_set_pam;
for n=1:length(offset_data)
    % test trained SVMs
    class_pos = [1];
    [avg_loss, misclass]=SVM_test(offset_bin, class_pos, w_bin, b_bin, reg_pen);
    missed_bin(n) = misclass;
    
    class_pos = [0 2];
    [avg_loss, misclass]=SVM_test(offset_pam, class_pos, w_pam, b_pam, reg_pen);
    missed_pam(n) = misclass;
    
    % offset the test set
    offset_bin(:,2) = padarray(offset_bin(2:end,2),1,0,'post');
    offset_pam(:,2) = padarray(offset_pam(2:end,2),1,0,'post');
end

% plot data
hold on
title('Effect of desynchronization on SVM classification')
xlabel('Offset')
ylabel('Missed Symbols')
xlim([0 offset_data(end)])
plot(offset_data, missed_bin, offset_data, missed_pam)
legend('Binary','4-PAM')