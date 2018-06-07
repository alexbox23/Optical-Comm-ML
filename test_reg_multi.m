train_length = 1024;
learning_rate = 10;
tolerance = 0.1;
slowdown = true;

file_RX = 'data/data_PAM4_RX(small).csv';
file_labels = 'data/labels_PAM4_TX.csv';

shuffle = false;
[training_set, test_set]=data_parser(file_RX, file_labels, train_length, shuffle);


class_pos = [0 2];
offset_training_set = training_set;
offset_training_set(:,2) = abs(training_set(:,2)-data_mean*ones(length(training_set(:,2)), 1));
offset_test_set = test_set;
offset_test_set(:,2) = abs(test_set(:,2)-data_mean*ones(length(test_set(:,2)), 1));

data = vertcat(training_set, test_set);
data_mean = mean(data(:,2));

regularizers = 0:1e-7:1e-5;
missed_data = zeros(length(regularizers), 1);
loss_data = zeros(length(regularizers), 1);
epoch_data = zeros(length(regularizers), 1);
w_data = zeros(length(regularizers), 1);

for n=1:length(regularizers)
    disp(n);
    reg_pen = regularizers(n);
   
    [epoch, loss, w_lsb, b_lsb] = SVM_train(offset_training_set, class_pos, learning_rate, tolerance, reg_pen, slowdown);
    
    [avg_loss, misclass] = SVM_test(offset_test_set, class_pos, w_lsb, b_lsb, reg_pen);
    
    missed_data(n) = misclass;
    loss_data(n) = avg_loss;
    epoch_data(n) = epoch;
    w_data(n) = norm(w_lsb);
end
    
hold on
title('Soft-margin SVM Training Results vs. Regularizer (PAM4)')
xlabel('Regularizer \lambda')
xlim([0 regularizers(end)])
yyaxis right
ylabel('LSB Loss')
plot(regularizers, loss_data)
yyaxis left
ylabel('||w_{LSB}||')
plot(regularizers, w_data)

