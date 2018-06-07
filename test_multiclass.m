train_length = 128;
reg_pen = 0;
learning_rate = 10;
tolerance = 0.01;
slowdown = false;

file_RX = 'data/data_PAM4_RX(small).csv';
file_labels = 'data/labels_PAM4_TX.csv';

shuffle = false;
[training_set, test_set]=data_parser(file_RX, file_labels, train_length, shuffle);

data = vertcat(training_set, test_set);
data_mean = mean(data(:,2));

class_pos = [0 1];
[epoch, loss, w_msb, b_msb] = SVM_train(training_set, class_pos, learning_rate, tolerance, reg_pen, slowdown);
disp(class_pos);
disp(epoch);

training_set(:,2) = abs(training_set(:,2)-data_mean*ones(length(training_set(:,2)), 1));

class_pos = [0 3];
[epoch, loss, w_lsb, b_lsb] = SVM_train(training_set, class_pos, learning_rate, tolerance, reg_pen, slowdown);
disp(class_pos);
disp(epoch);

class_pos = [0 1];
[avg_loss, misclass] = SVM_test(test_set, class_pos, w_msb, b_msb, reg_pen);
disp(avg_loss);
disp(misclass);

test_set(:,2) = abs(test_set(:,2)-data_mean*ones(length(test_set(:,2)), 1));

class_pos = [0 3];
[avg_loss, misclass] = SVM_test(test_set, class_pos, w_lsb, b_lsb, reg_pen);
disp(avg_loss);
disp(misclass);

