train_length = 2048*16/2;
reg_pen = 1e-4;
learning_rate = 10;
tolerance = 0.1;
slowdown = true;

file_RX = 'data/data_PAM4_RX(small).csv';
file_labels = 'data/labels_PAM4_TX.csv';

shuffle = false;
[training_set, test_set]=data_parser(file_RX, file_labels, train_length, shuffle);

class_pos = [0];
[epoch, loss, w0, b0] = SVM_train(training_set, class_pos, learning_rate, tolerance, reg_pen, slowdown);
disp(class_pos);
disp(epoch);
disp(loss);

class_pos = [1];
[epoch, loss, w1, b1] = SVM_train(training_set, class_pos, learning_rate, tolerance, reg_pen, slowdown);
disp(class_pos);
disp(epoch);
disp(loss);

class_pos = [2];
[epoch, loss, w2, b2] = SVM_train(training_set, class_pos, learning_rate, tolerance, reg_pen, slowdown);
disp(class_pos);
disp(epoch);
disp(loss);

class_pos = [3];
[epoch, loss, w3, b3] = SVM_train(training_set, class_pos, learning_rate, tolerance, reg_pen, slowdown);
disp(class_pos);
disp(epoch);
disp(loss);

weights = [w0 w1 w2 w3];
biases = [b0 b1 b2 b3];
classes = [0 1 2 3];
for c=1:4
    [avg_loss, misclass] = SVM_test(test_set, [classes(c)], weights(c), biases(c), reg_pen);
    disp(c);
    disp(avg_loss);
    disp(misclass);
end