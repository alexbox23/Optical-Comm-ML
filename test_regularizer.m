train_length = 128;
learning_rate = 10;
tolerance = 0.1;

regularizers = 0:1e-6:3e-4;
missed_data = zeros(length(regularizers), 1);
loss_data = zeros(length(regularizers), 1);
epoch_data = zeros(length(regularizers), 1);
w_data = zeros(length(regularizers), 1);

for n=1:length(regularizers)
    reg_pen = regularizers(n);
    [epoch, total_loss, missed_bits,loss,w] = binary_SVM(train_length,reg_pen,learning_rate,tolerance);
    missed_data(n) = missed_bits;
    loss_data(n) = total_loss;
    epoch_data(n) = epoch;
    w_data(n) = norm(w);
end
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