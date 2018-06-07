rng(2018);
disp('need to add offsets to function args to run');
train_length = 128;
reg_pen = 0;
learning_rate = 10;
tolerance = 0.01;

offset_data = 0:16;
missed_bin = zeros(length(offset_data),1);
missed_pam = zeros(length(offset_data),1);

for n=1:length(offset_data)
    offset = offset_data(n);
    [epoch, total_loss, missed_bits,loss,w]=binary_SVM(train_length,reg_pen,learning_rate,tolerance,offset);
    missed_bin(n) = missed_bits;
    [epoch10,epoch01, total_loss, missed_bits,missed_syms,loss01,loss10]=svmp4(train_length,reg_pen,learning_rate,tolerance, offset);
    missed_pam(n) = missed_syms;
end

hold on
title('Effect of decoherence on SVM classification')
xlabel('Offset')
ylabel('Missed Symbols')
xlim([0 offset_data(end)])
plot(offset_data, missed_bin, offset_data, missed_pam)
legend('Binary','4-PAM')