%% hard_data_main.m
% Collects and plots data for binary and 4-PAM SVM
% Ability to vary training set size, tolerance, learning rate, and learning
% rate reduction cutoff
% Calls helper functions which in turn call SVMs

%% Data Collection

%tolerance=logspace(-5,0,21);
tolerance=[1,2,3,4,5,10,20,30,40,50];
%tolerance=[2,4,8,16,32,48,64,96,128,192,256,384,512,768];

% Storage initializers
time2=zeros(length(tolerance),1);
epochs2=zeros(length(tolerance),1);
totalLoss2=zeros(length(tolerance),1);
missedSyms2=zeros(length(tolerance),1);

time4=zeros(length(tolerance),1);
epochs4MSB=zeros(length(tolerance),1);
epochs4LSB=zeros(length(tolerance),1);
totalLoss4=zeros(length(tolerance),1);
missedSyms4=zeros(length(tolerance),1);

for i=1:length(tolerance)
[epochs2(i), totalLoss2(i), missedSyms2(i),loss2{i},~,time2(i)]=binary_SVM_helper(64,0.0000,10,0.01,3, tolerance(i));
[epochs4MSB(i),epochs4LSB(i), totalLoss4(i),~,missedSyms4(i),loss4{i},loss4m{i},time4(i)]=svmp4_helper(64,0.0000,10,0.01,3, tolerance(i));
end

%% Complexity and Accuracy Plotting
figure
yyaxis left
plot(tolerance, time2);
hold on
plot(tolerance, time4);
ylabel('Run Time (s)')
yyaxis right
plot(tolerance, epochs2);
hold on
plot(tolerance, epochs4MSB);
plot(tolerance, epochs4LSB);
xlim([0 50]);
ylabel('Training Epochs')
xlabel('Learning Rate Reduction Cutoff');
title({'Complexity Dependence on Learning Rate Reduction Cutoff';'Training Set Size 64, Tolerance 0.01, Initial Rate 10'});
legend('Binary Run Time','4-PAM Run Time','Binary Epochs','4-PAM Epochs (MSB)','4-PAM Epochs (LSB)','Location','Northwest');

figure
yyaxis left
plot(tolerance, totalLoss2);
hold on
plot(tolerance, totalLoss4);
ylabel('Testing Total Loss')
yyaxis right
plot(tolerance, missedSyms2);
hold on
plot(tolerance, missedSyms4);
xlim([0 50]);
ylabel('Incorrect Symbols')
xlabel('Learning Rate Reduction Cutoff');
title({'Accuracy Dependence on Learning Rate Reduction Cutoff';'Training Set Size 64, Tolerance 0.01, Initial Rate 10'});
legend('Binary Loss','4-PAM Loss','Binary Misses','4-PAM Misses','Location','Northeast');

%% Loss Plotting

figure
for i=[1,6,10]
    plot(loss2{i})
    hold on
end
xlabel('Epoch');
ylabel('Loss');
title({'Loss Diminution Dependence on Learning Rate Reduction Cutoff (Binary)';'Learning Rate 10, Termination Threshold 0.01'});
legend('Reduction Cutoff 1','Reduction Cutoff 10','Reduction Cutoff 50','Location','Northeast');

figure
for i=[1,6,10]
    plot(loss4{i})
    hold on
end
xlabel('Epoch');
ylabel('Loss');
title({'Loss Diminution Dependence on Learning Rate Reduction Cutoff (LSB)';'Learning Rate 10, Termination Threshold 0.01'});
legend('Reduction Cutoff 1','Reduction Cutoff 10','Reduction Cutoff 50','Location','Northeast');

figure
for i=[1,6,10]
    plot(loss4m{i})
    hold on
end
xlabel('Epoch');
ylabel('Loss');
title({'Loss Diminution Dependence on Learning Rate Reduction Cutoff (MSB)';'Learning Rate 10, Termination Threshold 0.01'});
legend('Reduction Cutoff 1','Reduction Cutoff 10','Reduction Cutoff 50','Location','Northeast');