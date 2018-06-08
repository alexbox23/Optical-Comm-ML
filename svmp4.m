function [epoch10,epoch01, total_loss, missed_bits,missed_syms,loss01,loss10]=svmp4(train_syms,reg_pen,learning_rate,tolerance,cutoff)
% SVM for 4-PAM RX data
% args:
  % train_syms: Number of symbols used in the training set, remainder to
    % be used in test set
  % reg_pen: l2 regularization penalty, lambda
  % learning rate: unnormalized learning rate parameter, mu
  % tolerance: hinge loss tolerance requirement to leave trainig mode
  % cutoff: multiplier of tolerance for when to reduce learning rate
% outputs:
  % epoch10: total number of training epochs (MSB)
  % epoch01: total number of training epochs (LSB)
  % total_loss: total hinge loss of testing set (sum MSB+LSB)
  % missed_bits: incorrectly classified bits in the testing set
  % missed_syms: incorrectly classified symbols in the testing set
  % loss01: vector of training hinge loss across epochs (LSB)
  % loss10: vector of training hinge loss across epochs (MSB)

% Load data and labels from csv files
fid = fopen('data/data_PAM4_RX(small).csv');
data = textscan(fid, '%f %f', 'Delimiter', ',', 'HeaderLines', 7);
fclose(fid);
data = cell2mat(data);
data_mean=mean(data(:,2));

fid = fopen('data/labels_PAM4_TX.csv');
labels = textscan(fid, '%f', 'Delimiter', ',');
fclose(fid);
labels = cell2mat(labels);

% Determine number of samples per symbol
learning_rate2=learning_rate;
bit_length = 0.04; %time length of one bit (ns)
T = data(2,1); %sampling interval (ns)
bit_samples = bit_length/T; %number of samples in one bit
train_length=train_syms*bit_samples;

% Randomize order of and parse data into one vector
% Note difference from binary SVM: here we actually randomize the order of
% the data vector.  In binary, we randomly re-index the vector.
order=randperm(length(data)/bit_samples);
newdata=zeros(length(data),1);
newlabels=zeros(length(labels),1);
for n=1:length(data)/bit_samples
    i=order(n);
    newdata((n-1)*16+1:n*16)=data((i-1)*16+1:i*16,2);
    newlabels(n)=labels(i);
end

% Flags
here01=1;
here10=1;

% Produce training and testing data specifically from entire data set
training_set = zeros(train_length, 3);
for n=1:train_length
    training_set(n,2) = newdata(n); %electrical signal value
    training_set(n,3) = newlabels(ceil(n/bit_samples)); %label
end

test_length = length(data) - train_length;
test_set = zeros(test_length, 3);
for n=1:test_length
    test_set(n,2) = newdata(n+train_length);
    test_set(n,3) = newlabels(ceil((n+train_length)/bit_samples));
end


% Initializations
w10 = ones(bit_samples, 1);
b10 = 1;
w01 = -ones(bit_samples, 1);
b01 = 1;
lambda = reg_pen; %regularizer
epoch10 = 1;
epoch01 = 1;
loss10 = zeros(1,1);
loss01 = zeros(1,1);
hinge_loss10 = 1;
hinge_loss01 = 1;

% Train MSB SVM
while hinge_loss10 >= tolerance % Exit when loss is low enough
    % Initializations
    hinge_loss10 = 0;
    sub_grad_w10 = zeros(bit_samples, 1);
    sub_grad_b10 = 0;
    
    %Training on w
    for n=1:train_length/bit_samples
        x = training_set(bit_samples*(n-1)+1:bit_samples*n,2); % Pull data
        class10 = training_set(bit_samples*n,3); % Determine label
        if or(class10 == 0,class10 ==1)
            class10 = -1;
        else
            class10 = 1;
        end
        value = 1 - class10 * (dot(w10, x) - b10); % Determine sub-gradient for w
        if value > 0
            sub_grad_w10 = sub_grad_w10 - class10 * x;
        end
    end
    sub_grad_w10 = bit_samples*sub_grad_w10/train_length + 2*lambda*w10; % Scale sub-gradient
    w10 = w10 - learning_rate*sub_grad_w10; % Update w
    
    %Training on b
    for n=1:train_length/bit_samples
        x = training_set(bit_samples*(n-1)+1:bit_samples*n,2); % Pull data
        class10 = training_set(bit_samples*n,3); % Determine label
        if or(class10 == 0,class10 ==1)
            class10 = -1;
        else
            class10 = 1;
        end
        value = 1 - class10 * (dot(w10, x) - b10); % Determine sub-gradient for b
        if value > 0
            sub_grad_b10 = sub_grad_b10 + class10;
        end
    end
    sub_grad_b10 = bit_samples*sub_grad_b10/train_length; % Scale sub-gradient
    b10 = b10 - learning_rate*sub_grad_b10; % Update b
   
    % Determine hinge loss and check threshold
    for n=1:train_length/bit_samples
        x = training_set(bit_samples*(n-1)+1:bit_samples*n,2); % Pull data
        class10 = training_set(bit_samples*n,3); % Determine label
        if or(class10 == 0,class10 ==1)
            class10 = -1;
        else
            class10 = 1;
        end
        value = 1 - class10 * (dot(w10, x) - b10);
        hinge_loss10 = hinge_loss10 + max(0, value); % Determine hinge loss
    end
    hinge_loss10 = bit_samples*hinge_loss10/train_length + lambda*norm(w10)^2; %Scale hinge loss
    loss10(epoch10) = hinge_loss10; % Store hinge loss
    epoch10 = epoch10 + 1;
    if hinge_loss10 < cutoff*tolerance % Reduce learning rate if loss is low enough
        if (here10)
            learning_rate = learning_rate/2;
            here10=0;
        end
    end
end

% Train LSB SVM
% Create offset training set.  Scales and takes magnitude of data.  This
% makes the LSB separable and solveable by the SVM.
offset_training_set=abs(training_set(:,2)'-data_mean*ones(1,length(training_set(:,2))));
while hinge_loss01 >= tolerance % Exit when loss is low enough
    % Initializations
    hinge_loss01 = 0;
    sub_grad_w01 = zeros(bit_samples, 1);
    sub_grad_b01 = 0;
    
    % Training on w
    for n=1:train_length/bit_samples
        x = offset_training_set(bit_samples*(n-1)+1:bit_samples*n); % Pull data
        class01 = training_set(bit_samples*n,3); % Determine label
        if or(class01 == 3,class01 ==1)
            class01 = 1;
        else
            class01 = -1;
        end
        value = 1 - class01 * (dot(w01, x) - b01); % Determine sub-gradient for w
        if value > 0
            sub_grad_w01 = sub_grad_w01 - class01 * x';
        end
    end
    sub_grad_w01 = bit_samples*sub_grad_w01/train_length + 2*lambda*w01; % Scale sub-gradient
    w01 = w01 - learning_rate2*sub_grad_w01; % Update w
    
    %Training on b
    for n=1:train_length/bit_samples
        x = offset_training_set(bit_samples*(n-1)+1:bit_samples*n); % Pull data
        class01 = training_set(bit_samples*n,3); % Determine label
        if or(class01 == 3,class01 ==1)
            class01 = 1;
        else
            class01 = -1;
        end
        value = 1 - class01 * (dot(w01, x) - b01); % Determine sub-gradient for b
        if value > 0
            sub_grad_b01 = sub_grad_b01 + class01;
        end
    end
    sub_grad_b01 = bit_samples*sub_grad_b01/train_length; % Scale sub-gradient
    b01 = b01 - learning_rate2*sub_grad_b01; % Update b
    
    % Determine hinge loss and check threshold
    for n=1:train_length/bit_samples
        x = offset_training_set(bit_samples*(n-1)+1:bit_samples*n); % Pull data
        class01 = training_set(bit_samples*n,3); % Determine label
        if or(class01 == 3,class01 ==1)
            class01 = 1;
        else
            class01 = -1;
        end
        value = 1 - class01 * (dot(w01, x) - b01);
        hinge_loss01 = hinge_loss01 + max(0, value); % Determine hinge loss
    end
    hinge_loss01 = bit_samples*hinge_loss01/train_length + lambda*norm(w01)^2; % Scale hinge loss
    loss01(epoch01) = hinge_loss01; % Store hinge loss
    epoch01 = epoch01 + 1;
    if hinge_loss01 < cutoff*tolerance % Reduce learning rate if loss is low enough
        if (here01)
            learning_rate2 = learning_rate2/2;
            here01=0;
        end
    end
end

% Initializations
total_loss = 0;
total_loss10 = 0;
prediction10(n) = zeros(1,1);
missed_bits = 0;
realbit=0;

% Testing
% Test on MSB
for n=1:test_length/bit_samples
    x = test_set(bit_samples*(n-1)+1:bit_samples*n,2); % Pull data
    
    % Determine loss for each classification
    class = 1;
    hinge_loss_1 = max(0, 1 - class * (dot(w10, x) - b10));
    class = -1;
    hinge_loss_0 = max(0, 1 - class * (dot(w10, x) - b10));
    
    % Make prediction for MSB
    if hinge_loss_1 < hinge_loss_0
        prediction10(n) = 1;
        total_loss10 = total_loss10 + hinge_loss_1;
    else
        prediction10(n) = 0;
        total_loss10 = total_loss10 + hinge_loss_0;
    end
    if or(test_set(bit_samples*n,3) == 0,test_set(bit_samples*n,3) ==1)
            realbit = 0;
        else
            realbit = 1;
    end
    
    % Check if correct
    if not(prediction10(n) == realbit)
        missed_bits = missed_bits + 1;
    end
end
% Scale total loss
total_loss10 = bit_samples*total_loss10/test_length + lambda*norm(w10)^2;

% Test on LSB
total_loss01 = 0;
prediction01(n) = zeros(1,1);
for n=1:test_length/bit_samples
    x = abs(test_set(bit_samples*(n-1)+1:bit_samples*n,2)-data_mean*ones(bit_samples,1)); % Pull data
    
    % Determine loss for each classification
    class = 1;
    hinge_loss_1 = max(0, 1 - class * (dot(w01, x) - b01));
    class = -1;
    hinge_loss_0 = max(0, 1 - class * (dot(w01, x) - b01));
    
    % Make prediction for LSB
    if hinge_loss_1 < hinge_loss_0
        prediction01(n) = 1;
        total_loss01 = total_loss01 + hinge_loss_1;
    else
        prediction01(n) = 0;
        total_loss01 = total_loss01 + hinge_loss_0;
    end
    
    % Check if correct
    if or(test_set(bit_samples*n,3) == 0,test_set(bit_samples*n,3) ==2)
            realbit = 0;
        else
            realbit = 1;
    end
    if not(prediction01(n) == realbit)
        missed_bits = missed_bits + 1;
    end
end

% Scale total loss
total_loss01 = bit_samples*total_loss01/test_length + lambda*norm(w01)^2;
total_loss=total_loss10+total_loss01;

% Determine number of incorrect symbols from incorrect bits
sym_predict(n)= zeros(1,1);
missed_syms=0;
for n=1:test_length/bit_samples
    if ((prediction01(n)==0)&(prediction10(n) == 0))
        sym_predict(n)=0;
    elseif ((prediction01(n)==1)&(prediction10(n) == 0))
        sym_predict(n)=1;
    elseif ((prediction01(n)==0)&(prediction10(n) == 1))
        sym_predict(n)=2;
    else
        sym_predict(n)=3;
    end
    if not(sym_predict(n) == test_set(bit_samples*n,3))
        missed_syms = missed_syms + 1;
    end
end

epoch01=epoch01-1;
epoch10=epoch10-1;
end