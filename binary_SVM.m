function [epoch, total_loss, missed_bits,loss,w]=binary_SVM(train_syms,reg_pen,learning_rate,tolerance,cutoff)
% SVM for Binary NRZ RX data
% args:
  % train_syms: Number of symbols used in the training set, remainder to
    % be used in test set
  % reg_pen: l2 regularization penalty, lambda
  % learning rate: unnormalized learning rate parameter, mu
  % tolerance: hinge loss tolerance requirement to leave trainig mode
  % cutoff: multiplier of tolerance for when to reduce learning rate
% outputs:
  % epoch: total number of training epochs
  % total_loss: total hinge loss of testing set
  % missed_bits: incorrectly classified bits in the testing set
  % loss: vector of training hinge loss across epochs
  % w: optimized hyperplane vector
  
% Load data and labels from csv files
fid = fopen('data/data_Binary_NRZ_RX(small).csv');
data = textscan(fid, '%f %f', 'Delimiter', ',', 'HeaderLines', 7);
fclose(fid);
data = cell2mat(data);
here=1;
fid = fopen('data/labels_Binary_NRZ_TX.csv');
labels = textscan(fid, '%f', 'Delimiter', ',');
fclose(fid);
labels = cell2mat(labels);

% Determine number of samples per symbol
bit_length = 0.04; %time length of one bit (ns)
T = data(2,1); %sampling interval (ns)
bit_samples = bit_length/T; %number of samples in one bit

% Parse data and labels into one data vector
train_length=bit_samples*train_syms;
test_length = length(data) - train_length;
set=zeros(length(data),3);
for n=1:length(data)
    set(n,1) = mod(data(n,1), bit_length); %time wrt clock cycle (ns)
    set(n,2) = data(n,2); %electrical signal value
    set(n,3) = labels(floor(data(n,1)/bit_length) + 1); %label
end

% Set order of data to randomize index.  Randomizes across training and
% testing sets, causing cross-validation across runs.
order=randperm(length(data)/bit_samples);

% Initializations
w = zeros(bit_samples, 1);
b = 0;
lambda = reg_pen; %regularizer
epoch = 1;
loss = zeros(1,1);
hinge_loss = 1;

while hinge_loss >= tolerance % Exit when loss is low enough
    %Initializations
    hinge_loss = 0;
    sub_grad_w = zeros(bit_samples, 1);
    sub_grad_b = 0;
    
    % Training on w
    for i=1:train_length/bit_samples
        n=order(i); % Select data from randomized order
        x = set(bit_samples*(n-1)+1:bit_samples*n,2); % Pull data
        class = set(bit_samples*n,3); % Determine label
        if class == 0
            class = -1;
        end
        value = 1 - class * (dot(w, x) - b); % Determine sub-gradient for w
        if value > 0
            sub_grad_w = sub_grad_w - class * x;
        end
    end
    sub_grad_w = sub_grad_w/train_length + 2*lambda*w; % Scale sub-gradient
    w = w - learning_rate*sub_grad_w; % Update w
    
    % Training on b
    for i=1:train_length/bit_samples
        n=order(i); % Select data from randomized order
        x = set(bit_samples*(n-1)+1:bit_samples*n,2); % Pull data
        class = set(bit_samples*n,3); % Determine label
        if class == 0
            class = -1;
        end
        value = 1 - class * (dot(w, x) - b); % Determine sub-gradient for b
        if value > 0 
            sub_grad_b = sub_grad_b + class;
        end
    end
    sub_grad_b = sub_grad_b/train_length; % Scale sub-gradient
    b = b - learning_rate*sub_grad_b; % Update b
    
    % Determine hinge loss and check threshold
    for i=1:train_length/bit_samples 
        n=order(i); % Select data from randomized order
        x = set(bit_samples*(n-1)+1:bit_samples*n,2); % Pull data
        class = set(bit_samples*n,3); % Determine label
        if class == 0
            class = -1;
        end
        value = 1 - class * (dot(w, x) - b);
        hinge_loss = hinge_loss + max(0, value); % Determine hinge loss
    end
    hinge_loss = bit_samples*hinge_loss/train_length + lambda*norm(w)^2; %Scale hinge loss
    loss(epoch) = hinge_loss; % Store hinge loss
    epoch = epoch + 1;
    if and(hinge_loss < cutoff*tolerance,here) % Reduce learning rate if loss is low enough
        here=0;
        learning_rate = learning_rate/2;
    end
end

% Initialization
total_loss = 0;
missed_bits = 0;

% Testing
for i=train_length/bit_samples+1:train_length/bit_samples+test_length/bit_samples
    n=order(i); % Select data from randomized order
    x = set(bit_samples*(n-1)+1:bit_samples*n,2); % Pull data
    
    % Determine loss for each classification
    class = 1;
    hinge_loss_1 = max(0, 1 - class * (dot(w, x) - b));
    class = -1;
    hinge_loss_0 = max(0, 1 - class * (dot(w, x) - b));
    
    % Make prediction
    if hinge_loss_1 < hinge_loss_0
        prediction = 1;
        total_loss = total_loss + hinge_loss_1;
    else
        prediction = 0;
        total_loss = total_loss + hinge_loss_0;
    end
    
    % Check if correct
    if not(prediction == set(bit_samples*n,3))
        missed_bits = missed_bits + 1;
    end
end

% Scale total loss
total_loss = total_loss/test_length + lambda*norm(w)^2;
end