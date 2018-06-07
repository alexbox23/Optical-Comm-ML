function [epoch, total_loss, missed_bits,loss,w]=binary_SVM(train_length,reg_pen,learning_rate,tolerance)
% SVM for Binary NRZ RX data
%clearvars -except train_length reg_pen learning_rate tolerance
fid = fopen('data/data_Binary_NRZ_RX(small).csv');
data = textscan(fid, '%f %f', 'Delimiter', ',', 'HeaderLines', 7);
fclose(fid);
data = cell2mat(data);

fid = fopen('data/labels_Binary_NRZ_TX.csv');
labels = textscan(fid, '%f', 'Delimiter', ',');
fclose(fid);
labels = cell2mat(labels);

bit_length = 0.04; %time length of one bit (ns)
T = data(2,1); %sampling interval (ns)
bit_samples = bit_length/T; %number of samples in one bit

% train_portion = train_size/length(data); %proportion of data used for training
% train_length = floor(length(data) * train_portion);
training_set = zeros(train_length, 3);
for n=1:train_length
    training_set(n,1) = mod(data(n,1), bit_length); %time wrt clock cycle (ns)
    training_set(n,2) = data(n,2); %electrical signal value
    training_set(n,3) = labels(floor(data(n,1)/bit_length) + 1); %label
end

test_length = length(data) - train_length;
test_set = zeros(test_length, 3);
for n=1:test_length
    test_set(n,1) = mod(data(n+train_length,1), bit_length);
    test_set(n,2) = data(n+train_length,2);
    test_set(n,3) = labels(floor(data(n+train_length,1)/bit_length) + 1);
end

set=zeros(length(data),3);
for n=1:length(data)
    set(n,1) = mod(data(n,1), bit_length); %time wrt clock cycle (ns)
    set(n,2) = data(n,2); %electrical signal value
    set(n,3) = labels(floor(data(n,1)/bit_length) + 1); %label
end

order=randperm(length(data)/bit_samples);

%training
%disp('Training...')
w = zeros(bit_samples, 1);
b = 0;
lambda = reg_pen; %regularizer
epoch = 1;
loss = zeros(1,1);
hinge_loss = 1;

while hinge_loss >= tolerance
    hinge_loss = 0;
    sub_grad_w = zeros(bit_samples, 1);
    sub_grad_b = 0;
    for i=1:train_length/bit_samples
        n=order(i);
        x = set(bit_samples*(n-1)+1:bit_samples*n,2);
        class = set(bit_samples*n,3);
        if class == 0
            class = -1;
        end
        value = 1 - class * (dot(w, x) - b);
        if value > 0
            sub_grad_w = sub_grad_w - class * x;
        end
    end
    sub_grad_w = sub_grad_w/train_length + 2*lambda*w;
    w = w - learning_rate*sub_grad_w;
    
    for i=1:train_length/bit_samples
        n=order(i);
        x = set(bit_samples*(n-1)+1:bit_samples*n,2);
        class = set(bit_samples*n,3);
        if class == 0
            class = -1;
        end
        value = 1 - class * (dot(w, x) - b);
        if value > 0
            sub_grad_b = sub_grad_b + class;
        end
    end
    sub_grad_b = sub_grad_b/train_length;
    b = b - learning_rate*sub_grad_b;
    
    for i=1:train_length/bit_samples
        n=order(i);
        x = set(bit_samples*(n-1)+1:bit_samples*n,2);
        class = set(bit_samples*n,3);
        if class == 0
            class = -1;
        end
        value = 1 - class * (dot(w, x) - b);
        hinge_loss = hinge_loss + max(0, value);
    end
    hinge_loss = bit_samples*hinge_loss/train_length + lambda*norm(w)^2;
    loss(epoch) = hinge_loss;
    epoch = epoch + 1;
%     if hinge_loss < 2*tolerance
%         learning_rate = 1;
%     end
end

%disp('finished testing. epochs:')
%disp(epoch)
%testing
total_loss = 0;
missed_bits = 0;
for i=train_length/bit_samples+1:train_length/bit_samples+test_length/bit_samples
    n=order(i);
    x = set(bit_samples*(n-1)+1:bit_samples*n,2);
    class = 1;
    hinge_loss_1 = max(0, 1 - class * (dot(w, x) - b));
    class = -1;
    hinge_loss_0 = max(0, 1 - class * (dot(w, x) - b));
    if hinge_loss_1 < hinge_loss_0
        prediction = 1;
        total_loss = total_loss + hinge_loss_1;
    else
        prediction = 0;
        total_loss = total_loss + hinge_loss_0;
    end
    if not(prediction == set(bit_samples*n,3))
        missed_bits = missed_bits + 1;
    end
end
total_loss = bit_samples*total_loss/test_length + lambda*norm(w)^2;
% disp(missed_bits)
%disp('total loss:')
%disp(total_loss)
%disp('missed bits')
%disp(missed_bits)

end