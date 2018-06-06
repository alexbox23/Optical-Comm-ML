% SVM for Binary NRZ RX data
clc;
clear;
fid = fopen('data_PAM4_RX(small).csv');
data = textscan(fid, '%f %f', 'Delimiter', ',', 'HeaderLines', 7);
fclose(fid);
data = cell2mat(data);

fid = fopen('data/labels_PAM4_TX.csv');
labels = textscan(fid, '%f', 'Delimiter', ',');
fclose(fid);
labels = cell2mat(labels);

bit_length = 0.04; %time length of one bit (ns)
T = data(2,1); %sampling interval (ns)
bit_samples = bit_length/T; %number of samples in one bit

train_portion = 0.125; %proportion of data used for training
train_length = floor(length(data) * train_portion);
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

%training
disp('training...')
w10 = ones(16, 1);
b10 = 1;
w01 = -ones(16, 1);
b01 = 1;
lambda = 0; %regularizer
learning_rate = 10;
epoch10 = 1;
epoch01 = 1;
loss10 = zeros(1,1);
loss01 = zeros(1,1);
hinge_loss10 = 1;
hinge_loss01 = 1;
tolerance = 0.05;

%Train MSB SVM
while hinge_loss10 >= tolerance
    hinge_loss10 = 0;
    sub_grad_w10 = zeros(16, 1);
    sub_grad_b10 = 0;
    for n=1:train_length/16
        x = training_set(16*(n-1)+1:16*n,2);
        class10 = training_set(16*n,3);
        if or(class10 == 0,class10 ==1)
            class10 = -1;
        else
            class10 = 1;
        end
        value = 1 - class10 * (dot(w10, x) - b10);
        if value > 0
            sub_grad_w10 = sub_grad_w10 - class10 * x;
        end
    end
    sub_grad_w10 = 16*sub_grad_w10/train_length + 2*lambda*w10;
    w10 = w10 - learning_rate*sub_grad_w10;
    
    for n=1:train_length/16
        x = training_set(16*(n-1)+1:16*n,2);
        class10 = training_set(16*n,3);
        if or(class10 == 0,class10 ==1)
            class10 = -1;
        else
            class10 = 1;
        end
        value = 1 - class10 * (dot(w10, x) - b10);
        if value > 0
            sub_grad_b10 = sub_grad_b10 + class10;
        end
    end
    sub_grad_b10 = 16*sub_grad_b10/train_length;
    b10 = b10 - learning_rate*sub_grad_b10;
    
    for n=1:train_length/16
        x = training_set(16*(n-1)+1:16*n,2);
        class10 = training_set(16*n,3);
        if or(class10 == 0,class10 ==1)
            class10 = -1;
        else
            class10 = 1;
        end
        value = 1 - class10 * (dot(w10, x) - b10);
        hinge_loss10 = hinge_loss10 + max(0, value);
    end
    hinge_loss10 = 16*hinge_loss10/train_length + lambda*norm(w10)^2;
    loss10(epoch10) = hinge_loss10;
    epoch10 = epoch10 + 1;
    if hinge_loss10 < 2*tolerance
        learning_rate = 1;
    end
end

%Train LSB SVM
disp('epochs:')
disp(epoch10)
disp('training loss')
disp(hinge_loss10)
lambda = 0; %regularizer
tolerance = 0.05;
learning_rate = 1;
while hinge_loss01 >= tolerance
    hinge_loss01 = 0;
    sub_grad_w01 = zeros(16, 1);
    sub_grad_b01 = 0;
    for n=1:train_length/16
        x = training_set(16*(n-1)+1:16*n,2);
        class01 = training_set(16*n,3);
        if (class01 ==3)
            class01 = 1;
        else
            class01 = -1;
        end
        value = 1 - class01 * (dot(w01, x) - b01);
        if value > 0
            sub_grad_w01 = sub_grad_w01 - class01 * x;
        end
    end
    sub_grad_w01 = 16*sub_grad_w01/train_length + 2*lambda*w01;
    w01 = w01 - learning_rate*sub_grad_w01;
    
    for n=1:train_length/16
        x = training_set(16*(n-1)+1:16*n,2);
        class01 = training_set(16*n,3);
        if (class01 ==3)
            class01 = 1;
        else
            class01 = -1;
        end
        value = 1 - class01 * (dot(w01, x) - b01);
        if value > 0
            sub_grad_b01 = sub_grad_b01 + class01;
        end
    end
    sub_grad_b01 = 16*sub_grad_b01/train_length;
    b01 = b01 - learning_rate*sub_grad_b01;
    
    for n=1:train_length/16
        x = training_set(16*(n-1)+1:16*n,2);
        class01 = training_set(16*n,3);
        if (class01 ==3)
            class01 = 1;
        else
            class01 = -1;
        end
        value = 1 - class01 * (dot(w01, x) - b01);
        hinge_loss01 = hinge_loss01 + max(0, value);
    end
    hinge_loss01 = 16*hinge_loss01/train_length + lambda*norm(w01)^2;
    loss01(epoch01) = hinge_loss01;
    epoch01 = epoch01 + 1;
    if hinge_loss01 < 2*tolerance
        learning_rate = 1;
    end
end

disp('finished training. epochs:')
disp(epoch01)
disp('training loss')
disp(hinge_loss01)

%testing
total_loss = 0;
total_loss10 = 0;
prediction10(n) = zeros(1,1);
missed_bits = 0;
realbit=0;
for n=1:test_length/16
    x = test_set(16*(n-1)+1:16*n,2);
    class = 1;
    hinge_loss_1 = max(0, 1 - class * (dot(w10, x) - b10));
    class = -1;
    hinge_loss_0 = max(0, 1 - class * (dot(w10, x) - b10));
    if hinge_loss_1 < hinge_loss_0
        prediction10(n) = 1;
        total_loss10 = total_loss10 + hinge_loss_1;
    else
        prediction10(n) = 0;
        total_loss10 = total_loss10 + hinge_loss_0;
    end
    if or(test_set(16*n,3) == 0,test_set(16*n,3) ==1)
            realbit = 0;
        else
            realbit = 1;
    end
    if not(prediction10(n) == realbit)
        missed_bits = missed_bits + 1;
    end
end
total_loss10 = 16*total_loss10/test_length + lambda*norm(w10)^2;

total_loss01 = 0;
prediction01(n) = zeros(1,1);
for n=1:test_length/16
    x = test_set(16*(n-1)+1:16*n,2);
    class = 1;
    hinge_loss_1 = max(0, 1 - class * (dot(w01, x) - b01));
    class = -1;
    hinge_loss_0 = max(0, 1 - class * (dot(w01, x) - b01));
    if hinge_loss_1 < hinge_loss_0
        prediction01(n) = 1;
        total_loss01 = total_loss01 + hinge_loss_1;
    else
        prediction01(n) = 0;
        total_loss01 = total_loss01 + hinge_loss_0;
    end
    if or(test_set(16*n,3) == 0,test_set(16*n,3) ==2)
            realbit = 0;
        else
            realbit = 1;
    end
    if not(prediction01(n) == realbit)
        missed_bits = missed_bits + 1;
    end
end
total_loss01 = 16*total_loss01/test_length + lambda*norm(w01)^2;
total_loss=total_loss10+total_loss01;

sym_predict(n)= zeros(1,1);
missed_syms=0;
for n=1:test_length/16
    if ((prediction01(n)==0)&(prediction10(n) == 0))
        sym_predict(n)=0;
    elseif ((prediction01(n)==1)&(prediction10(n) == 0))
        sym_predict(n)=1;
    elseif ((prediction01(n)==0)&(prediction10(n) == 1))
        sym_predict(n)=2;
    else
        sym_predict(n)=3;
    end
    if not(sym_predict(n) == test_set(16*n,3))
        missed_syms = missed_syms + 1;
    end
end

disp('test loss:')
disp(total_loss)
disp('missed bits')
disp(missed_bits)
disp('missed symbols')
disp(missed_syms)