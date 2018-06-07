function[missed_bits,error_rate]=kernelbin_SVM(gamma,mu,ro,train_size,snr)

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

train_portion = train_size/2048; %proportion of data used for training
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
nsp=.0313/snr;
disp('training...')
a=zeros(train_length/16,1);
b=zeros(train_length/16,1);
sig=zeros(train_length/16);
x=zeros(train_length/16,16);
h=zeros(train_length/16,16);
yh=zeros(train_length/16,1);
y=zeros(train_length/16,1);
sam=1:train_length/16;

for i=1:train_length/16
    x(i,1:16) = training_set(16*(i-1)+1:16*i,2)+0*normrnd(0,nsp,16,1);
    y(i)=training_set(16*i,3);
    if y(i)==0
        y(i)=-1;
    end
end

sig=datasample(sam,train_length/16, 'replace',false);

for n=1:train_length/16
    yhs=0;
    for m=1:train_length/16
        h(sig(n),1:16)=x(sig(n),:);
        yhs=yhs+mu*y(m)*(a(m)+b(m)*exp(-gamma*dot(h((sig(n)),:),x(m,1:16))));
    end
    yh(sig(n))=yhs;
    b(sig(n))=(1-2*mu*ro)*b(sig(n));
    if y(sig(n))*yh(sig(n))<=1
        a(sig(n))=a(sig(n))+1;
        b(sig(n))=b(sig(n))+1;
    end
end

%testing
error_rate = 0;
missed_bits = 0;
yp=zeros(test_length/16,1);
xt=zeros(test_length/16,1);
predict=zeros(test_length/16,1);
for i=1:test_length/16
    xt(i,1:16) = test_set(16*(i-1)+1:16*i,2)+normrnd(0,nsp,16,1);
    
    for m=1:train_length/16
        yp(i)=yp(i)+mu*y(m)*(a(m)+b(m)*exp(-gamma*dot(xt(i,1:16),x(m,1:16))));
    end
    if yp(i)<0
        predict(i)=0;
    else
        predict(i)=1;
    end
    if predict(i)==test_set(16*i,3)
    else
        missed_bits=missed_bits+1;
    end
end
error_rate=missed_bits/length(predict);
disp('Error Rate:')
disp(error_rate)
disp('Missed bits')
disp(missed_bits)
end