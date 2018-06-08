function[missed_bits,error_rate]=kernelbin_SVM(gamma,mu,ro,train_size,snr) %RBF SVM input, gamma, mu, rho, training set sybols, and SNR, return error rate and missed symbols

fid = fopen('data/data_Binary_NRZ_RX(small).csv'); %read binary nrz receive data file to data
data = textscan(fid, '%f %f', 'Delimiter', ',', 'HeaderLines', 7);
fclose(fid);
data = cell2mat(data);

fid = fopen('data/labels_Binary_NRZ_TX.csv'); %read binary nrz trasmit label to label
labels = textscan(fid, '%f', 'Delimiter', ',');
fclose(fid);
labels = cell2mat(labels);

bit_length = 0.04; %time length of one bit (ns)
T = data(2,1); %sampling interval (ns)
bit_samples = bit_length/T; %number of samples in one bit

train_portion = train_size/2048; %proportion of data used for training
train_length = floor(length(data) * train_portion);
training_set = zeros(train_length, 3);
for n=1:train_length %generate training data sequence
    training_set(n,1) = mod(data(n,1), bit_length); %time wrt clock cycle (ns)
    training_set(n,2) = data(n,2); %electrical signal value
    training_set(n,3) = labels(floor(data(n,1)/bit_length) + 1); %label
end

test_length = length(data) - train_length;
test_set = zeros(test_length, 3);
for n=1:test_length %generate test data sequence
    test_set(n,1) = mod(data(n+train_length,1), bit_length);
    test_set(n,2) = data(n+train_length,2);
    test_set(n,3) = labels(floor(data(n+train_length,1)/bit_length) + 1);
end

%training
noise_en=1; %noise enable
nsp=.0313/snr;
disp('training...')
a=zeros(train_length/16,1); %initialize a error values
b=zeros(train_length/16,1); %initialize b error values
sig=zeros(train_length/16); %initialize sigma sample vector
x=zeros(train_length/16,16); %initialize x data feature vectors
h=zeros(train_length/16,16); %initialize h data feature vectors
yh=zeros(train_length/16,1); %initialize y hat label estimate
y=zeros(train_length/16,1); %initialize y label
sam=1:train_length/16;  %initialize sample vector

for i=1:train_length/16 %train svm to training set
    x(i,1:16) = training_set(16*(i-1)+1:16*i,2)+noise_en*normrnd(0,nsp,16,1); %read training data into x
    y(i)=training_set(16*i,3); %read training label to y
    if y(i)==0 %change 0 label to -1
        y(i)=-1;
    end
end

sig=datasample(sam,train_length/16, 'replace',false); %random sample symbol integer position without replacement

for n=1:train_length/16 % evaluate RBF stochastic gradient descent
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
missed_bits = 0; %initalize missed bits
yp=zeros(test_length/16,1); %initialize y prediction sum
xt=zeros(test_length/16,1); %initialize x test values
predict=zeros(test_length/16,1); %initialize bit prediction
for i=1:test_length/16 %test data classification
    xt(i,1:16) = test_set(16*(i-1)+1:16*i,2)+noise_en*normrnd(0,nsp,16,1); %read test set data to x test
    
    for m=1:train_length/16 %run RBF SVM classification
        yp(i)=yp(i)+mu*y(m)*(a(m)+b(m)*exp(-gamma*dot(xt(i,1:16),x(m,1:16))));
    end
    if yp(i)<0 %use prediction sum to generate prediction bit
        predict(i)=0;
    else
        predict(i)=1;
    end
    if predict(i)==test_set(16*i,3) %compare prediction bit to test data label
    else
        missed_bits=missed_bits+1; %update missed bit count
    end
end
error_rate=missed_bits/length(predict);%normalized error rate
disp('Error Rate:') %display error rate
disp(error_rate)
disp('Missed bits') %display number of missed bits
disp(missed_bits)
end