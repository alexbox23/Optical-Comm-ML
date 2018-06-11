%Simple Maximum Likelihood Classifier for 4PAM data transmission over 
%optical channel. A training set is chosen with train symbols variable 
%(#/2048 symbols). The training set is used to build a probabilty estimate 
%of the mean and variance of the data sample feature vector. Test symbols
%are compared against the feature mean vectors for each label using a 
%weighted norm where the weights are inversely proportional to the sample 
%variance. The smallest norm determines the predicted label which is
%compared against the tx label. # of symbol errors and error rate returned.

clear
clc

train_symbols=64;

fid = fopen('data/data_PAM4_RX(small).csv'); %read pam receive data file to data
data = textscan(fid, '%f %f', 'Delimiter', ',', 'HeaderLines', 7);
fclose(fid);
data = cell2mat(data);

fid = fopen('data/labels_PAM4_TX.csv'); %read pam trasmit label to label
labels = textscan(fid, '%f', 'Delimiter', ',');
fclose(fid);
labels = cell2mat(labels);

bit_length = 0.04; %time length of one bit (ns)
T = data(2,1); %sampling interval (ns)
bit_samples = bit_length/T; %number of samples in one bit

train_portion = 64/2048; %proportion of data used for training
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

y0=[];
y1=[];
y2=[];
y3=[];

for i=1:train_length/16 %split training set values into label groups
    x(1:16,1) = training_set(16*(i-1)+1:16*i,2);
    if training_set(16*i,3)==0 %matrix of data feature collumns corresponding to label=0
        y0=[y0,x(:)];
    elseif training_set(16*i,3)==1 %matrix of data feature collumns corresponding to label=1
        y1=[y1,x(:)];
    elseif training_set(16*i,3)==2 %matrix of data feature collumns corresponding to label=2
        y2=[y2,x(:)];
    else %matrix of data feature collumns corresponding to label=3
        y3=[y3,x(:)];
    end
end

%find the mean and variance of data feature vectors for both binary labels
%normalize mean and variance to zero if there was no label found in the training set
%create weights using inverse of variance and avoid division by zero
%normalize weights to sum to 1

mu0=mean(y0,2);
if isempty(mu0)
    mu0=zeros(16,1);
end
mu1=mean(y1,2);
if isempty(mu1)
    mu1=zeros(16,1);
end
mu2=mean(y2,2);
if isempty(mu2)
    mu2=zeros(16,1);
end
mu3=mean(y3,2);
if isempty(mu3)
    mu3=zeros(16,1);
end
var0=var(y0,0,2);
if isempty(var0)
    var0=zeros(16,1);
end
var1=var(y1,0,2);
if isempty(var1)
    var1=zeros(16,1);
end
var2=var(y2,0,2);
if isempty(var2)
    var2=zeros(16,1);
end
var3=var(y3,0,2);
if isempty(var3)
    var3=zeros(16,1);
end
if var0==zeros(16,1)
    wu0=ones(16,1);
else
    wu0=1./var0;
end
if var1==zeros(16,1)
    wu1=ones(16,1);
else
    wu1=1./var1;
end
if var2==zeros(16,1)
    wu2=ones(16,1);
else
    wu2=1./var2;
end
if var3==zeros(16,1)
    wu3=ones(16,1);
else
    wu3=1./var3;
end
w0=wu0./sum(wu0);
w1=wu1./sum(wu1);
w2=wu2./sum(wu2);
w3=wu3./sum(wu3);

%initialize missed bits and prediction vector
missed_syms=0;
predict=0;

for i=1:test_length/16 %test data classification
        xt(i,1:16) = test_set(16*(i-1)+1:16*i,2); %read test set data to x test
        norm0=0;
        norm1=0;
        norm2=0;
        norm3=0;
        for n=1:16 %find weighted distance for naive ML estimation, distances with lower variance are penalized heavier
            norm0=norm0+w0(n)*(xt(i,n)-mu0(n))^2;
            norm1=norm1+w1(n)*(xt(i,n)-mu1(n))^2;
            norm2=norm2+w2(n)*(xt(i,n)-mu2(n))^2;
            norm3=norm3+w3(n)*(xt(i,n)-mu3(n))^2;
        end
        if norm0<=norm1&&norm0<=norm2&&norm0<=norm3 %use norm distance to predict bit. Closer distance is correlated to higher probability
            predict(i)=0;
        elseif (norm1<norm0)&&(norm1<=norm2)&&(norm1<=norm3)
            predict(i)=1;
        elseif (norm2<norm0)&&(norm2<norm1)&&(norm2<=norm3)
            predict(i)=2;
        elseif (norm3<norm0)&&(norm3<norm1)&&(norm3<norm2)
            predict(i)=3;
        else
            predict(i)=-1;
        end
        if predict(i)==test_set(16*i,3) %compare prediction bit to test data label
        else
        missed_syms=missed_syms+1; %update missed bit count
        end
end

error_rate=missed_syms/length(predict);%normalized error rate
disp('Error Rate:') %display error rate
disp(error_rate)
disp('Missed symbols') %display number of missed bits
disp(missed_syms)

