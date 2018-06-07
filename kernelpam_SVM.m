function[missed_sym,error_rate]=kernelbin_SVM(gamma,mu,ro,train_size)

fid = fopen('data/data_PAM4_RX(small).csv');
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
disp('training...')

sym_predict=zeros(test_length/16,1);
predict=zeros(test_length/16,2);
for s=1:2
a=zeros(train_length/16,1);
b=zeros(train_length/16,1);
x=zeros(train_length/16,16);
h=zeros(train_length/16,16);
yh=zeros(train_length/16,1);
y=zeros(train_length/16,1);
sam=1:train_length/16;

    for i=1:train_length/16
        x(i,1:16) = training_set(16*(i-1)+1:16*i,2);
        y(i)=training_set(16*i,3);
        if s==1
            if or(y(i)==0,y(i)==1)
                y(i)=-1;
            else
                y(i)=1;
            end
        else
            if or(y(i)==0,y(i)==3)
                y(i)=-1;
            else
                y(i)=1;
            end
        end
    end

    sig=datasample(sam,train_length/16, 'replace',false);

    for n=1:train_length/16
        yhs=0;
        for m=1:train_length/16
            h(sig(n),1:16)=x(sig(n),:);
            yhs=yhs+mu(s)*y(m)*(a(m)+b(m)*exp(-gamma(s)*dot(h((sig(n)),:),x(m,1:16))));
        end
        yh(sig(n))=yhs;
        b(sig(n))=(1-2*mu(s)*ro(s))*b(sig(n));
        if y(sig(n))*yh(sig(n))<=1
            a(sig(n))=a(sig(n))+1;
            b(sig(n))=b(sig(n))+1;
        end
    end
    
    
    %testing
    yp=zeros(test_length/16,1);
    xt=zeros(test_length/16,1);
    for i=1:test_length/16
        xt(i,1:16) = test_set(16*(i-1)+1:16*i,2);

        for m=1:train_length/16
            yp(i)=yp(i)+mu(s)*y(m)*(a(m)+b(m)*exp(-gamma(s)*dot(xt(i,1:16),x(m,1:16))));
        end
        if yp(i)<0
            predict(i,s)=0;
        else
            predict(i,s)=1;
        end
    end
end

missed_sym=0;

for i=1:test_length/16
        if  predict(i,1)==1&&predict(i,2)==1
            sym_predict(i)=2;
        elseif predict(i,1)==1&&predict(i,2)==0
            sym_predict(i)=3;
        elseif predict(i,1)==0&&predict(i,2)==1
            sym_predict(i)=1;
        else
            sym_predict(i)=0;
        end
        if sym_predict(i)~=test_set(16*i,3)
            missed_sym=missed_sym+1;
        end
end

error_rate=missed_sym/length(predict);
disp('Error Rate:')
disp(error_rate)
disp('Missed symbols')
disp(missed_sym)
end