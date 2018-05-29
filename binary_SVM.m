% SVM for Binary NRZ RX data

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

train_portion = 0.5; %proportion of data used for training
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
