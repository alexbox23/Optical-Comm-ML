% Automatically generate labels from TX data

% select option for data
encoding = 'NRZ'; %binary
% encoding = 'PAM4'; %base 4

if strcmp(encoding, 'NRZ')
    fid = fopen('data/data_Binary_NRZ_TX(small).csv');
elseif strcmp(encoding, 'PAM4')
    fid = fopen('data/data_PAM4_TX(small).csv');
else
    disp('Encoding option not valid.')
    return
end

data = textscan(fid, '%f %f', 'Delimiter', ',', 'HeaderLines', 7);
fclose(fid);
data = cell2mat(data);

% partition data according to bit rate
bit_length = 0.04; %time length of one bit (ns)
T = data(2,1); %sampling interval (ns)
bit_samples = bit_length/T; %number of samples in one bit
signal_length = data(end,1); %length of entire signal (ns)

labels = zeros(round(signal_length/bit_length),1);

for n=1:length(labels)
    bit_start = (n-1)*bit_samples + 1;
    bit_end = n*bit_samples;
    bit_data = data(bit_start:bit_end,2);
    
    % for each clock cycle, perform thresholding to assign label
    if strcmp(encoding, 'NRZ')
        votes = [sum(bit_data < 0.5) sum(bit_data >= 0.5)];
        if votes(1) > votes(2)
            labels(n) = 0;
        else
            labels(n) = 1;
        end
    elseif strcmp(encoding, 'PAM4')
        votes = [0 0 0 0];
        votes(1) = sum(bit_data < 250);
        votes(2) = sum(bit_data >= 250 & bit_data < 500);
        votes(3) = sum(bit_data >= 500 & bit_data < 750);
        votes(4) = sum(bit_data >= 750 & bit_data < 1000);
        [M, I] = max(votes);
        labels(n) = I - 1;
    end
end

% export csv
if strcmp(encoding, 'NRZ')
    csvwrite('data/labels_Binary_NRZ_TX.csv', labels);
elseif strcmp(encoding, 'PAM4')
    csvwrite('data/labels_PAM4_TX.csv', labels);
end
disp('Labeled data has been exported.')