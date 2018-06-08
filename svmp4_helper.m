function [epoch10,epoch01, total_loss, missed_bits,missed_syms,loss01,loss10,time]=svmp4_helper(train_syms,reg_pen,learning_rate,tolerance,N,cutoff)
% Calls svmp4.m N times and returns averages
% args:
  % train_syms: Number of symbols used in the training set, remainder to
    % be used in test set
  % reg_pen: l2 regularization penalty, lambda
  % learning rate: unnormalized learning rate parameter, mu
  % tolerance: hinge loss tolerance requirement to leave trainig mode
  % N: number of trials
  % cutoff: multiplier of tolerance for when to reduce learning rate
% outputs:
  % epoch10: average total number of training epochs (MSB)
  % epoch01: average total number of training epochs (LSB)
  % total_loss: average total hinge loss of testing set (sum LSB+MSB)
  % missed_bits: average incorrectly classified bits in the testing set
  % missed_syms: average incorrectly classified symbols in the testing set
  % loss01: average vector of training hinge loss across epochs (LSB)
  % loss10: average vector of training hinge loss across epochs (MSB)
  % time: average time per trial

% Initialize storage vectors
epochs10=zeros(N,1);
epochs01=zeros(N,1);
totalLoss=zeros(N,1);
missedBits=zeros(N,1);
missedSyms=zeros(N,1);
ws=zeros(N,16);
times=zeros(N,1);

% Run 4-PAM SVM N times
for i=1:N
    tic
    [epochs10(i), epochs01(i), totalLoss(i), missedBits(i),missedSyms(i),losses01{i},losses10{i}]=svmp4(train_syms,reg_pen,learning_rate,tolerance,cutoff);
    times(i)=toc;
end

% Average and return results
time=mean(times);
epoch10=mean(epochs10);
epoch01=mean(epochs01);
total_loss=mean(totalLoss);
missed_bits=mean(missedBits);
missed_syms=mean(missedSyms);
w=zeros(16,1);
for i=1:16
    w(i)=mean(ws(:,i));
end

% As loss vectors are different lengths, pad short vectors with zeros
l=zeros(1,N);
for i=1:N
   l(i)=length(losses01{i});
end
maxLength=max(l);
Ls=zeros(N,maxLength);
loss01=zeros(1,maxLength);
for i=1:N
    for k=1:maxLength
        if k<=l(i)
            Ls(i,k)=losses01{i}(k);
        end
    end
end
for i=1:maxLength
    loss01(i)=mean(Ls(:,i));
end

l=zeros(1,N);
for i=1:N
   l(i)=length(losses10{i});
end
maxLength=max(l);
Ls=zeros(N,maxLength);
loss10=zeros(1,maxLength);
for i=1:N
    for k=1:maxLength
        if k<=l(i)
            Ls(i,k)=losses10{i}(k);
        end
    end
end
for i=1:maxLength
    loss10(i)=mean(Ls(:,i));
end

end
