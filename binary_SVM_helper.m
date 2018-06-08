function [epoch, total_loss, missed_bits,loss,w,time]=binary_SVM_helper(train_length,reg_pen,learning_rate,tolerance,N,cutoff)
% Calls binary_SVM.m N times and returns averages
% args:
  % train_length: Number of symbols used in the training set, remainder to
    % be used in test set
  % reg_pen: l2 regularization penalty, lambda
  % learning rate: unnormalized learning rate parameter, mu
  % tolerance: hinge loss tolerance requirement to leave trainig mode
  % N: number of trials
  % cutoff: multiplier of tolerance for when to reduce learning rate
% outputs:
  % epoch: average total number of training epochs
  % total_loss: average total hinge loss of testing set
  % missed_bits: average incorrectly classified bits in the testing set
  % loss: average vector of training hinge loss across epochs
  % w: average optimized hyperplane vector
  % time: average time per trial

% Initialize storage vectors
epochs=zeros(N,1);
totalLoss=zeros(N,1);
missedSyms=zeros(N,1);
ws=zeros(N,16);
times=zeros(N,1);

% Run binary SVM N times
for i=1:N
    tic
    [epochs(i), totalLoss(i), missedSyms(i),losses{i},ws(i,:)]=binary_SVM(train_length,reg_pen,learning_rate,tolerance,cutoff);
    times(i)=toc;
end

% Average and return results
time=mean(times);
epoch=mean(epochs);
total_loss=mean(totalLoss);
missed_bits=mean(missedSyms);
w=zeros(16,1);
for i=1:16
    w(i)=mean(ws(:,i));
end

% As loss vectors are different lengths, pad short vectors with zeros
l=zeros(1,N);
for i=1:N
   l(i)=length(losses{i});
end
maxLength=max(l);
Ls=zeros(N,maxLength);
loss=zeros(1,maxLength);
for i=1:N
    for k=1:maxLength
        if k<=l(i)
            Ls(i,k)=losses{i}(k);
        end
    end
end
for i=1:maxLength
    loss(i)=mean(Ls(:,i));
end


end

