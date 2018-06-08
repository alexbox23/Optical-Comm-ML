% SVM_train - function to train an SVM hyperplane
%
% arguments: 
%   training_set - array of labeled feature vectors
%                   column 1: time (ns)
%                   column 2: electrical signal (a.u.)
%                   column 3: labels (0 or 1 for Binary, 0-3 for PAM4)
%   class_pos - array of labels to be represented by class = +1
%   learning_rate - step size for gradient descent (\mu)
%   tolerance - upper bound condition for training loss
%   reg_pen - soft margin regularization penalty (\lambda)
%   slowdown - boolean to toggle slowdown option
%
%
% returns:
%   epoch - number of epochs needed to train within tolerance
%   loss - array of every epoch's training loss
%   w - optimized hyperplane weight vector
%   b - optimized hyperplane bias constant

function [epoch, loss, w, b]=SVM_train(training_set, class_pos, learning_rate, tolerance, reg_pen, slowdown)
    slow_tol = 2*tolerance; % slowdown option: tolerance for triggering slow_rate
    slow_rate = 1;          % slowdown option: smaller value for learning_rate
    
    bit_samples = 16;       % hardcoded partitioning of data
    train_length = length(training_set);
    
    w = zeros(bit_samples, 1);
    b = 0;
    lambda = reg_pen;       % regularizer
    epoch = 1;
    loss = zeros(1,1);
    hinge_loss = 1;

    while hinge_loss >= tolerance
        %gradient descent
        sub_grad_w = zeros(bit_samples, 1);
        sub_grad_b = 0;
        for n=1:train_length/bit_samples
            x = training_set(bit_samples*(n-1)+1:bit_samples*n,2);
            label = training_set(bit_samples*n,3);
            if ismember(label, class_pos)
                class = 1;
            else
                class = -1;
            end
            value = 1 - class * (dot(w, x) - b);
            if value > 0
                sub_grad_w = sub_grad_w - class * x;
                sub_grad_b = sub_grad_b + class;
            end
        end
        sub_grad_w = bit_samples*sub_grad_w/train_length + 2*lambda*w;
        w = w - learning_rate*sub_grad_w;
        sub_grad_b = bit_samples*sub_grad_b/train_length;
        b = b - learning_rate*sub_grad_b;
        
        %calculate hinge loss
        hinge_loss = 0;
        for n=1:train_length/bit_samples
            x = training_set(bit_samples*(n-1)+1:bit_samples*n,2);
            label = training_set(bit_samples*n,3);
            if ismember(label, class_pos)
                class = 1;
            else
                class = -1;
            end
            value = 1 - class * (dot(w, x) - b);
            hinge_loss = hinge_loss + max(0, value);
        end
        hinge_loss = bit_samples*hinge_loss/train_length + lambda*norm(w)^2;
        loss(epoch) = hinge_loss;
        epoch = epoch + 1;
        if slowdown && hinge_loss < slow_tol
            learning_rate = slow_rate;
        end
    end
end