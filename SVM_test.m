% SVM_test - function to test an SVM hyperplane
%
% arguments: 
%   test_set - array of labeled feature vectors
%                   column 1: time (ns)
%                   column 2: electrical signal (a.u.)
%                   column 3: class (+1 or -1)
%   class_pos - array of labels to be represented by class = +1
%   w - hyperplane weight vector
%   b - hyperplane bias constant
%   reg_pen - soft margin regularization penalty (\lambda)
%
% returns:
%   avg_loss - average loss over test_set
%   misclass - number of misclassified inputs



function [avg_loss, misclass]=SVM_test(test_set, class_pos, w, b, reg_pen)
    bit_samples = 16;       % hardcoded partitioning of data
    test_length = length(test_set);
    
    lambda = reg_pen;       % regularizer
    
    avg_loss = 0;
    misclass = 0;
    for n=1:test_length/bit_samples
        x = test_set(bit_samples*(n-1)+1:bit_samples*n,2);
        class = 1;
        hinge_loss_1 = max(0, 1 - class * (dot(w, x) - b));
        class = -1;
        hinge_loss_0 = max(0, 1 - class * (dot(w, x) - b));
        if hinge_loss_1 < hinge_loss_0
            prediction = 1;
            avg_loss = avg_loss + hinge_loss_1;
        else
            prediction = 0;
            avg_loss = avg_loss + hinge_loss_0;
        end
        label = test_set(bit_samples*n,3);
        if ismember(label, class_pos)
            class = 1;
        else
            class = -1;
        end
        if not(prediction == class)
            misclass = misclass + 1;
        end
    end
    avg_loss = bit_samples*avg_loss/test_length + lambda*norm(w)^2;
end