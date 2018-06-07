% gamma = 3400;
% mu = 1/1000;
% rho = 0;
% tl= 512

% gamma = 1500;
% mu = 1/100;
% rho = 0;
% tl= 64
clear
clc

g=linspace(1000,2000,11);
m=[1,.316,.1,.0316,.01,.00316,.001,.000316,.0001];
r=linspace(0,100,21);
tl=linspace(128,1024,8);
snr=linspace(1,100,40);
for i=1:11
    for n=1:9
        [missed_bits(i,n),error_rate(i,n)]=kernelbin_SVM(g(i),m(n),0,64,10000000000);
    end
end

contour(log10(m),g,error_rate)
title('RBF SVM Tuning for Training Length 64'), xlabel('Mu'), ylabel('Gamma'),zlabel('Error Rate')
