clear
clc
%This file is used to run the radial basis function kernel for 4PAM data

%these variables were not tuned. RBF was determined to be overfitted and
%tuning took excessive time
gamma = [1000,2500];
mu = [3,.0001];
rho = [0,0];
train= 512;

[missed_sym,error_rate]=kernelpam_SVM(gamma,mu,rho,train);