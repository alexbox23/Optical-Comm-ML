clear
clc
%This file is used to run the radial basis function kernel with tuned
%values of perform a tuning sweep using oneshot variable

%saved data for binary kernel tuning at test length 512
% gamma = 3400;
% mu = 1/1000;
% rho = 0;
% tl= 512

%saved data for binary kernel tuning at test length 64
gamma = 1500;
mu = 1/100;
rho = 0;
train = 64;
noise = 100000;

oneshot=1;%run kernal svm once=1 tuning sweep=0
g=linspace(1000,2000,11); %gamma sweep values
m=[1,.316,.1,.0316,.01,.00316,.001,.000316,.0001]; %mu sweep values
r=linspace(0,100,21); %rho sweep values (unswept default=0)
tl=linspace(64,1024,16); % training length sweep values (unswept default=64)
snr=linspace(1,100,40); % snr sweep values (unswept default=100)

if oneshot==1
    [missed_bits,error_rate]=kernelbin_SVM(gamma,mu,rho,train,noise);
else
    for i=1:11 %gamma sweep
        for n=1:9 %mu sweep
            [missed_bits(i,n),error_rate(i,n)]=kernelbin_SVM(g(i),m(n),r(1),tl(1),snr(40));
        end
    end
    contour(log10(m),g,error_rate) %gamma and mu tuning contour plot
    title('RBF SVM Tuning for Training Length 64'), xlabel('Mu'), ylabel('Gamma'),zlabel('Error Rate')%plot labeling
end

