function [alpha] = grad_ascent( X,Y,C )

N=size(X,1);
alpha=zeros(N,1);
alp2=zeros(N,1);

% Tolerence (for convergence)
norm1=Inf;
tol=10e-5;

% Linear Kernel
Ker=X*X'; 

total_iterations=0;

while norm1>tol
    
    total_iterations=total_iterations+1;
    alp_old=alpha;
    w1=(alp_old.*Y).*X;
    
    for i=1:N
        % Adatron
        eta=1/Ker(i,i);
        
        for j=1:N
            alp2(j,1)=alpha(j,1)*Y(j,1)*Ker(i,j);
        end
        
        alpha(i,1)= alpha(i,1)+eta*(1-(Y(i,1)*sum(alp2)));
    
        if alpha(i,1)<0
            alpha(i,1)=0;
        elseif alpha(i,1)>C
            alpha(i,1)=C; 
        end
        
    end
    
    w2=(alpha.*Y).*X;
    norm1=norm(w2-w1);
end

total_iterations
end

