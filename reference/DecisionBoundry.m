function DecisionBoundry( X,Y,Yn )

figure
hold on
imagesc([min(X(:,1)) max(X(:,1))], [min(X(:,2)) max(X(:,2))], Yn);
scatter(X(Y==1,1),X(Y==1,2),'+g')
scatter(X(Y==-1,1),X(Y==-1,2),'.r')
axis([min(X(:,1)) max(X(:,1)) min(X(:,2)) max(X(:,2))])
xlabel('{x_1}')
ylabel('{x_2}')
legend('Positive Class','Negative Class')
hold off
end

