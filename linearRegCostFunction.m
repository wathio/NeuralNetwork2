function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
[a b]=size(theta);
fprintf("\nThe size of theta are: %d \t %d \n",a ,b);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%X=[ones(m,1) X]; NOTE Ones have been added in the script ex5.m

h=X*theta;

%J_0=sum(sum((h-y).^2));
J_0=(h-y)'*(h-y)/(2*m);
Grad_0=(X'*(h-y))/m;

theta(1)=0
J=J_0 + lambda/(2*m)*(sum(theta.^2)) ;

grad=Grad_0 + theta*lambda/m;









% =========================================================================

grad = grad(:);

end
