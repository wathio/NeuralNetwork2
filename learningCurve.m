function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% Initializing some parameters
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ======================  CODE  ======================

   fprintf("\n ITERATING OVER THE NUMBER OF TRAINING EXAMPLES \n");
for i=1:m
   %X=X(1:i,:);
   %X=[ones(size(X,1),1) X]; %adding  the intercept
   %m1=size(X,2); %
   %y=y(1:i,:);
  
   %theta=zeros(m1,1);
   %[l1,c1]=size(theta);
   %[l2,c2]=size(X);
   %fprintf("the size of theta at iteration %d is %d\t\t%d\n\n",i,l1,c1);
   %fprintf("the size of X at iteration %d is %d\t\t%d\n\n",i,l2,c2);
   %Compute Gradient and cost J
   fprintf("\n Linear Regression at iteration: %d \n\n", i);
   %[J, grad] = linearRegCostFunction(theta, X, y,1);
   %Optimize theta trough trainLinearReg()
   [theta] = trainLinearReg(X(1:i,:), y(1:i,:),lambda);
   %Compute error_train
   fprintf("\n Evaluating error_train and error_val at iteration: %d \n", i);
  
   [J, grad] = linearRegCostFunction(X(1:i,:), y(1:i,:), theta,0);
            error_train(i)=J;
   [J, grad] = linearRegCostFunction(Xval,yval,theta,0);
            error_val(i)=J;
   
end




% -------------------------------------------------------------

% =========================================================================

end
