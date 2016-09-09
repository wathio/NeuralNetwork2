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

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
% 1-choose sample size 2-defined new size of theta, 3-compute new optimized theta
%4- set lambda to 0, 5-apply costFunctionReg on training sample to compute   J_train=error_train.
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
