function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. 
%

% Selected values of lambda.
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% Initialization
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
error_test=0;
% ======================CODE======================

for i=1:numel(lambda_vec)
    lambda=lambda_vec(i);
    [theta]=trainLinearReg(X,y,lambda);
    [J,grad]=linearRegCostFunction(X,y,theta,0);
            error_train(i)=J;
    [J,grad]=linearRegCostFunction(Xval,yval,theta,0);
            error_val(i)=J; 
            
end  









% =========================================================================

end
