    
   % THIS PROGRAM COMPUTE THE ERROR ON THE SET OF DATA RESERVED FOR
   % TESTING THE ACCURACY OF THE HYPOTHESIS/MODEL
   % PLEASE MAKE AVAILABLE THE VARIABLE IN ARGUMENTS BEFORE RUNING THE PROGRAM
   
    function [error_test,theta]=test_validation(X,y,Xtest,ytest,lambda,p)
    
    % adding features to testing sample X
    X_poly=polyFeatures(X,p);
    [X_poly, mu, sigma] = featureNormalize(X_poly); 
    X=[ones(size(X_poly,1),1) X_poly];
    
    % adding features to training sample X
    X_poly_test = polyFeatures(Xtest, p);
    X_poly_test = bsxfun(@minus, X_poly_test, mu);
    X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
    X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];  % Add Ones

    
    % initializing theta for optimization
    theta=zeros(size(X,2),1);
    [l,c]=size(theta);
    [l1,c1]=size(X_poly_test);
    fprintf("\nthe size of theta are: %d\t\t%d\n",l,c);
    fprintf("\nthe size of X_test are: %d\t\t%d\n",l1,c1);
   
    fprintf("\nOPTIMIZING FOR THETA\n");
    [theta]=trainLinearReg(X,y,lambda);
    fprintf("\nComputing the error_test\n");
    [J,grad]=linearRegCostFunction(X_poly_test,ytest,theta,0);
    error_test=J;
