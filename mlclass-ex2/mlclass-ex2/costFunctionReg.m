function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



hypothesis = sigmoid(X*theta);
J1 = 0;J2=0;
for i=1:m
    hyp = hypothesis(i);
    cost= - (y(i)*log(hyp)) - ((1-y(i))*(log(1 - hyp)));
    J1 = J1 + cost;
end
J1 = J1/m;
for j=2:length(theta)
    J2 = J2 + theta(j)*theta(j);
end
J2=(J2*lambda)/(2*m);
J=J1+J2;
    

for j=1:length(theta)
for i=1:m
    grad(j) = grad(j) + (hypothesis(i) - y(i))*X(i,j);
end
grad(j) = grad(j)/m;
if(j>1) 
    grad(j) = grad(j) + (lambda*theta(j))/m;
end
end

    



% =============================================================

end
