function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
o=(theta'*X')';
v=sigmoid(o);
w=log(v).*(y.*-1)-(1-y).*log(1-v);
J=sum(w)/m;


grad = zeros(size(theta));
grad(1) =sum((v-y).*X(:,1))*(1/m);
grad(2)=sum((v-y).*X(:,2))*(1/m);
grad(3)=sum((v-y).*X(:,3))*(1/m);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
