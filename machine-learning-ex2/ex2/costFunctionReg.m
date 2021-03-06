function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


o=(theta'*X')';
v=sigmoid(o);
w=sum(log(v).*(y.*-1)-(1-y).*log(1-v))/m;
regular=((sum(theta.^2)-(theta(1)^2))*(lambda/(2*m)));
J=w+regular;

grad = zeros(size(theta));
%grad(1) =sum((v-y).*(1/m));
grad=(((v-y)'*X)*(1/m))';
grad=grad+(lambda/m)*theta;
grad(1) =sum((v-y).*(1/m));
%for i=2:size(theta)
%    grad(i) =grad(i)+(lambda/m)*theta(i) ;
%end
%x=grad
%grad = zeros(size(theta));
%grad(1) =sum((v-y).*X(:,1))*(1/m);
%grad(2)=sum((v-y).*X(:,2))*(1/m);
%grad(3)=sum((v-y).*X(:,3))*(1/m);




% =============================================================

end
