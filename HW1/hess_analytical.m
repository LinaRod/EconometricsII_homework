function Hess = hess_analytical(b,X,z)
% Calculate the analytical Hessian of the Probit log-likelihood
% Inputs:
%   b - parameter vector (k x 1)
%   X - covariate matrix (N x k)
%   z - binary outcome vector (N x 1)
% Output:
%   Hess - analytical Hessian matrix (k x k)

[N,k] = size(X);
Hess = zeros(k,k);

Xb = X * b';                  % N x 1
Zb = (2*z - 1) .* Xb;        % simplifies z*Xb + (1-z)*(-Xb)

lambda = normpdf(Zb) ./ normcdf(Zb);   % N x 1

% Hessian contribution for each observation
for i = 1:N
    Hess = Hess + (lambda(i) * (lambda(i) + Zb(i))) * (X(i,:)' * X(i,:));
end
