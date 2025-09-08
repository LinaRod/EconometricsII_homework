% Compute Outer Product of Scores (gradient) for Probit model
% Inputs:
%   b - parameter vector (k x 1)
%   X - covariate matrix (N x k)
%   z - outcome vector (N x 1)
% Output:
%   OPS - k x k matrix, sum_i S_i * S_i'


function OPS = OP(b,X,z)
    % S is the Nx2 matrix of partial derivatives for each observation
    % score
    Z = (2*z - 1) .* X;
    Lamda = normpdf(Z*b') ./ normcdf(Z*b');
    S = Z .* Lamda;

    OPS = S'*S;         % outer product of score
end
