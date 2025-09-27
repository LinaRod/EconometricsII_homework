function [bfgls,se] = lina_gls(Y,X)
%MY_GLS Panel FGLS Estimation with Ridge Regularization
%
% This function estimates coefficients and standard errors for panel data
% using Feasible Generalized Least Squares (FGLS) that accounts for:
% - Heteroskedasticity across cross-sectional units (states)
% - Potential correlation of residuals across units
%
% INPUTS:
%   Y - N x T matrix of dependent variable (states x time)
%   X - N x T*number_of_regressors matrix of independent variables
%
% OUTPUTS:
%   bfgls - estimated coefficients (FGLS)
%   se     - standard errors of the coefficients

%% 1. Initialize dimensions
[N, T] = size(Y);                % N = number of states, T = number of time periods
nobs = N * T;                     % total number of observations
numx = size(X,2) / T;             % number of regressors per time period

%% 2. Step 1: Ordinary Least Squares (OLS)
% Flatten the panel data to long vector format
yvec = reshape(Y, nobs, 1);      % NT x 1
xvec = reshape(X, nobs, []);     % NT x K (number of regressors total)

% OLS estimator
bols = (xvec' * xvec) \ (xvec' * yvec); % K x 1
umat = reshape(yvec - xvec*bols, N, T); % Residuals matrix (N x T)

%% 3. Step 2: Estimate residual variance per unit (varn)
% Variance per state (row-wise mean squared residuals)
varn = mean(umat.^2, 2);           % N x 1, u_i^2 averaged over T periods

%% 4. Step 3: Estimate residual correlation matrix (gmat)
% Correlation matrix of residuals across states
gmat = (1./T) .* ((umat*umat') ./ (sqrt(varn)*sqrt(varn)')); 
% Add ridge to avoid singularity and improve numerical stability
ridge = 0.1;
gmat = ridge*eye(N) + (1-ridge).*gmat; % convex combination of correlation and identity
gmat = inv(gmat);                       % inverse for GLS weighting

%% 5. Step 4: Construct GLS weights
weights = varn.^(-1/2);                 % weights to normalize heteroskedastic residuals
weights = kron(weights, ones(1,T));     % repeat weights across time
weights = reshape(weights, [], 1);      % NT x 1

%% 6. Step 5: Weighted (FGLS) dependent and independent variables
% Apply GLS weights and correlation structure
yhet = ((weights .* yvec)' * kron(eye(T), gmat))';   % NT x 1
xhet = ((repmat(weights, 1, size(xvec,2)) .* xvec)' * kron(eye(T), gmat))'; % NT x K

%% 7. Step 6: FGLS estimation
bfgls = (xhet' * xhet) \ (xhet' * yhet);  % FGLS coefficients (K x 1)

%% 8. Step 7: Standard errors
resid_het = yhet - xhet*bfgls;            % FGLS residuals
s2 = (resid_het' * resid_het) / (nobs - numx); % estimate residual variance
vcmat = inv(xhet'*xhet) / s2;             % covariance matrix of coefficients
se = sqrt(diag(vcmat));                   % standard errors

end
