function [bfgls,se] = my_gls(Y,X)

% Panel estimation.

N = size(Y,1);                                                              % Number of states.
T = size(Y,2);                                                              % Number of years.
nobs = N*T;                                                                 % Sample size.

numx = size(X,2)/T;                                                         % Number of regressors.

% Step 1: OLS.

yvec = reshape(Y,nobs,1);
xvec = reshape(X,nobs,[]);

bols = (xvec'*xvec)\xvec'*yvec;
umat = reshape(yvec - xvec*bols,N,T);

% Estimate of variance matrix (with ridge).

ridge = 0.1;

varn = mean(umat.^2,2);
gmat = (1./T).*((umat*umat')./((varn.^(1/2))*(varn.^(1/2))'));                                       
gmat = ridge*eye(N) + (1-ridge).*gmat;
gmat = inv(gmat);

% Step 2: FGLS.

weights = varn.^(-1/2);
weights = kron(weights,ones(1,T));
weights = reshape(weights,[],1);

yhet = ((weights.*yvec)'*kron(eye(T),gmat))';                               % Normalize so residuals have variance 1 and weight by variance matrix.
xhet = ((repmat(weights,1,size(xvec,2)).*xvec)'*kron(eye(T),gmat))';

bfgls = (xhet'*xhet)\xhet'*yhet;

resid_het = yhet - xhet*bfgls;
s2 = (resid_het'*resid_het)/(nobs-numx);

vcmat = inv(xhet'*xhet)./s2;

se = (diag(vcmat)).^(1/2);

end

