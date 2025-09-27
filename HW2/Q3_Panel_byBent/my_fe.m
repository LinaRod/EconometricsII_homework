function[Y,X] = my_fe(Y,X,tfe,cfe,W)

% This code removes the fixed effects (time or cross section; tfe and cfe, respectively) with wighting matrix, W, where W'W is the covariance matrix.

N = size(Y,1);
T = size(Y,2);

if tfe == 1
    % Time fixed effects.
    
    fe_Y = (sum(Y.*W))./(sum(W.^2));
    Y = Y - (W.*(ones(N,1)*fe_Y));
    
    fe_X = (sum(X.*W))./(sum(W.^2));
    X = X - (W.*(ones(N,1)*fe_X));
end

if cfe == 1 
    % Cross section fixed effects.
    
    fe_Y = (sum((Y.*W),2))./(sum((W).^2,2));
    Y = Y - (W.*(fe_Y*ones(1,T)));   
   
    fe_X = (sum((X.*W),2))./(sum((W).^2,2));
    X = X - (W.*(fe_X*ones(1,T)));
end

end

