%% Econometrics II 2025 Fall

% Q2-3: Now we are talking about selectivity models and its MLE

% when we ignore selection, we estimate beta0, beta1, and sigma_u
% with selection, we also estimate the parameters in the selection binary
% choice model, as well as the corerlation rho

clc;
clear;

global x y

N = 100;                                           % Number of observations.

% True parameters
beta0 = 0.5;                                       % Beta 0.
beta1 = 3;                                         % Beta 1.
gamma0 = 1;                                        % Gamma 0.
gamma1 = 2;                                        % Gamma 1.
sigmau = 2;                                        % Standard deviation for u.
rho    = -0.9;                                     % Correlation between u and v.

sim = 100;                                          % Number of simulations.
results_standard = zeros(sim, 3);                  % Results for standard ML (biased)
results_selection = zeros(sim, 6);                 % Results for selection model ML (less biased but should be consistent)

% Generate covariates (as given in the problem)
x = ((1:N)'/N).*normrnd(0,1,N,1);
w = ((1:N)'/N).*normrnd(0,1,N,1);

for s = 1:sim 
    % Generate data with selection
    e = mvnrnd([0; 0],[sigmau^2 rho*sigmau; rho*sigmau 1],N);
    u = e(:,1);                                                             
    v = e(:,2);                                                             
    
    y0 = beta0 + beta1*x + u;                                              
    z0 = gamma0 + gamma1*w + v;                                            
    
    y = y0.*double((z0 > 0));                                               
    y((z0 <= 0)) = nan;                                                     
    z = double((z0 > 0));                                                   
    
    % Get selected sample
    selected = ~isnan(y);
    y_selected = y(selected);
    x_selected = x(selected);
    
    options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton');
    
    % 1. STANDARD LINEAR MODEL ML (BIASED) - Ignoring selection
    b0_standard = [0,0,1];
    [b_standard, ~] = fminunc(@(b) standard_linear_logl(b, x_selected, y_selected), b0_standard, options);
    results_standard(s,:) = b_standard';
    
    % 2. SELECTION MODEL ML (CORRECT) - Accounting for selection
    % Simple starting values
    b0_selection = [beta0, beta1, sigmau, gamma0, gamma1, rho];  % probably should change this 
    [b_selection, ~] = fminunc(@(b) selection_model_logl(b, x, w, y, z), b0_selection, options);
    results_selection(s,:) = b_selection';
    
end

%% DISPLAY RESULTS - BIAS DEMONSTRATION
fprintf('=== BIAS IN STANDARD LINEAR MODEL ML ESTIMATION (IGNORING SELECTION) ===\n');
fprintf('True parameters: beta0 = %.3f, beta1 = %.3f, sigma_u = %.3f\n', beta0, beta1, sigmau);
fprintf('Standard ML estimates: beta0 = %.3f, beta1 = %.3f, sigma = %.3f\n', ...
    mean(results_standard(:,1)), mean(results_standard(:,2)), mean(results_standard(:,3)));

fprintf('\nBIAS (Standard ML - True):\n');
fprintf('beta0 bias: %.4f (%.1f%%)\n', mean(results_standard(:,1)) - beta0, ...
    100*(mean(results_standard(:,1)) - beta0)/beta0);
fprintf('beta1 bias: %.4f (%.1f%%)\n', mean(results_standard(:,2)) - beta1, ...
    100*(mean(results_standard(:,2)) - beta1)/beta1);
fprintf('sigma bias: %.4f (%.1f%%)\n', mean(results_standard(:,3)) - sigmau, ...
    100*(mean(results_standard(:,3)) - sigmau)/sigmau);

fprintf('\n=== CORRECT SELECTION MODEL ML ESTIMATION (ACCOUNTING FOR SELECTION) ===\n');
fprintf('True parameters: beta0 = %.3f, beta1 = %.3f, gamma0 = %.3f, gamma1 = %.3f, sigma_u = %.3f, rho = %.3f\n', ...
    beta0, beta1, gamma0, gamma1, sigmau, rho);
fprintf('Selection ML estimates: beta0 = %.3f, beta1 = %.3f, gamma0 = %.3f, gamma1 = %.3f, sigma = %.3f, rho = %.3f\n', ...
    mean(results_selection(:,1)), mean(results_selection(:,2)), mean(results_selection(:,4)), ...
    mean(results_selection(:,5)), mean(results_selection(:,3)), mean(results_selection(:,6)));

fprintf('\nBIAS (Selection ML - True):\n');
fprintf('beta0 bias: %.4f (%.1f%%)\n', mean(results_selection(:,1)) - beta0, ...
    100*(mean(results_selection(:,1)) - beta0)/beta0);
fprintf('beta1 bias: %.4f (%.1f%%)\n', mean(results_selection(:,2)) - beta1, ...
    100*(mean(results_selection(:,2)) - beta1)/beta1);
fprintf('gamma0 bias: %.4f (%.1f%%)\n', mean(results_selection(:,4)) - gamma0, ...
    100*(mean(results_selection(:,4)) - gamma0)/gamma0);
fprintf('gamma1 bias: %.4f (%.1f%%)\n', mean(results_selection(:,5)) - gamma1, ...
    100*(mean(results_selection(:,5)) - gamma1)/gamma1);
fprintf('sigma bias: %.4f (%.1f%%)\n', mean(results_selection(:,3)) - sigmau, ...
    100*(mean(results_selection(:,3)) - sigmau)/sigmau);
fprintf('rho bias: %.4f (%.1f%%)\n', mean(results_selection(:,6)) - rho, ...
    100*(mean(results_selection(:,6)) - rho)/rho);

%% LIKELIHOOD FUNCTIONS

% 1. Standard linear model ML (ignores selection - BIASED)
function logL = standard_linear_logl(params, x, y)
    beta0 = params(1);
    beta1 = params(2);
    sigma = max(params(3), 0.001);
    
    n = length(y);
    residuals = y - beta0 - beta1*x;
    
    % Standard normal regression log-likelihood (ignores selection)
    logL = n*log(sigma) + sum(residuals.^2)/(2*sigma^2);
end

% 2. Correct selection model ML (accounts for selection)
function logL = selection_model_logl(params, x, w, y, z)
    beta0 = params(1);
    beta1 = params(2);
    sigma_u = params(3);
    gamma0 = params(4);
    gamma1 = params(5);
    rho = max(min(params(6), 0.999), -0.999);  % Constrain rho to (-1,1)
    
    logL = 0;
    n = length(z);
    
    for i = 1:n
        if z(i) == 0  % Non-selected observations
            index_z = gamma0 + gamma1*w(i);
            prob_z0 = normcdf(-index_z);
            logL = logL + log(prob_z0);
            
        else  % Selected observations (z(i) == 1)
            % Density of y (log)
            u_std = (y(i) - beta0 - beta1*x(i)) / sigma_u;
            density_y = log(1/sigma_u) + log(normpdf(u_std));
            
            % Conditional probability
            condition = (gamma0 + gamma1*w(i) + rho * u_std) / sqrt(1 - rho^2);
            prob_conditional = log(normcdf(condition));
            
            logL = logL + density_y + prob_conditional;
        end
    end
    
    % Return negative log-likelihood for minimization
    logL = -logL;
end