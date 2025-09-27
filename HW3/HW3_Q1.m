%% Econometrics II 2025 Fall

% Q1: This question requires us to simulate a panel AR(1) model and
% estimate the paramter a under different sample size N and time periods T

% I think the main takeaway from this question is to see that the fixed
% effect estimator is biased in a dynamic panel model, this bias is smaller
% as the time period T gets larger


clear all; 
close all; 
clc;

%% parameters
rng(123); 
a_values = [0.5, 0.9, 0.99, 0.999];
T_full = 50; N_full = 100;
M = 1000;                   % number of simulations

% different options of T and N for subsets
T_subsets = [5, 15, 50];
N_subsets = [10, 100];

results_individual = zeros(length(a_values), length(T_subsets), length(N_subsets), M);
results_panel = zeros(length(a_values), length(T_subsets), length(N_subsets), M);

% the bias is saved for average across all 1000 simulations
bias_individual = zeros(length(a_values), length(T_subsets), length(N_subsets));
bias_panel = zeros(length(a_values), length(T_subsets), length(N_subsets));
%% Loop over different value of a

for a_idx = 1:length(a_values)
    a_true = a_values(a_idx);
    % for each simulation
    for m = 1:M

        mu_i = randn(N_full, 1);    % the constant term

        % generate the full set
        y = zeros(N_full, T_full);
        e = randn(N_full, T_full);


        y(:,1) = mu_i + randn(N_full, 1);
        for t = 2:T_full
            y(:,t) = mu_i + a_true * y(:,t-1) + e(:,t);
        end

        % Get subset for further analysis
        for T_idx = 1:length(T_subsets)
            T = T_subsets(T_idx);
            for N_idx = 1:length(N_subsets)
                N = N_subsets(N_idx);

                y_sub = y(1:N, 1:T);

                %% i) for i=1 (time series)
                y_i1 = y_sub(1, :)';         % column vector, length T
                Y_ts = y_i1(2:end);          % dependent variable y_2,...,y_T
                X_ts = [ones(T-1,1), y_i1(1:end-1)];  % constant + lag y_1,...,y_{T-1}

                beta_ts = X_ts \ Y_ts;
                a_hat_ind = beta_ts(2);      % AR(1) coefficient                

                %% ii) for panel
                Y_panel = []; 
                X_panel = [];

                for i = 1:N
                    y_i = y_sub(i, :)';           % column vector, length T
                    Y_i = y_i(2:end);             % y_2,...,y_T
                    X_i = y_i(1:end-1);           % lag y_1,...,y_{T-1}

                    % within transformation
                    y_i_demean = Y_i - mean(Y_i);
                    x_i_demean = X_i - mean(X_i);

                    Y_panel = [Y_panel; y_i_demean];
                    X_panel = [X_panel; x_i_demean];
                end

                % No constant since its demeaned
                a_hat_panel = X_panel \ Y_panel;

                results_individual(a_idx, T_idx, N_idx, m) = a_hat_ind;
                results_panel(a_idx, T_idx, N_idx, m) = a_hat_panel;
            end
        end
    end
end

%% Document the bias for all different value of N and T
for a_idx = 1:length(a_values)
    for T_idx = 1:length(T_subsets)
        for N_idx = 1:length(N_subsets)
            bias_individual(a_idx, T_idx, N_idx) = ...
                mean(results_individual(a_idx, T_idx, N_idx, :)) - a_values(a_idx);
            bias_panel(a_idx, T_idx, N_idx) = ...
                mean(results_panel(a_idx, T_idx, N_idx, :)) - a_values(a_idx);
        end
    end
end





