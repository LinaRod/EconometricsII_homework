function [Y_demeaned, X_demeaned] = lina_fe(Y, X, tfe, csfe, W)
%MY_FE_WEIGHTED Remove fixed effects with weights
%
% This function removes time fixed effects, entity fixed effects, or both (TWFE)
% from panel data matrices Y and X using a weighting matrix W.
%
% INPUTS:
%   Y, X           - N x T matrices (entities x time)
%   tfe            - 1 to remove time fixed effects
%   csfe           - 1 to remove cross-section fixed effects
%   W              - N x T weighting matrix
%
% OUTPUTS:
%   Y_demeaned, X_demeaned - fixed effects removed after winthin transformation

N = size(Y,1);

Y_demeaned = Y;
X_demeaned = X;

%% 1. Remove time fixed effects (weighted column means)
if tfe
    % Weighted column mean: sum(Y.*W)/sum(W.^2)
    time_mean_Y = sum(Y_demeaned.*W)./sum(W.^2); % 1 x T
    time_mean_X = sum(X_demeaned.*W)./sum(W.^2);

    Y_demeaned = Y_demeaned - ones(N,1) .* time_mean_Y .* W;
    X_demeaned = X_demeaned - ones(N,1) .* time_mean_X .* W;
end

%% 2. Remove entity fixed effects (weighted row means)
if csfe
    % Weighted row mean: sum(Y.*W,2)/sum(W.^2,2)
    individual_mean_Y = sum(Y_demeaned.*W,2) ./ sum(W.^2,2); % N x 1
    individual_mean_X = sum(X_demeaned.*W,2) ./ sum(W.^2,2);

    Y_demeaned = Y_demeaned - individual_mean_Y .* W;
    X_demeaned = X_demeaned - individual_mean_X .* W;
end

%% 3. Two-way correction: add back weighted grand mean
if tfe && csfe
    grand_mean_Y = sum(Y(:).*W(:))/sum(W(:));
    grand_mean_X = sum(X(:).*W(:))/sum(W(:));

    Y_demeaned = Y_demeaned + grand_mean_Y;
    X_demeaned = X_demeaned + grand_mean_X;
end

end
