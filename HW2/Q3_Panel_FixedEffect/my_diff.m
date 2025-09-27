% -------------------------------------------------------------------------
function [diff_data] = my_diff(data,k,T)
if k == 1
    diff_data = data(:,2:end)-data(:,1:end-1);
else
    T_diff = floor((T-1)/k);
    diff_data = zeros(size(data,1),T_diff);
    
    for i = 1:T_diff
        diff_data(:,T_diff-i+1)= data(:,T-(i-1)*(k+1)) - data(:,T-(i-1)*(k+1)-k);
    end
end
end
% -------------------------------------------------------------------------
% Purpose:
%   Computes k-th order differences of panel/time series data.
%   For k > 1, this function generates **non-overlapping differences**.
%
% Inputs:
%   data : n × T matrix, where n = number of cross-sectional units (e.g., states),
%          T = number of time periods
%   k    : order of the difference
%   T    : total number of periods (equal to size(data,2))
%
% Output:
%   diff_data : differenced data matrix
%
% Notes:
%   - If k == 1, the code produces standard first differences:
%         diff_data(:,t) = data(:,t+1) - data(:,t)
%
%   - If k > 1, the code computes **non-overlapping k-period differences**.
%     This means it skips ahead k observations each time, instead of sliding
%     forward one step like standard differences.
%
%     Example: data = [x1, x2, x3, x4, x5, x6]
%       * Standard (overlapping) 2-period differences:
%           [x3-x1, x4-x2, x5-x3, x6-x4]
%         → every difference uses data that overlaps with the next.
%
%       * Non-overlapping 2-period differences (this function):
%           [x3-x1, x6-x4]
%         → each difference is based on disjoint intervals,
%           so no observation is reused.%
%
% Use case:
%   Non-overlapping differences are useful when we want to reduce
%   autocorrelation that comes from overlapping intervals.
