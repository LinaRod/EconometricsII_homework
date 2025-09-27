function [diff_data] = my_diff(data,k,T)

% This code takes non-overlapping kth differences. The data set has T periods (columns).

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

