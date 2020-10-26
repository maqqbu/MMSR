function [y] = remove(x, F, a, num_tasks)
% remove the larget F and smallest F valye of x
% Attention: x is supposed to be a column vector
y = sort(x);
if length(find(y < a)) < F
    y(find(y < a)) = [];
else
    y(1:F) = [];
end

m = size(y,1);
if length(find(y > a)) < F
    y(find(y > a)) = [];
else
    y(m-F+1:m) = [];
end
if y == 0
   y = 1 /sqrt(num_tasks);
end
end
