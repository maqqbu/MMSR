function [gradient] = grad1(x, C, N)
%
% gradient = N .* (C - x * x') * x;
 gradient = 2 * (C - x * x') * x;
end

