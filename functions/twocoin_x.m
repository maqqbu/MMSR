function [x] = twocoin_x(Omega, C, Y, i, F)
m = size(C, 1);
H = zeros(2);
% remove the largest and smallest F values
for j =  1 : m
    fval(j) = abs(C(i,j))/norm(Y(j, :));
end

[~, I] = sort(fval);
remove_id = [I(1:F), I(m-F+1: m)];
Omega(i, remove_id) = 0;

for j = 1 : m
    H = H + 2 * Y(j, :)' * Y(j, :) * Omega(i, j);
end

f = -2 * Y' * (C(i, :) .* Omega(i, :))';

lb = -0.98 * ones(2, 1);
ub = 0.98 * ones(2, 1);
options = optimset('Display', 'off');
x = quadprog(H,f,[],[],[],[],lb,ub,[], options);
end

