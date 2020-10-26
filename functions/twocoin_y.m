function [y] = twocoin_y(Omega, C, X, j, F)
m = size(C, 1);
H = zeros(2);
% remove the largest and smallest F values
for i =  1 : m
    fval(i) = abs(C(i,j))/norm(X(i, :));
end

[~, I] = sort(fval);
remove_id = [I(1:F), I(m-F+1: m)];
Omega(remove_id, j) = 0;

for i = 1 : m
    H = H + 2 * X(i, :)' * X(i, :) * Omega(i, j);
end

f = -2 * X' * (C(:, j) .* Omega(:, j));

% lb = -0.48 * ones(2, 1);
% ub = 0.48 * ones(2, 1);
lb = -0.98 * ones(2, 1);
ub = 0.98 * ones(2, 1);
options = optimset('Display', 'off');
y = quadprog(H,f,[],[],[],[],lb,ub, [],options);
end


