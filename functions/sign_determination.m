function [s_sign] = sign_determination(C, s_abs)
S = sign(C);
n = length(s_abs);
S_upper = S .* triu(ones(n));
[new_node_end_I new_node_end_J] = ind2sub(size(S), find(S_upper == 1));
S(find(S == 1)) = 0; % delete the original edge with same color pattern
I = eye(n);
S_new = I(new_node_end_I, :) + I(new_node_end_J, :); % add new edge
m = size(S_new, 1);
A = [abs(S) S_new'; S_new zeros(m)];
n_new = size(A, 1);
W = 1./sum(A, 2) * ones(1, n_new) .* A; % construct the weight matrix
[V, D] = eigs(W + eye(n_new), 1, 'sm');
s_sign = sign(V(1 : n)) .* s_abs;
end

