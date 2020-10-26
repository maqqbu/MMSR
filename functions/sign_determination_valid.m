function [s_sign] = sign_determination(C, s_abs)
S = sign(C);
n = length(s_abs);
% delete invalid nodes
invalid_idx = find(sum(C, 2) == 0);
valid_idx = setdiff([1 : n], invalid_idx);
S_valid = S;
S_valid(invalid_idx, :) = [];
S_valid(:, invalid_idx) = [];
k = size(S_valid, 1); % the num of valid nodes
upper_idx = triu(ones(k));
S_valid_upper = S_valid .* upper_idx;
[new_node_end_I new_node_end_J] = ind2sub(size(S_valid), find(S_valid_upper == 1));
S_valid(find(S_valid == 1)) = 0; % delete the original edge with same color pattern
I = eye(k);
S_valid_new = I(new_node_end_I, :) + I(new_node_end_J, :); % add new edge
m = size(S_valid_new, 1);
A = [abs(S_valid) S_valid_new'; S_valid_new zeros(m)];
n_new = size(A, 1);
W = 1./sum(A, 2) * ones(1, n_new) .* A; % construct the weight matrix
[V, D] = eigs(W + eye(n_new), 1, 'sm');
sign_vector = sign(V);
s_sign = zeros(n, 1);
s_sign(valid_idx) = sign(sum(sign_vector(1 : k)))* s_abs(valid_idx) .* sign_vector(1 : k);
end

