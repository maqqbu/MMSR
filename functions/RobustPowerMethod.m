function vec_max = RobustPowerMethod(M)
L = 100;
iter = 100;
lambda_max = -1;
d = size(M,1);
vec_max = zeros(d,1);
for i = 1:L
    ranvec = rand(d,1) - 0.5*ones(d,1);
    ranvec = ranvec / norm(ranvec);
    for j = 1:iter
        ranvec = TensorMapping(M,eye(d),ranvec,ranvec);
        ranvec = ranvec / norm(ranvec);
    end
    lambda = TensorMapping(M,ranvec,ranvec,ranvec);
    if lambda > lambda_max
        lambda_max = lambda;
        vec_max = ranvec;
    end
end
end