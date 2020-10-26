function muW = SolveCFG(x1,x2,x3)

[k n] = size(x1);
x1_tilde = (x3*x2'/n)/(x1*x2'/n) * x1;
x2_tilde = (x3*x1'/n)/(x2*x1'/n) * x2;
M2 = x1_tilde*x2_tilde'/n;
M3 = zeros(k,k,k);

for i1 = 1:k
    for i2 = 1:k
        for i3 = 1:k
            M3(i1,i2,i3) = sum(x1_tilde(i1,:).*x2_tilde(i2,:).*x3(i3,:))/n;
        end
    end
end

% Whitening
[U S V] = svd(M2);
W = U*(S^-0.5);
M3W = TensorMapping(M3,W,W,W);

% Robust Power Method: estimate cfmatrix(:,:,3)
mu3_p = zeros(k,k);
w_p = zeros(k,1);
for c = 1:k
   vec = RobustPowerMethod(M3W);
   lambda = TensorMapping(M3W,vec,vec,vec);
   for i1 = 1:k
        for i2 = 1:k
            for i3 = 1:k
                M3W(i1,i2,i3) = M3W(i1,i2,i3) - lambda*vec(i1)*vec(i2)*vec(i3);
            end
        end
   end
   mu_tmp = lambda*pinv(W')*vec;
   [value index] = max(mu_tmp);
   if w_p(index) ~= 0
       empty_slots = find(w_p == 0);
       index = empty_slots(1);
   end
   mu3_p(:,index) = mu_tmp;
   w_p(index) = (1/lambda)^2;
end

muW = [mu3_p w_p];
end