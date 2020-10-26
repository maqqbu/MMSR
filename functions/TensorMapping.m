function M = TensorMapping(T,A1,A2,A3)
    [n1 n2 n3] = size(T);
    m1 = size(A1,2);
    m2 = size(A2,2);
    m3 = size(A3,2);
    
    M = zeros(m1,m2,m3);
    for i1 = 1:m1
        for i2 = 1:m2
            for i3 = 1:m3
                % calculate (i1,i2,i3) entry
                for j1 = 1:n1
                    for j2 = 1:n2
                        for j3 = 1:n3
                            M(i1,i2,i3) = M(i1,i2,i3) + T(j1,j2,j3)*A1(j1,i1)*A2(j2,i2)*A3(j3,i3);
                        end
                    end
                end
            end
        end
    end
end

