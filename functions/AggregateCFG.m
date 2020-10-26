function newmu = AggregateCFG(mu, mode)
    k = size(mu,1);
    if mode == 0
        % general model
        newmu = mu;
    elseif mode == 1    
        % one-coin model
        p = sum(diag(mu))/sum(sum(mu));
        newmu = eye(k)*(p - (1-p)/(k-1)) + ones(k,k)*(1-p)/(k-1);
    elseif mode == 2
        % category-level one-coin model
        p = (diag(mu))'./sum(mu,1);
        newmu = diag(p - (1-p)/(k-1)) + repmat((1-p)/(k-1),k,1);
    end
end
