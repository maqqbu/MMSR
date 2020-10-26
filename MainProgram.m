clc
clear
close
addpath('functions')
bird_list = 0 : 2 : 20;
dog_list = 0 : 6 : 54;
RTE_list = 0 : 8 : 80;
temp_list = 0 : 5 : 40;
web_list = 0 : 10 : 80;
wsd_list = 0 : 2 : 18;
trec_list = 0 : 30 : 240;
adult2_list = 0 : 10 : 20;
fashion1_list = 0 : 10 : 90;
fashion2_list = 0 : 10 : 90;
anger_list = 0 : 2 : 20;
disgust_list = 0 : 2 : 20;
fear_list = 0 : 2 : 20;
joy_list = 0 : 2 : 20;
sadness_list = 0 : 2 : 20;
surprise_list = 0 : 2 : 20;
valence_list = 0 : 2 : 20;
% To test other datasets, replace "temp_list" with other datasets, also load the other datasets 
corruption_num_list = temp_list;
for z = 1 : length(corruption_num_list)
    for times = 1 : 3
        % load datasets
         name = ['./corrupted_datasets/temp/temp_76_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        %name = ['./corrupted_datasets/surprise/surprise_38_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        %name = ['./corrupted_datasets/bird/bird_39_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        % name = ['./corrupted_datasets/bird/bird_39_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        % name = ['./corrupted_datasets/dog/dog_raw_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        % name = ['./corrupted_datasets/RTE/RTE_raw_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        % name = ['./corrupted_datasets/web/web_raw_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        % name = ['./corrupted_datasets/wsd/wsd_34_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        % name = ['./corrupted_datasets/adult2/adult_raw_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        % name = ['./corrupted_datasets/fashion1/fashion1_raw_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        % name = ['./corrupted_datasets/fashion2/fashion2_raw_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        % name = ['./corrupted_datasets/anger/anger_38_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        % name = ['./corrupted_datasets/disgust/disgust_38_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        % name = ['./corrupted_datasets/fear/fear_38_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        % name = ['./corrupted_datasets/joy/joy_38_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        % name = ['./corrupted_datasets/sadness/sadness_38_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        % name = ['./corrupted_datasets/valence/valence_38_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        load(name)
        [num_workers, num_tasks] = size(Y_obs);
        num_class = max(ground_truth);
        y = ground_truth;
        A = [];
        Z = zeros(num_tasks, num_class, num_workers);
        for i = 1 : num_class
            index = find(Y_obs_seperate(:,:,i) ~= 0);
            [worker_idx , task_idx] = ind2sub([num_workers, num_tasks], index);
            class_idx = i * ones(size(task_idx));
            A = [A; task_idx, worker_idx, class_idx];
            index_Z = sub2ind(size(Z), task_idx, class_idx, worker_idx);
            Z(index_Z) = 1;
        end
        
        B(:, 1) = [1 : num_tasks];
        B(:, 2) = ground_truth;
        n = num_tasks;
        m = num_workers;
        k = num_class;
        
        Nround = 1;
        mode = 1;
        
        error1_predict = zeros(1, Nround);
        error2_predict = zeros(1, Nround);
        
        
        % ====================================================
        % n = max(A(:,1));
        % m = max(A(:,2));
        % k = max(B(:,2));
        %
        % y = zeros(n,1);
        % for i = 1:size(B,1)
        %     y(B(i,1)) = B(i,2);
        % end
        valid_index = find(y > 0);
        %
        % Z = zeros(n,k,m);
        % for i = 1:size(A,1)
        %     Z(A(i,1),A(i,3),A(i,2)) = 1;
        % end
        %%
        %===================== majority vote ================
        q = mean(Z,3);
        [I J] = max(q');
        accuracy = 0;
        for j = 1:n
            maxq = max(q(j,:));
            if y(j) > 0 && q(j,y(j)) == maxq
                accuracy = accuracy + 1 / size(find(q(j,:) == maxq), 2);
            end
        end
        error_majority_vote = 1 - accuracy / size(valid_index,1);
        error_mv(times, z) = error_majority_vote
        %         %===================== Sewoung Oh ================
        t = zeros(n,k-1);
        for l = 1:k-1
            U = zeros(n,m);
            for i = 1:size(A,1)
                U(A(i,1),A(i,2)) = 2*(A(i,3) > l)-1;
            end
            
            B = U - ones(n,1)*(ones(1,n)*U)/n;
            [U S V] = svd(B);
            u = U(:,1);
            v = V(:,1);
            u = u / norm(u);
            v = v / norm(v);
            pos_index = find(v>=0);
            if sum(v(pos_index).^2) >= 1/2
                t(:,l) = sign(u);
            else
                t(:,l) = -sign(u);
            end
        end
        
        J = ones(n,1)*k;
        for j = 1:n
            for l = 1:k-1
                if t(j,l) == -1
                    J(j) = l;
                    break;
                end
            end
        end
        error_KOS = mean(y(valid_index) ~= (J(valid_index)))
        error_kos(times, z) = error_KOS;
        %         %===================== Ghosh-SVD ================
        t = zeros(n,k-1);
        for l = 1:k-1
            O = zeros(n,m);
            for i = 1:size(A,1)
                O(A(i,1),A(i,2)) = 2*(A(i,3) > l)-1;
            end
            
            [U S V] = svd(O);
            u = sign(U(:,1));
            if u'*sum(O,2) >= 0
                t(:,l) = sign(u);
            else
                t(:,l) = -sign(u);
            end
        end
        
        J = ones(n,1)*k;
        for j = 1:n
            for l = 1:k-1
                if t(j,l) == -1
                    J(j) = l;
                    break;
                end
            end
        end
        error_GhostSVD = mean(y(valid_index) ~= (J(valid_index)))
        error_ghost(times, z) = error_GhostSVD;
        %         % %===================== Ratio of Eigenvalues ================
        t = zeros(n,k-1);
        for l = 1:k-1
            O = zeros(n,m);
            for i = 1:size(A,1)
                O(A(i,1),A(i,2)) = 2*(A(i,3) > l)-1;
            end
            G = abs(O);
            %
            %             % ========== algorithm 1 =============
            %         [U S V] = svd(O'*O);
            %         v1 = U(:,1);
            %         [U S V] = svd(G'*G);
            %         v2 = U(:,1);
            %         v1 = v1./v2;
            %         u = O*v1;
            % ========== algorithm 2 =============
            R1 = (O'*O)./(G'*G+10^-8);
            R2 = (G'*G > 0)+1-1;
            [U S V] = svd(R1);
            v1 = U(:,1);
            [U S V] = svd(R2);
            v2 = U(:,1);
            v1 = v1./v2;
            u = O*v1;
            
            if u'*sum(O,2) >= 0
                t(:,l) = sign(u);
            else
                t(:,l) = -sign(u);
            end
        end
        
        J = ones(n,1)*k;
        for j = 1:n
            for l = 1:k-1
                if t(j,l) == -1
                    J(j) = l;
                    break;
                end
            end
        end
        error_RatioEigen = mean(y(valid_index) ~= (J(valid_index)))
        error_roe(times, z) = error_RatioEigen;
        %===================== EM with majority vote ================
        q = mean(Z,3);
        q = q ./ repmat(sum(q,2),1,k);
        mu = zeros(k,k,m);
        
        % EM update
        for iter = 1:Nround
            for i = 1:m
                mu(:,:,i) = (Z(:,:,i))'*q;
                mu(:,:,i) = AggregateCFG(mu(:,:,i),mode);
                
                for c = 1:k
                    mu(:,c,i) = mu(:,c,i)/sum(mu(:,c,i));
                end
            end
            
            
            
            q = zeros(n,k);
            for j = 1:n
                for c = 1:k
                    for i = 1:m
                        if Z(j,:,i)*mu(:,c,i) > 0
                            q(j,c) = q(j,c) + log(Z(j,:,i)*mu(:,c,i));
                        end
                    end
                end
                q(j,:) = exp(q(j,:));
                q(j,:) = q(j,:) / sum(q(j,:));
            end
            
            [I J] = max(q');
            error1_predict(iter) = mean(y(valid_index) ~= (J(valid_index))')
        end
        error1_predict;
        error_em_mv(times, z) = mean(error1_predict, 2);
        %===================== EM with spectral method ==============
        %method of moment
        group = mod(1:m,3)+1;
        Zg = zeros(n,k,3);
        cfg = zeros(k,k,3);
        for i = 1:3
            I = find(group == i);
            Zg(:,:,i) = sum(Z(:,:,I),3);
        end
        
        x1 = Zg(:,:,1)';
        x2 = Zg(:,:,2)';
        x3 = Zg(:,:,3)';
        
        muWg = zeros(k,k+1,3);
        muWg(:,:,1) = SolveCFG(x2,x3,x1);
        muWg(:,:,2) = SolveCFG(x3,x1,x2);
        muWg(:,:,3) = SolveCFG(x1,x2,x3);
        
        mu = zeros(k,k,m);
        for i = 1:m
            x = Z(:,:,i)';
            x_alt = sum(Zg,3)' - Zg(:,:,group(i))';
            muW_alt = (sum(muWg,3) - muWg(:,:,group(i)));
            mu(:,:,i) = (x*x_alt'/n) / (diag(muW_alt(:,k+1)/2)*muW_alt(:,1:k)');
            
            mu(:,:,i) = max( mu(:,:,i), 10^-6 );
            mu(:,:,i) = AggregateCFG(mu(:,:,i),mode);
            for j = 1:k
                mu(:,j,i) = mu(:,j,i) / sum(mu(:,j,i));
            end
        end
        
        % EM update
        for iter = 1:Nround
            q = zeros(n,k);
            for j = 1:n
                for c = 1:k
                    for i = 1:m
                        if Z(j,:,i)*mu(:,c,i) > 0
                            q(j,c) = q(j,c) + log(Z(j,:,i)*mu(:,c,i));
                        end
                    end
                end
                q(j,:) = exp(q(j,:));
                q(j,:) = q(j,:) / sum(q(j,:));
            end
            
            for i = 1:m
                mu(:,:,i) = (Z(:,:,i))'*q;
                
                mu(:,:,i) = AggregateCFG(mu(:,:,i),mode);
                for c = 1:k
                    mu(:,c,i) = mu(:,c,i)/sum(mu(:,c,i));
                end
            end
            
            [I J] = max(q');
            error2_predict(iter) = mean(y(valid_index) ~= (J(valid_index))')
        end
        error_em_sm(times, z) = mean(error2_predict, 2);
        %==================M_MSR===============================================
        X = abs(C);
        num_worker = size(C, 1);
        num_tasks = size(Y_obs, 2);
        num_class = length(unique(ground_truth));
        [m, n] = size(X);
        idx = abs(sign(N));
        idx = (idx==1); % the positions of the observed entries
        u = rand(m, 1);
        v = rand(n,1);
        
        sparsity = min(sum(sign(N)));
        F_parameter = floor(sparsity/2) - 1;
        for t  = 1:10000
            v_pre = v;
            u_pre = u;
            for j = 1:n
                target_v = X(:,j);
                target_v = target_v(idx(:,j))./u(idx(:,j));
                a = mean(remove(target_v, F_parameter, v(j), num_tasks));
                if isnan(a)
                    v(j) = v(j);
                else
                    v(j) = a ;
                end
            end
            
            for i = 1:m
                target_u = X(i,:)';
                target_u = target_u(idx(i,:))./v(idx(i,:));
                a  = mean(remove(target_u, F_parameter, u(i), num_tasks));
                if isnan(a)
                    u(i) = u(i);
                else
                    u(i) = a;
                end
                
            end
            
            M=u*v';
            if norm(u * v' - u_pre * v_pre', 'fro') < 1e-10
                break
            end
        end
        %%
        k = sqrt(norm(u)/norm(v));
        x_track_1 = u / k;
        x_track_2 = sign_determination_valid(C, x_track_1);
        x_track_3 = min(x_track_2, 1-1./sqrt(num_tasks));
        x_MSR = max(x_track_3, -1/(num_class-1)+1./sqrt(num_tasks));
        % prediction
        probWorker = x_MSR.*(num_class-1)/(num_class)+1/num_class; % x is s in JMLR, which is not probability
        weights = log(probWorker.*(num_class-1)./(1-probWorker));
        score = zeros(num_class, num_tasks);
        for i = 1 : num_class
            score(i, :) = sum(Y_obs_seperate(:, :, i) .* repmat(weights, 1, num_tasks), 1);
        end
        [~, predlabel] = max(score);
        
        prediction_error1 = sum(predlabel' ~= ground_truth)/num_tasks
        error_msr(times, z) = prediction_error1;
        %====================== PGD ==========================================
        x_p = 0.5*ones(num_worker,1);
        alpha=1e-5;
        x1 = zeros(num_worker,1);
        t = 0;
        while sum(abs(x_p-x1))>1e-10
            x1=x_p;
            x_p = x_p + alpha*grad1(x_p,abs(C),N);
            x_p=min(x_p,1-1./sqrt(num_tasks));
            x_p=max(x_p,-1/(num_class-1)+1./sqrt(num_tasks));
            sum(abs(x_p-x1));
            t = t + 1;
            if t == 300000
                break
            end
        end
        probWorker = x_p.*(num_class-1)/(num_class)+1/num_class; % x is s in JMLR, which is not probability
        predlabel = zeros(num_tasks,1);
        Error = 0;
        weights = log(probWorker.*(num_class-1)./(1-probWorker));
        score = zeros(num_worker, num_tasks);
        for i = 1 : num_class
            score(i, :) = sum(Y_obs_seperate(:, :, i) .* repmat(weights, 1, num_tasks), 1);
        end
        [~, predlabel] = max(score);
        prediction_error2 = sum(predlabel' ~= ground_truth)/num_tasks
        
        error_pgd(times, z) = prediction_error2;
        %====================== M-MSR twocoin ==========================================
        warning off;
        
        M_target = C;
        ground_truth(find(ground_truth == 2)) = -1;
        Y_obs(find(Y_obs == 2)) = -1;
        Omega = sign(N);
        M_target = M_target * 4;
        m = size(M_target, 1);
        n = size(Y_obs, 2);
        Y = rand(m, 2);
        X = rand(m, 2);
        F = corruption_num_list(z) + 1;
        for itr = 1 : 200
            Y_pre = Y;
            
            for i = 1 : m
                X(i, :) = twocoin_x(Omega, M_target, Y, i, F)';
            end
            
            for j = 1 : m
                Y(j, :) = twocoin_y(Omega, M_target, X, j, F)';
            end
            
            
            if norm(Y - Y_pre, 'fro') < 1e-3
                break
            end
            
        end
        norm((M_target - X * Y') .* Omega, 'fro')/ norm(M_target .* Omega)
        M_estimate = X * Y';
        [V, D] = eigs(M_estimate, 2);
        V_new = V * sqrt(D);
        V_sum = sum(V_new, 1);
        a = V_sum(1);
        b = V_sum(2);
        % solution 1
        cos_x = a/m + b/m;
        sin_x = a/m - b/m;
        k1 = sqrt(1/(cos_x^2 + sin_x^2));
        U_1 = [cos_x -sin_x; sin_x cos_x] * k1;
        solution_1 = (V_new * (U_1)' + 1)/2;
        solution_1 = min(solution_1, 0.98); % projecction to the valid interval
        solution_1 = max(solution_1, 0.02);
        
        clear cos_x sin_x k
        % solution 2
        sin_x = a/m - b/m;
        cos_x = a/m + b/m;
        k2 = 1;
        U_2 = [cos_x sin_x; sin_x -cos_x] * k2;
        solution_2 = (V_new * (U_1)' + 1)/2;;
        solution_2 = min(solution_2, 0.98); % projecction to the valid interval
        solution_2 = max(solution_2, 0.02);
        
        
        skill_R =  solution_1(:, 1);
        skill_L =  solution_1(:, 2);
        
        % give prediction
        % prediction 1
        
        
        s_pred = skill_R;
        t_pred = skill_L;
        weight_pos = log(s_pred ./ (1 - s_pred));
        weight_neg = log(t_pred ./ (1 - t_pred));
        Y_obs_pos = zeros(m, n);
        Y_obs_pos(find(Y_obs == 1)) = 1;
        Y_obs_neg = zeros(m, n);
        Y_obs_neg(find(Y_obs == -1)) = 1;
        score(1, :) = sum(Y_obs_pos .* repmat(weight_pos, 1, n), 1);
        score(2, :) = sum(Y_obs_neg .* repmat(weight_neg, 1, n), 1);
        [~, predlabel_1] = max(score);
        predlabel_1(find(predlabel_1 == 2)) = -1;
        
        error_1 = sum( ground_truth ~= predlabel_1' )/n
        
        
        % prediction 2
        s_pred = skill_L;
        t_pred = skill_R;
        weight_pos = log(s_pred ./ (1 - s_pred));
        weight_neg = log(t_pred ./ (1 - t_pred));
        Y_obs_pos = zeros(m, n);
        Y_obs_pos(find(Y_obs == 1)) = 1;
        Y_obs_neg = zeros(m, n);
        Y_obs_neg(find(Y_obs == -1)) = 1;
        score(1, :) = sum(Y_obs_pos .* repmat(weight_pos, 1, n), 1);
        score(2, :) = sum(Y_obs_neg .* repmat(weight_neg, 1, n), 1);
        [~, predlabel_2] = max(score);
        predlabel_2(find(predlabel_2 == 2)) = -1;
        
        
        error_2 = sum( ground_truth ~= predlabel_2' )/n
        prediction_error = (error_1 + error_2)/2
        error_msr_twocoin(times, z) = prediction_error;
        z
    end
end
error_pre = [mean(error_mv, 1); mean(error_kos, 1); mean(error_ghost, 1); mean(error_roe, 1);...
    mean(error_em_mv, 1); mean(error_em_sm, 1); mean(error_msr, 1); mean(error_pgd, 1); mean(error_msr_twocoin, 1)]
%% Results visualization
% compute mean error of each algorithms
error_mean = [mean(error_mv, 1); mean(error_kos, 1); mean(error_ghost, 1); mean(error_roe, 1);...
    mean(error_em_mv, 1);  mean(error_pgd, 1);   mean(error_em_sm, 1);...
    mean(error_msr_twocoin, 1); mean(error_msr, 1)];


error_pos_std = error_mean + [std(error_mv); std(error_kos); std(error_ghost); std(error_roe);...
    std(error_em_mv);  std(error_pgd);   std(error_em_sm);...
    std(error_msr_twocoin); std(error_msr)];


error_neg_std = error_mean - [std(error_mv); std(error_kos); std(error_ghost); std(error_roe);...
    std(error_em_mv);  std(error_pgd);   std(error_em_sm);...
    std(error_msr_twocoin); std(error_msr)];



color_list = {[0 0 0], [0.9290 0.6940 0.1250],[0.4940 0.1840 0.5560], [0.4660 0.6740 0.1880], [0.8500 0.44 0.84],...
    [0.74 0.56 0.56], [0 0.4470 0.7410], [0.3010 0.7450 0.9330], [1 0 0]};


marker_list = {'p', 'o',  '<', 'x','s','^', 'd', 'x', '>'};
fill_color_list = color_list;

x =  corruption_num_list;
figure
for i = 1 : 9
    A(i) = plot(x, error_mean(i, :), 'Color', color_list{i}, 'LineStyle', '-',...
        'LineWidth', 1.5, 'Marker', marker_list{i},  'MarkerFaceColor', color_list{i},'MarkerSize',5);
    hold on
end


for i = 1 : 9
    xforfill = [x, fliplr(x)];
    yforfill = [error_neg_std(i, :), fliplr(error_pos_std(i, :))];
    fill(xforfill, yforfill, fill_color_list{i} , 'FaceAlpha',0.15,'EdgeAlpha', 0,'EdgeColor','r');
    hold on
end



xlabel('# corrupted workers','fontsize',26, 'Interpreter','none');
ylabel('Prediction error', 'fontsize', 26)
xlim([0 corruption_num_list(end)])
ylim([0 1])
set(gca,'FontSize', 16)
[h,icons] = legend([A(1), A(2), A(3), A(4), A(5), A(6), A(7), A(8), A(9)],...
    'MV','KOS', 'GhostSVD', 'EoR', 'MV-D&S', 'PGD', 'OPT-D&S', 'M-MSR-twocoin', 'M-MSR','FontSize', 14);
icons = findobj(icons,'Type','line');
icons = findobj(icons,'Marker','none','-xor');
set(icons,'MarkerSize',5);
