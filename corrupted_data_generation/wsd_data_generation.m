clc
clear all;

X=tdfread('../original_datasets/wsd.standardized.tsv');
worker_ids = string(X.x0x21amt_worker_ids);
item_ids = string(X.orig_id);
workername = unique(worker_ids);
itemname = unique(item_ids);
goldlabel = X.gold;

for i = 1:length(workername)
    if find(worker_ids == workername(i))
        index = find(worker_ids == workername(i));
        workers(index) = i;
    end
end

for i = 1:length(itemname)
    if find(item_ids == itemname(i))
        index = find(item_ids == itemname(i));
        items(index) = i;
        y(i) = goldlabel(index(1));
    end
end
Ylabel = zeros(length(workername),length(itemname));
for i = 1:length(workername)
    for j = 1:length(itemname)
        index = (worker_ids ==workername(i));
        index = (item_ids==itemname(j)).*index;
        if find(index==1)
            lab = find(index==1);
            Ylabel(i,j) = X.response(lab);
        end
    end
end
Y_obs = Ylabel;
ground_truth = y';
worker_num_tasks = sum(sign(Y_obs), 2);


num_class = max(unique(ground_truth));
num_tasks = size(Y_obs, 2);

% the corrupted style of adversaries
possible_answer = repmat([1 : num_class]', 1, num_tasks);
true_answer = ones(num_class, 1)* ground_truth';
possible_answer(possible_answer == true_answer) = []; % delete the true answer from possible answer matrix
wrong_answer = reshape(possible_answer, num_class - 1, num_tasks);
wrong_answer_vec = wrong_answer(1, :);

% for f = 0: 0.1 : 1   % the fraction of tasks that adversaries give right answer
f = 0.1;
rand_task_idx = randperm(num_tasks);
correct_task_idx = rand_task_idx(1 : round(num_tasks * f));%let f fraction tasks be correctly predicted by adversary
oppo_task_idx = rand_task_idx(round(num_tasks * f)+1 : num_tasks);
Y1 = Y_obs;


%  for remove_upper_bound = 0 : 10 : 80
remove_upper_bound = 0;
Y_obs = Y1;
index_invalid = find(worker_num_tasks < remove_upper_bound); % remove workers who did tasks less than this num
Y_obs(index_invalid, :) = [];

Y2 = Y_obs;
num_worker = size(Y_obs, 1);

N_original = zeros(num_worker);
for i = 1 : num_worker
    for j = 1 : num_worker
        if i == j
            N_original(i, j) == 0;
        else
            N_original(i, j) = sum(Y_obs(i, :) & Y_obs(j, :));
        end
    end
end

ad_sparsity = 0.5;
for F = 0 : 2 : 18
    
    for times = 1 : 3
  
        Y_obs = Y2;
        num_group = 2;
        rand_worker_idx = randperm(num_worker);
        corrp_idx = rand_worker_idx(1 : F);
        group_capacity = floor(F/num_group);
        for i = 0 : num_group-1
            corrp_idx_current = corrp_idx(i*group_capacity + 1: (i+1)*group_capacity);
            rand_task_idx = randperm(num_tasks);
            correct_task_idx = rand_task_idx(1 : round(num_tasks * f));%let f fraction tasks be correctly predicted by adversary
            oppo_task_idx = rand_task_idx(round(num_tasks * f)+1 : num_tasks);
            current_answer = zeros(1, num_tasks);
            current_answer(correct_task_idx) = ground_truth(correct_task_idx)';
            current_answer(oppo_task_idx) = wrong_answer_vec(oppo_task_idx);
            Y_obs(corrp_idx_current, :) = ones(length(corrp_idx_current), 1) * current_answer;
            clear corrp_idx_current
        end
        remaining_idx = corrp_idx((i+1)*group_capacity +1 : F);
        if F ~= 0
            Y_obs(remaining_idx, :) = ones(length(remaining_idx), 1) * current_answer;
        end
        
        Y_obs(corrp_idx, :) = Y_obs(corrp_idx, :) .*  binornd(1, ad_sparsity, F, num_tasks);
        
        
        
        Y_obs_seperate = zeros(num_worker, num_tasks, num_class);
        for i = 1 : num_class
            seperate1 = zeros(num_worker, num_tasks);
            seperate1(find(Y_obs == i)) = 1;
            Y_obs_seperate(:, :, i) = seperate1;
        end
        
        N = zeros(num_worker);
        for i = 1 : num_worker
            for j = 1 : num_worker
                if i == j
                    N(i, j) == 0;
                else
                    N(i, j) = sum(Y_obs(i, :) & Y_obs(j, :));
                end
            end
        end
        
        C = zeros(num_worker);
        for i = 1 : num_worker
            for j = 1 : num_worker
                if N(i, j) ~= 0
                    valid_idx = Y_obs(i, :) & Y_obs(j, :);
                    C(i, j) = num_class/((num_class - 1) * N(i,j)) * sum((Y_obs(i, :) ...
                        == Y_obs(j, :)).* valid_idx) - 1/(num_class - 1); % multilabel equation in JMLR
                end
            end
        end
        sparsity = min(sum(sign(N)));
        
        save_name = ['../corrupted_datasets/wsd/wsd_34_corruption_', num2str(F),'_', num2str(times), '.mat'];
        save(save_name, 'Y_obs', 'ground_truth', 'C', 'N', 'Y_obs_seperate', 'F', 'corrp_idx');
    end
    clear seperate1 Y_obs_seperate
end
