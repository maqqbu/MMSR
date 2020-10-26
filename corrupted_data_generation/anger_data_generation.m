clc
clear all;

X=tdfread('../original_datasets/all_collected_data/anger.standardized.tsv');
worker_ids = string(X.x0x21amt_worker_ids);
item_ids = string(X.orig_id);
worker_label_ids = X.response;
gold_label = X.gold;

% transform the ground truth score to multi classes: negative, moderate and strong
neg_idx = find(gold_label == 0);
pos_idx = find(gold_label > 0);
gold_label(neg_idx) = 1;
gold_label(pos_idx) = 2;

% transform the worker label score to multi classes
neg_idx = find(worker_label_ids == 0);
pos_idx = find(worker_label_ids > 0);
worker_label_ids(neg_idx) = 1;
worker_label_ids(pos_idx) = 2;
% generate the sequence of the workers and the tasks
item_name = unique(item_ids);
worker_name = unique(worker_ids);
label_name = unique(gold_label);
num_tasks = length(item_name);
num_worker = length(worker_name);
num_class = length(label_name);
% generate Y_obs
Y_obs = zeros(num_worker, num_tasks);

for i = 1 : length(item_ids)
    worker_idx = find(worker_name == worker_ids(i));
    task_idx = find(item_name == item_ids(i));
    Y_obs(worker_idx, task_idx) = worker_label_ids(i);
end
% generate ground_truth vector
ground_truth = zeros(num_tasks, 1); % generate ground_truth vector
for i = 1 : num_tasks
    index = find(item_ids == item_name(i));
    ground_truth(i) = gold_label(index(1));
end

worker_num_tasks = sum(sign(Y_obs), 2);


% the corrupted style of adversaries
possible_answer = repmat([1 : num_class]', 1, num_tasks);
true_answer = ones(num_class, 1)* ground_truth';
possible_answer(possible_answer == true_answer) = []; % delete the true answer from possible answer matrix
wrong_answer = reshape(possible_answer, num_class - 1, num_tasks);
wrong_answer_vec = wrong_answer(1, :);

% the fraction of tasks that adversaries give right answer
f = 0.1;
rand_task_idx = randperm(num_tasks);
correct_task_idx = rand_task_idx(1 : round(num_tasks * f));%let f fraction tasks be correctly predicted by adversary
oppo_task_idx = rand_task_idx(round(num_tasks * f)+1 : num_tasks);
Y1 = Y_obs;


%  for remove_upper_bound = 0 : 10 : 80
remove_upper_bound = 0;
Y_obs = Y1;
index_invalid = find(worker_num_tasks <= remove_upper_bound); % remove workers who did tasks less than this num
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


graph_sparsity = sum(sum(sign(N_original)))/num_worker^2;

%%
ad_sparsity = 0.5;
for F = 0 : 2 : 20
    
    for times = 1 : 3
        Y_obs = Y2;

        % corrupt workers randomly
        rand_worker_idx = randperm(num_worker);
        corrp_idx = rand_worker_idx(1 : F);
        Y_obs(corrp_idx, correct_task_idx) = ones(F,1) * ground_truth(correct_task_idx)';
        Y_obs(corrp_idx, oppo_task_idx) = ones(F,1) * wrong_answer_vec(oppo_task_idx);
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
        
        save_name = ['../corrupted_datasets/anger/anger_38_corruption_', num2str(F),'_', num2str(times), '.mat'];
        save(save_name, 'Y_obs', 'ground_truth', 'C', 'N', 'Y_obs_seperate', 'F', 'corrp_idx');
    end
    clear seperate1 Y_obs_seperate
end
