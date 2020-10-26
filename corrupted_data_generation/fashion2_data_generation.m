clc
clear all;
name_non_experts = '../original_datasets/Fashion_Dataset/Annotations/MTurk_NonExperts_Results.csv';
name_mv = '../original_datasets/Fashion_Dataset/Annotations/Majority_Voting.csv';
T_non_experts = readtable(name_non_experts);
T_mv = readtable(name_mv);
gold_ids = string(T_mv.Url);
gold_label_ids = string(T_mv.Q2ExpertVote);

% read data we need
Input(:, 1) = string(T_non_experts.Input_Url1);
Input(:, 2) = string(T_non_experts.Input_Url2);
Input(:, 3) = string(T_non_experts.Input_Url3);
Input(:, 4) = string(T_non_experts.Input_Url4);
Answer(:, 1) = string(T_non_experts.Answer_specialty1);
Answer(:, 2) = string(T_non_experts.Answer_specialty2);
Answer(:, 3) = string(T_non_experts.Answer_specialty3);
Answer(:, 4) = string(T_non_experts.Answer_specialty4);
workers = str2double(string(T_non_experts.WorkerIndex));

% generate crowd matrix
worker_ids = [];
item_ids = [];
worker_label_ids = [];
for i = 1 : 4
    worker_ids = [worker_ids; workers];
    item_ids = [item_ids; Input(:, i)];
    worker_label_ids = [worker_label_ids; Answer(:, i)];
end

% transform label to number
gold_label = zeros(size(gold_label_ids));
gold_label(find(gold_label_ids == "Yes")) = 1;
gold_label(find(gold_label_ids == "No")) = 2;
worker_label = zeros(size(worker_label_ids));
worker_label(find(worker_label_ids == 'Yes')) = 1;
worker_label(find(worker_label_ids == 'No')) = 2;

% delete data with missing information
index = find(gold_label == 0); % delete data that the ground_truth information which is not valid
gold_ids(index) = [];
gold_label(index) = [];

index = find(worker_label == 0); % delete data that the worker label is not valid
worker_ids(index) = [];
item_ids(index) = [];
worker_label(index) = [];

item_name = unique(item_ids); % delete data which does not have ground_truth information
data = [worker_ids item_ids worker_label];
miss_idx = [];
for i = 1 : length(item_name)
    if isempty(find(gold_ids == item_name(i)))
        miss_idx =[miss_idx; find(item_ids == item_name(i))]; % the index of the samples with missing ground_truth labels
    end
end
data(miss_idx, :) = []; % delete the data with missing information

% generate Y_obs and ground_truth data
item_name = unique(data(:, 2));
worker_name = unique(data(:, 1));
label_name = unique(data(:, 3));
num_class = length(label_name);
num_worker = length(worker_name);
num_tasks = length(item_name);

ground_truth = zeros(num_tasks, 1); % generate ground_truth vector
for i = 1 : num_tasks
    if find(gold_ids == item_name(i))
        index = find(gold_ids == item_name(i));
        ground_truth(i) = gold_label(index);
    end
end

Y_obs = zeros(num_worker, num_tasks);

for i = 1 : length(data)
    worker_idx = find(worker_name == data(i, 1));
    task_idx = find(item_name == data(i, 2));
    Y_obs(worker_idx, task_idx) = data(i, 3);
end

worker_num_tasks = sum(sign(Y_obs), 2);

% for remove_upper_bound = 0 : 10 : 80
remove_upper_bound = 0;
index_invalid = find(worker_num_tasks < remove_upper_bound); % remove workers who did tasks less than this num
Y_obs(index_invalid, :) = [];

%remove tasks which does not have an answer(the answer removed in last step)
task_idx_invalid = find(sum(Y_obs, 1) == 0);
Y_obs(:, task_idx_invalid) = [];
ground_truth(task_idx_invalid) = [];
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
for F = 0 : 10 : 90
    
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
        
        save_name = ['../corrupted_datasets/fashion2/fashion2_raw_corruption_', num2str(F),'_', num2str(times), '.mat'];
        save(save_name, 'Y_obs', 'ground_truth', 'C', 'N', 'Y_obs_seperate', 'F', 'corrp_idx');
    end
    clear seperate1 Y_obs_seperate
end