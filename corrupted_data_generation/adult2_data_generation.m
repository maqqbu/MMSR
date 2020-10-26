clc
clear all
filename1 = '../original_datasets/adult2/labels.txt';
filename2 = '../original_datasets/adult2/gold.txt';
[worker_ids, item_ids, worker_label_ids] = textread(filename1 , '%s %s %s');
[gold_ids, gold_label_ids] = textread(filename2 , '%s %s');
worker_ids = string(worker_ids);
item_ids = string(item_ids);
worker_label_ids = string(worker_label_ids);
gold_ids = string(gold_ids);
gold_label_ids = string(gold_label_ids);
workername = unique(worker_ids);
itemname = unique(item_ids);
labelname = unique(gold_label_ids);


worker_label = zeros(size(worker_label_ids));
gold_label = zeros(size(gold_label_ids));
for i = 1:length(labelname)
    if find(worker_label_ids == labelname(i))
        index = find(worker_label_ids == labelname(i));
        worker_label(index) = i;
    end
    if find(gold_label_ids == labelname(i))
        index = find(gold_label_ids == labelname(i));
        gold_label(index) = i;
    end
end

% % plug into the original generation model
items = unique(item_ids);
num_tasks = length(items);
data = [worker_ids item_ids worker_label];

miss_idx = [];

for i = 1 : num_tasks
    if isempty(find(gold_ids == items(i)))
        miss_idx =[miss_idx; find(item_ids == items(i))]; % the index of the samples with missing labels
    end
end
data(miss_idx, :) = []; % delete the data with missing information



items = unique(data(:, 2));
workers = unique(data(:, 1));
labels = unique(data(:, 3));
num_class = length(labels);
num_tasks = length(items);
num_worker = length(workers);

% generate ground_truth vector
ground_truth = zeros(num_tasks, 1);
for i = 1 : length(items)
    if find(gold_ids == items(i))
        index = find(gold_ids == items(i));
        ground_truth(i) = gold_label(index);
    end
end

Y_obs = zeros(num_worker, num_tasks);

for i = 1 : length(data)
    worker_idx = find(workers == data(i, 1));
    task_idx = find(items == data(i, 2));
    Y_obs(worker_idx, task_idx) = data(i, 3);
end

worker_num_tasks = sum(sign(Y_obs), 2);

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


% for remove_upper_bound = 0 : 5 : 30
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

graph_density = sum(sum(sign(N_original)))/(num_worker^2 - num_worker);

ad_sparsity = 0.5;
for F = 0 : 10 : 100
    
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
        
        save_name = ['../corrupted_datasets/adult2/adult_raw_corruption_', num2str(F),'_', num2str(times), '.mat'];
        save(save_name, 'Y_obs', 'ground_truth', 'C', 'N', 'Y_obs_seperate', 'F', 'corrp_idx');
    end
    clear seperate1 Y_obs_seperate
end


