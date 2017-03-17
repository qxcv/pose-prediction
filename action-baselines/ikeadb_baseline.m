function ikeadb_baseline()
%Go through Anoop's Ikea database features and figure out action class
% probabilities. Should also scrape original action labels, if possible.

% this is the order used to generate the dataset
ACTIONS = {...
    'attach leg 1', 'attach leg 2', 'attach leg 3', 'attach leg 4', ...
    'detach leg 1', 'detach leg 2', 'detach leg 3', 'detach leg 4', ...
    'flip over', 'spin in', 'spin out', 'pick leg' ...
};
DIRS = {'/data/home/cherian/IkeaDataset/vgg/RGB/features', ...
        '/data/home/cherian/IkeaDataset/vgg/FLOW/features'};
DB_PATH = '/data/home/cherian/IkeaDataset/IkeaClipsDB_withactions.mat';

global IkeaDB;
if ~exist('IkeaDB', 'var') || isempty(IkeaDB)
    load(DB_PATH);
end

for i=1:length(DIRS)
    dir_path = DIRS{i};
    fprintf('Computing stats for %s\n', dir_path);
    [true_labels, est_probs] = processs_dir(dir_path, ACTIONS);
    print_stats(true_labels, est_probs);
    fprintf('\n\n\n\n\n');
end
end

function [true_labels, est_probs] = processs_dir(dir_path, actions)
global IkeaDB;
dents = dir(dir_path);
true_actions = cell([1 length(IkeaDB)]);
action_probs = cell([1 length(IkeaDB)]);
for i=1:length(dents)
    name = dents(i).name;
    if strcmp(name, '.') || strcmp(name, '..')
        continue
    end
    ident_matches = regexp(name, '^seq_(\d+)_(\d+)_(\d+).mat$', 'tokens');
    assert(length(ident_matches) == 1 && length(ident_matches{1}) == 3);
    ikeadb_idx = str2double(ident_matches{1}{1});
    if isempty(true_actions{ikeadb_idx})
        fprintf('Working on sequence %d (%d in total)\n', ikeadb_idx, ...
            length(IkeaDB));
    end
    start_frame = str2double(ident_matches{1}{2});
    end_frame = str2double(ident_matches{1}{3});
    nframes = end_frame - start_frame + 1;
    l = load(fullfile(dir_path, name));
    % recall that feat is F*T
    feat = l.feat;
    softmax_in = feat(end-length(actions)+1:end, :)';
    probs = softmax(softmax_in);
    assert(all(size(probs) == [nframes, length(actions)]));
    
    % A note on matching: IkeaDB has 93 different videos (albeit with
    % non-sequential video IDs which have some gaps). The seq_X_*.mat files
    % in the features directory have Xs numbered from 1 to 93 with no gaps,
    % so I'm guessing that they identify the order of the video within
    % IkeaDB, not the corresponding .video_id.
    true_actions{ikeadb_idx} = IkeaDB(ikeadb_idx).activity_id;
    if isempty(action_probs{ikeadb_idx})
        action_probs{ikeadb_idx} = nan([...
            IkeaDB(ikeadb_idx).num_frames, ...
            length(actions)]);
    end
    % This should un-NaN the frames which we actually have actions for
    % (which won't be every frame).
    action_probs{ikeadb_idx}(start_frame:end_frame, :) = probs;
end
[true_labels, est_probs] ...
    = simplify_collected_data(action_probs, true_actions);
end

function [true_labels, est_probs] ...
    = simplify_collected_data(action_probs, true_actions)
% simplify data collected by process_dir so that it is easy to evaluate
% with generic functions.
est_probs = [];
true_labels = [];
for i=1:length(true_actions)
    true_seq = true_actions{i};
    prob_seq = action_probs{i};
    if isempty(true_seq) || isempty(prob_seq)
        continue
    end
    valid_steps = ~any(isnan(prob_seq), 2);
    valid_prob_seq = prob_seq(valid_steps, :);
    valid_true_seq = true_seq(valid_steps);
    if isempty(est_probs)
        est_probs = valid_prob_seq;
    else
        est_probs = cat(1, est_probs, valid_prob_seq);
    end
    if isempty(true_labels)
        true_labels = valid_true_seq;
    else
        true_labels = cat(1, true_labels, valid_true_seq);
    end
end
end

function print_stats(true_labels, est_probs)
assert(length(true_labels) == length(est_probs));
if isempty(true_labels)
    fprintf('No labels to compute stats for\n');
    return
end
[~, est_labels] = max(est_probs, [], 2);
% Accuracy (just % of correct labels)
accuracy = sum(true_labels == est_labels) / length(true_labels);

% F1 score
num_classes = size(est_probs, 2);
precisions = zeros([1 num_classes]);
recalls = zeros([1 num_classes]);
f1s = zeros([1 num_classes]);
supports = zeros([1 num_classes]);
for class=1:num_classes
    true_pos = sum((true_labels == class) & (est_labels == class));
    % true_neg = sum((true_labels ~= class) & (est_labels ~= class));
    false_pos = sum((true_labels ~= class) & (est_labels == class));
    false_neg = sum((true_labels == class) & (est_labels ~= class));
    precisions(class) = true_pos / (true_pos + false_pos);
    recalls(class) = true_pos / (true_pos + false_neg);
    f1s(class) = 2 * precisions(class) * recalls(class) ...
        / (precisions(class) + recalls(class));
    supports(class) = sum(true_labels == class);
end
% this is how sklearn computes multiclass accuracies
precision = wavg(precisions, supports);
recall = wavg(recalls, supports);
f1 = wavg(f1s, supports);

% mean negative log likelihood
% apparently the ' after 1:size(est_probs, 1) really is necessary
likelihoods = est_probs(...
    sub2ind(size(est_probs), (1:size(est_probs, 1))', true_labels));
assert(isvector(likelihoods) && length(likelihoods) == size(est_probs, 1));
mnll = mean(-log(likelihoods));

fprintf('Accuracy: %.5f\n', accuracy);
fprintf('Precision: %.5f\n', precision);
fprintf('Recall: %.5f\n', recall);
fprintf('F1: %.5f\n', f1);
fprintf('MNLL: %.5f\n', mnll);
end

function result = wavg(vals, weights)
total_weight = sum(weights);
assert(all(weights >= 0));
if total_weight <= 0
    result = mean(vals);
else
    norm_weights = weights / total_weight;
    result = sum(vals .* norm_weights);
end
end

function u = softmax(t)
% Apply softmax activation over last dimension of tensor.
exps = exp(t);
sums = sum(exps, ndims(t));
u = bsxfun(@rdivide, exps, sums);
end