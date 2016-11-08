% Train & test predictor (or something)

startup;

if ~exist('db', 'var')
    % XXX: Stupid speedup trick
    % db = IkeaDB;
    db = H36MDB;
end
sample_info = db.seqinfo(1);
offsets = sample_info.offsets;
njoints = sample_info.njoints;
ntrain = sample_info.ntrain;
% val_ids = find(db.is_val);
% XXX: Hack to use manually annotated test poses
val_ids = find(db.is_test);
if ~exist('predictors', 'var')
    predictors = struct('name', {...
          'extend', ...
          'average', ...
          'linear2', ...
          'velocity', ...
          ... 'svr', ...
          'gpr' ...
        }, 'is_trained', false, ...
        'predictor', {...
           Extend(offsets, njoints), ...
           Average(offsets, njoints), ...
           LeastSquares(offsets, njoints, 240:5:300), ...
           Velocity(offsets, njoints), ...
           ... SVR(offsets, njoints, 280:5:300), ...
           GPR(offsets, njoints, 280:5:300), ...
    });
end
 
% Train each model
if ~exist('train_seqs', 'var')
    train_seqs = cellfun(@db.seqinfo, num2cell(find(db.is_train)), ...
        'UniformOutput', false);
end
for pi=1:length(predictors)
    % XXX: Yet another hack to save re-training
    if ~predictors(pi).is_trained
        fprintf('Training %s\n', predictors(pi).name);
        pred = predictors(pi).predictor;
        pred.train(train_seqs);
        predictors(pi).is_trained = true;
    end
end

% Get predictions on the validation set
% XXX: Currently using test set because there are no reliable validation
% annotations (or training annotations, but that's a different story…).
fprintf('Training done. Evaluating on validation set\n');
all_preds = zeros([njoints, 2, length(offsets), length(val_ids), length(predictors)]);
all_gts = zeros([njoints, 2, length(offsets), length(val_ids)]);
test_poses = zeros([njoints, 2, ntrain, length(val_ids)]);
for val_seq_pos=1:length(val_ids)
    val_seq_id = val_ids(val_seq_pos);
    info = db.seqinfo(val_seq_id);
    assert(info.njoints == njoints);
    assert(all(info.offsets == offsets));
    
    test_poses(:, :, :, val_seq_pos) = info.poses(:, :, 1:info.ntrain);
    % all_gts(:, :, :, val_seq_pos) = info.poses(:, :, info.offsets);
    all_gts(:, :, :, val_seq_pos) = info.test_poses;
end
% TODO: Make this a parfor once I figure out how to shrink my predictors :/
for val_seq_pos=1:length(val_ids)        
    these_tps = test_poses(:, :, :, val_seq_pos);
    these_preds = zeros([njoints, 2, length(offsets), length(predictors)]);
    for i=1:length(predictors)
        predictor = predictors(i).predictor;
        preds = predictor.predict(these_tps);
        these_preds(:, :, :, i) = preds;
    end
    all_preds(:, :, :, val_seq_pos, :) = these_preds;
end

% thresholds = 0:0.025:0.5;
thresholds = 0:1:200;
all_pckh = zeros([length(thresholds), length(predictors), length(offsets)]);
for off_id=1:length(offsets)
    gts = squeeze(all_gts(:, :, off_id, :));
    cell_gts = squeeze(num2cell(gts, [1 2]));
    for pred_id = 1:length(predictors)
        preds = squeeze(all_preds(:, :, off_id, :, pred_id));
        cell_preds = squeeze(num2cell(preds, [1 2]));
        % Calculate PCKh@thresh averaged across wrists and elbows
        % XXX: No longer using pckh. Now using non-normalised measure :/
        % pckh = pck(cell_preds, cell_gts, thresholds, [1 2]);
        pckh = pck(cell_preds, cell_gts, thresholds);
        all_pckh(:, pred_id, off_id) = cellfun(@(c) mean(c([4 5 7 8])), pckh);
        % Uncomment next line to measure head/shoulders instead (I think; I
        % might have some of the joints wrong there)
        % all_pckh(:, pred_id, off_id) = cellfun(@(c) mean(c([1 2 3])), pckh);
    end
end

% Display PCKs in nice array of subplots
noff = length(offsets);
rows = floor(sqrt(noff));
cols = ceil(noff / rows);
figure;
for off_id=1:length(offsets)
    subplot(rows, cols, off_id);
    hold on;
    off_pcks = all_pckh(:, :, off_id);
    for pred_id=1:length(predictors)
        plot(thresholds, off_pcks(:, pred_id), ...
             'DisplayName', predictors(pred_id).name);
    end
    % title(sprintf('PCKh, frame %i', offsets(off_id)));
    title(sprintf('Accuracy, frame %i', offsets(off_id)));
    l = legend('show');
    ylim([0 1]);
    xlim([0 max(thresholds)]);
    l.set('location', 'best'); % why is this not the default?
    hold off;
end

% Save plots for 20 randomly chosen sequences
% dirname = fullfile('shots', datestr(datetime, 'yyyy-mm-ddTHH:MM:SS'));
% mkdir_p(dirname);
% rand_ids = randperm(length(val_ids));
% rand_ids = rand_ids(1:20);
% parfor rid=1:length(rand_ids)
%     val_seq_pos = rand_ids(rid);
%     fprintf('Saving seq %i/%i\n', rid, length(rand_ids));
% 
%     val_seq_id = val_ids(val_seq_pos);
% 
%     for off_id=1:length(offsets)
%         offset = offsets(off_id);
%         for pred_id=1:length(predictors)
%             figure('Visible', 'off');
%             axes('Visible', 'off');
%             % Need to use val_seq_id in calls to db, val_seq_pos when
%             % indexing into all_preds.
%             pred = all_preds(:, :, off_id, val_seq_pos, pred_id);
%             db.show_pose(val_seq_id, offset, pred);
%             hold off;
%             result_path = fullfile(dirname, sprintf('%s-frame-%i-seq-%i.jpg', ...
%                                                     predictors(pred_id).name, ...
%                                                     offset, val_seq_id));
%             print(gcf, '-djpeg', result_path, '-r 150');
%             delete(gcf);
%         end
%     end
% end
