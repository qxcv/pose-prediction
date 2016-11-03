% Train & test predictor (or something)

startup;

if ~exist('db', 'var')
    % XXX: Stupid speedup trick
    db = IkeaDB;
end
sample_info = db.seqinfo(1);
offsets = sample_info.offsets;
njoints = sample_info.njoints;
val_ids = find(db.is_val);
taps = 250:5:300;
predictors = struct('name', {...
      'extend', 'average', 'linear2', 'velocity', 'svr', ...
    }, 'predictor', {...
       Extend(offsets, njoints), ...
       Average(offsets, njoints), ...
       LeastSquares(offsets, njoints, taps), ...
       Velocity(offsets, njoints), ...
       SVR(offsets, njoints, taps), ...
});
 
% Train each model
if ~exist('train_seqs', 'var')
    % XXX: Another stupid speedup trick
    train_seqs = cellfun(@db.seqinfo, num2cell(find(db.is_train)), ...
        'UniformOutput', false);
end
for pi=1:length(predictors)
    fprintf('Training %s\n', predictors(pi).name);
    pred = predictors(pi).predictor;
    pred.train(train_seqs);
end

% Get predictions on the validation set
fprintf('Training done. Evaluating on validation set\n');
all_preds = zeros([njoints, 2, length(offsets), length(val_ids), length(predictors)]);
all_gts = zeros([njoints, 2, length(offsets), length(val_ids)]);
for val_seq_pos=1:length(val_ids)
    fprintf('Working on seq %i/%i\n', val_seq_pos, length(val_ids));
    
    val_seq_id = val_ids(val_seq_pos);
    info = db.seqinfo(val_seq_id);
    assert(info.njoints == njoints);
    assert(all(info.offsets == offsets));
    
    test_poses = info.poses(:, :, 1:info.ntrain);
    all_gts(:, :, :, val_seq_pos) = info.poses(:, :, info.offsets);
    
    these_preds = zeros([njoints, 2, length(offsets), length(predictors)]);
    for i=1:length(predictors)
        predictor = predictors(i).predictor;
        preds = predictor.predict(test_poses);
        these_preds(:, :, :, i) = preds;
    end
    all_preds(:, :, :, val_seq_pos, :) = these_preds;
end

thresholds = 0:0.025:0.5;
all_pckh = zeros([length(thresholds), length(predictors), length(offsets)]);
for off_id=1:length(offsets)
    for pred_id = 1:length(predictors)
        gts = squeeze(all_gts(:, :, off_id, :));
        preds = squeeze(all_preds(:, :, off_id, :, pred_id));
        cell_gts = squeeze(num2cell(gts, [1 2]));
        cell_preds = squeeze(num2cell(preds, [1 2]));
        % Calculate PCKh@thresh averaged across wrists and elbows
        pckh = pck(cell_preds, cell_gts, thresholds, [1 2]);
        all_pckh(:, pred_id, off_id) = cellfun(@(c) mean(c([4 5 7 8])), pckh);
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
    title(sprintf('PCKh, frame %i', offsets(off_id)));
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
