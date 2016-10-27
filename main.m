% Train & test predictor (or something)

db = IkeaDB;
sample_info = db.seqinfo(1);
offsets = sample_info.offsets;
njoints = sample_info.njoints;
val_ids = find(db.is_val);
predictors = struct('name', {'extend', 'average'}, ...
    'predictor', {Extend(offsets, njoints), Average(offsets, njoints)});
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
    
    for i=1:length(predictors)
        predictor = predictors(i).predictor;
        preds = predictor.predict(test_poses);
        all_preds(:, :, :, val_seq_pos, i) = preds;
    end
end

all_pckh = zeros([length(offsets), length(predictors)]);
for off_id=1:length(offsets)
    for pred_id = 1:length(predictors)
        gts = squeeze(all_gts(:, :, off_id, :));
        preds = squeeze(all_preds(:, :, off_id, :, pred_id));
        cell_gts = squeeze(num2cell(gts, [1 2]));
        cell_preds = squeeze(num2cell(preds, [1 2]));
        % Calculate PCKh@0.5 averaged across wrists and elbows
        pckh = pck(cell_preds, cell_gts, 0.5, [1 2]);
        all_pckh(off_id, pred_id) = mean(pckh{1}([4 5 7 8]));
    end
end

figure;
hold on;
for pred_id=1:length(predictors);
    plot(offsets, all_pckh(:, pred_id), ...
        'DisplayName', predictors(pred_id).name);
end
legend('show');
hold off;
