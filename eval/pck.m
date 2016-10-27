function accs = pck(preds, gts, threshs, norm_joints)
%PCK Compute Percentage Correct Keypoints (PCK), as per MODEC paper.
% See Section 6.1 of that paper. `preds` is cell array of predicted poses,
% `gts` is cell array of GT poses, `threshs` is array of thresholds,
% `norm_joints` is a pair indicating which two joints should be used for
% normalisation.
will_norm = exist('norm_joints', 'var');
if will_norm
    assert(length(norm_joints) == 2);
else
    warning('JointRegressor:pck:nonorms', ...
        'No normalisation joints provided, computing PCK at image scale');
end
assert(iscell(preds) && iscell(gts) && isnumeric(threshs));

% Find gt_norms, which gives a scale for each joint
pred_mat = cat(3, preds{:});
gt_mat = cat(3, gts{:});
if will_norm
    scale_diffs = squeeze(gt_mat(norm_joints(1), :, :) - gt_mat(norm_joints(2), :, :));
    scales = sqrt(sum(scale_diffs.^2, 1));
    % The smallest 1% of scales are probably people in awkward positions.
    % Can't do much with those scales, so we just replace them with
    % something that won't cause the PCK estimate to asplode.
    tiny = prctile(scales, 1);
    scales(scales < tiny) = tiny;
    assert(all(scales) > 0, 'Can''t have intersecting normalisation joints!');
else
    scales = ones(size(pred_mat, 1), length(preds));
end

% Now find distances for each joint
all_diffs = pred_mat - gt_mat;
all_norms = bsxfun(@rdivide, squeeze(sqrt(sum(all_diffs.^2, 2))), scales);
assert(all(size(all_norms) == [size(pred_mat, 1), length(preds)]));

% Remove stuff that's NaN
good_mask = ~squeeze(sum(isnan(gt_mat), 2));
assert(all(size(good_mask) == size(all_norms)));
assert(mean(good_mask(:)) >= 0.6); % Most joints should not be NaN
% We'll set the norms corresponding to NaN ground truths to inf so that
% they never fall under the threshold. When we compute the mean number of
% valid parts below, we'll have to account for that.
all_norms(~good_mask) = inf;
removed_joints = sum(~good_mask, 2);
num_valid_samples = size(all_norms, 2) - removed_joints;

accs = cell([1 length(threshs)]);
for tidx=1:length(threshs)
    thresh = threshs(tidx);
    accs{tidx} = sum(all_norms < thresh, 2) ./ num_valid_samples;
end
end
