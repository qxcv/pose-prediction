function pcps = pcp(preds, gts, limbs, dist_thresh)
%PCP Compute Percentage Correct Parts (strict) on input sequences
assert(iscell(preds) && iscell(gts));
assert(length(preds) >= 1);
assert(length(preds) == length(gts));
assert(ismatrix(preds{1}) && size(preds{1}, 2) == 2);

pred_mat = cat(3, preds{:});
gt_mat = cat(3, gts{:});
pcps = nan([1 length(limbs)]);

if ~exist('dist_thresh', 'var')
    % dist_thresh is the tolerance for error, as a fraction of limb length
    % ordinary PCP uses 0.5, but sweeping the value across a range can tell
    % you more about accuracy
    dist_thresh = 0.5;
end

for limb_id=1:length(limbs)
    limb = limbs{limb_id};
    start_gts = squeeze(gt_mat(limb(1), :, :));
    end_gts = squeeze(gt_mat(limb(2), :, :));
    lengths = dists_2d(start_gts, end_gts);
    threshs = lengths * dist_thresh;
    
    start_preds = squeeze(pred_mat(limb(1), :, :));
    end_preds = squeeze(pred_mat(limb(2), :, :));
    start_dists = dists_2d(start_preds, start_gts);
    end_dists = dists_2d(end_preds, end_gts);
    
    valid = (start_dists < threshs) & (end_dists < threshs);
    assert(length(valid) == length(preds));
    
    % Most joints should be valid, but there are some invalid ones set to
    % NaN (which makes their threshs(i) NaN as well). We need to make sure
    % that we don't get any of those.
    good_gt_mask = ~isnan(threshs);
    assert(mean(good_gt_mask) >= 0.6);
    valid = valid(good_gt_mask);
    assert(length(valid) >= 0.6 * length(preds));
    
    pcps(limb_id) = sum(valid) / length(valid);
    assert(~isnan(pcps(limb_id)) && isfinite(pcps(limb_id)));
end
end

function dists = dists_2d(mat1, mat2)
% Used to compute coordinatewise dists between matrices of 2D coordinates
assert(ismatrix(mat1) && ismatrix(mat2) && size(mat1, 1) == 2 ...
    && all(size(mat1) == size(mat2)));
dists = sqrt(sum((mat2 - mat1).^2, 1));
end
