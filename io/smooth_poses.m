function smoothed = smooth_poses(poses)
%SMOOTH_POSES Apply a linear filter to smooth out a pose sequence.
assert(ndims(poses) >= 2 && ndims(poses) <= 3);
assert(size(poses, 2) == 2);
smoothed = nan(size(poses));
for i=1:size(poses, 1)
    for j=1:size(poses, 2)
        % Use Matlab's default moving average smoother
        smoothed(i, j, :) = smooth(poses(i, j, :));
    end
end
end