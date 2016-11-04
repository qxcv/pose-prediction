function [normed, params] = norm_seq(seq)
%NORM_SEQ Normalise a sequence of poses
% Format: joints * xy * times
assert(ndims(seq) == 3);
assert(size(seq, 2) == 2);
normed = seq(:, :, 2:end) - seq(:, :, 1:end-1);
params.init = seq(:, :, 1);
end