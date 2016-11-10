function G = dtw_kernel(U, V)
%DTW_KERNEL Implements an RBF-like kernel based on DTW
% Several DTW kernels explained here:
% http://cbio.mines-paristech.fr/~jvert/talks/070608telecom/telecom.pdf
% I'm going to be implementing the first DTW kernel from those slides (I
% think).
global dtw_T dtw_gamma;
if isempty(dtw_T)
    dtw_T = 1;
end
if isempty(dtw_gamma)
    dtw_gamma = 1;
end
% U is m*p, V is n*p;
m = size(U, 1);
n = size(V, 1);
dists = zeros([m n]);
for i=1:m
    % Weird reshaping happens to be compatible with IndepMarkovLearner
    % code; probably stale by the time you're reading this.
    Ut = reshape(U(i, :), [], dtw_T)';
    for j=1:n
        Vt = reshape(V(j, :), [], dtw_T)';
        % dtw_c freaks out if not given a double
        dists(i, j) = dtw_c(double(Ut), double(Vt));
    end
end
G = exp(-dists / (dtw_T * 2 * dtw_gamma));
end