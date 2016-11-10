function orig = denorm_seq(normed, params)
%DENORM_SEQ Undo normalisation of a sequence
assert(ndims(normed) == 3);
assert(size(normed, 2) == 2);
assert(ismatrix(params.init) && size(params.init, 1) == size(normed, 1) ...
    && size(params.init, 2) == 2);
cs = cumsum(normed, 3);
trunc_orig = bsxfun(@plus, cs, params.init);
orig = cat(3, params.init, trunc_orig);
end