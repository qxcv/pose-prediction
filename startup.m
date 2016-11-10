if ~exist('started', 'var')
    addpath eval/ io/ predictors/ vis/ ext/ kernels/;
    get_libsvm;
    get_dtw;

    started = true;
end