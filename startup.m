if ~exist('started', 'var')
    addpath eval/ io/ predictors/ vis/ ext/
    get_libsvm;

    started = true;
end