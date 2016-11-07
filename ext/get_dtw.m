function get_dtw(ext_dir)
%GET_DTW Download DTW code from file exchange

if ~exist(ext_dir, 'var')
    ext_dir = 'ext';
end

url = 'https://au.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/43156/versions/5/download/zip';
dest_file = fullfile(ext_dir, 'dtw.zip');
dest_dir = fullfile(ext_dir, 'dtw');
ap_dir = fullfile(dest_dir, 'dynamic_time_warping_v2.1');
mex_fn = fullfile(ap_dir, ['dtw_c.' mexext]);
mex_src = fullfile(ap_dir, 'dtw_c.c');

if ~exist(mex_fn, 'file')
    if ~exist(dest_file, 'file')
        websave(dest_file, url);
    end
    unzip(dest_file, dest_dir);
    mex('-output', mex_fn, 'CFLAGS=-std=c11 -fPIC', mex_src);
end

addpath(ap_dir);
end

