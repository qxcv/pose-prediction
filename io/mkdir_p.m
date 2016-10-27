function mkdir_p(path)
%MKDIR_P Emulate mkdir -p
if ~exist(path, 'dir')
    mkdir(path);
end
end

