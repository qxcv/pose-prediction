function varargout = loadout(filename, varargin)
%LOADOUT Try [a, b, c] = loadout('file.mat', 'name1', 'name2', 'name3');
l = load(filename);
assert(nargout == nargin-1)
for i=1:nargout
    varargout{i} = l.(varargin{i}); %#ok<AGROW>
end
end

