function stripped = stripws(string)
%STRIPWS Strip whitespace from string
if ~verLessThan('matlab', '9.1')
    stripped = strip(string);
else
    stripped = regexprep(string, '(^\s+|\s+$)', '');
end
end