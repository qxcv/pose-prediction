function plot_joints(im, joints)
%PLOT_JOINTS Summary of this function goes here
%   Detailed explanation goes here
assert(ismatrix(joints) && size(joints, 2) == 2);
imshow(im);
hold on;
colors = get(gca, 'ColorOrder');
for j=1:length(joints)
    color = colors(mod(j - 1, length(colors)) + 1, :);
    xy = joints(j, :);
    x = xy(1);
    y = xy(2);
    plot(x, y, 'X', 'MarkerSize', 20, ...
        'MarkerEdgeColor', color);
    text(x, y, sprintf('%d', j), ...
        'Color', color, ...
        'FontSize', 20);
end
hold off;
end