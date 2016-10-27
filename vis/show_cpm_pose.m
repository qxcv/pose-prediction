function show_cpm_pose(im_path, bb, det_pose)
%SHOW_CPM_POSE Plot a pose in format used by CPM

% Skeleton used by CPM:
% 1: Head
% 2: Neck
% 3: Right sholder
% 4: Right elbow
% 5: Right wrist
% 6: Left shoulder
% 7: Left elbow
% 8: Left wrist

im = imread(im_path); im = imcrop(im, bb);
imshow(im);
hold on;
det_pose(:,1) = det_pose(:,1)*size(im,2)/368;
det_pose(:,2) = det_pose(:,2)*size(im,1)/368;
line([det_pose(1,1), det_pose(2,1)], [det_pose(1,2), det_pose(2,2)], ...
    'color', 'r', 'linewidth',3);
line([det_pose(2,1), det_pose(3,1)], [det_pose(2,2), det_pose(3,2)], ...
    'color', 'g', 'linewidth',3);
line([det_pose(3,1), det_pose(4,1)], [det_pose(3,2), det_pose(4,2)], ...
    'color', 'b', 'linewidth',3);
line([det_pose(4,1), det_pose(5,1)], [det_pose(4,2), det_pose(5,2)], ...
    'color', 'm', 'linewidth',3);
line([det_pose(2,1), det_pose(6,1)], [det_pose(2,2), det_pose(6,2)], ...
    'color', 'g', 'linewidth',3);
line([det_pose(6,1), det_pose(7,1)], [det_pose(6,2), det_pose(7,2)], ...
    'color', 'c', 'linewidth',3);
line([det_pose(7,1), det_pose(8,1)], [det_pose(7,2), det_pose(8,2)], ...
    'color', 'w', 'linewidth',3);
plot(det_pose(:,1), det_pose(:,2), 'rx');

end
