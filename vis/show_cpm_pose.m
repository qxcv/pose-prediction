function show_cpm_pose(maybe_path, bb, det_pose, image_relative)
%SHOW_CPM_POSE Plot a pose in format used by CPM.

% Unless image_relative is true, it will crop the image and assume that the
% pose was found by running a 368x368 CPM on the crop.

% Skeleton used by CPM:
% 1: Head
% 2: Neck
% 3: Right shoulder
% 4: Right elbow
% 5: Right wrist
% 6: Left shoulder
% 7: Left elbow
% 8: Left wrist

% maybe_path could be an image path or an image
if ischar(maybe_path)
    im = imread(maybe_path);
else
    im = maybe_path;
end
if ~isempty(bb)
    im = imcrop(im, bb);
end
if nargin < 4
    image_relative = false;
end
imagesc(im);
hold on;
if image_relative && ~isempty(bb)
    % oh crap, we cropped the frame, so now we need to shift the pose :/
    det_pose = single(det_pose) - bb(1:2);
elseif ~image_relative
    det_pose(:,1) = det_pose(:,1)*size(im,2)/368;
    det_pose(:,2) = det_pose(:,2)*size(im,1)/368;
end
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
