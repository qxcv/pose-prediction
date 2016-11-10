function offsets = parent_offsets(joints, pa)
%PARENT_OFFSETS Get parent offsets for joints in a sequence (except root)
% joints is J*2*T array of joint [x,y] locations, pa is 1*J parents array.
offsets = joints - joints(pa, :, :); % that was shorter than I expected
end

