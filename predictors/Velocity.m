classdef Velocity < Predictor
    % Uses velocity over previous frames to estimate future positions
    methods
        function obj = Velocity(varargin)
            obj@Predictor(varargin{:});
        end
        
        function poses = predict(obj, seq)
            poses = nan(obj.njoints, 2, length(obj.offsets));
            for i=1:length(obj.offsets)
                off = obj.offsets(i);
                % if off = length(seq) + i, then back = length(seq) - i
                back = 2 * length(seq) - off;
                poses(:, :, i) = 2 * seq(:, :, end) - seq(:, :, back);

                % This estimates velocity from the last pair instead
                % It does much worse :-)
                % end_pose = seq(:, :, end);
                % prev_pose = seq(:, :, end-1);
                % delta = off - length(seq);
                % poses(:, :, i) = end_pose  + delta * (end_pose - prev_pose);
                
                % TODO: What happens if I do a linear regression on the last
                % k poses?
            end
        end
    end
end
