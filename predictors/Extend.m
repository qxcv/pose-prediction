classdef Extend < Predictor
    % Simple predictor which tiles the last pose across all offsets
    methods
        function obj = Extend(varargin)
            obj@Predictor(varargin{:});
        end
        
        function poses = predict(obj, seq)
            poses = nan(obj.njoints, 2, length(obj.offsets));
            for i=1:length(obj.offsets)
                poses(:, :, i) = seq(:, :, end);
            end
        end
    end
end
