classdef Average < Predictor
    % Outputs mean pose across entire sequence
    methods
        function obj = Average(varargin)
            obj@Predictor(varargin{:});
        end
        
        function poses = predict(obj, seq)
            out = mean(seq, 3);
            poses = nan(obj.njoints, 2, length(obj.offsets));
            for i=1:length(obj.offsets)
                poses(:, :, i) = out;
            end
        end
    end
end
