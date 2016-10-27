classdef Predictor < handle
    % Interface class for predictors
    properties
        offsets
        njoints
    end
    
    methods
        function obj = Predictor(offsets, njoints)
            if nargin > 0
                obj.offsets = offsets;
                obj.njoints = njoints;
            else
                obj.offsets = 1;
                obj.njoints = 14;
            end
        end
    end
    
    methods(Abstract)
        predict(obj, seq);
    end
end
