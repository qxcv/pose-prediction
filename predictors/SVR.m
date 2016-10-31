classdef SVR < IndepMarkovLearner
    % Learn k-th order Markov chain with Support Vector Regression (SVR)
    methods
        function obj = SVR(varargin)
            obj@IndepMarkovLearner(varargin{:});
        end
        
        function model = train_point(obj, X, Y)
            return fitrlinear
        end
        
        function point = predict_point(obj, model, X)
        end
        
    end
end
