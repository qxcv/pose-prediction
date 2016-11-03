classdef SVR < IndepMarkovLearner
    % Learn k-th order Markov chain with Support Vector Regression (SVR)
    methods
        function obj = SVR(varargin)
            obj@IndepMarkovLearner(varargin{:});
        end
        
        function model = train_point(~, X, Y)
            model = fitrlinear(X, Y);
        end
        
        function point = predict_point(~, model, X)
            point = model.predict(X);
        end
        
    end
end
