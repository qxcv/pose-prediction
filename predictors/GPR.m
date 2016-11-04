classdef GPR < IndepMarkovLearner
    % Learn k-th order Markov chain with Gaussian Process Regression (GPR)
    methods
        function obj = GPR(varargin)
            obj@IndepMarkovLearner(varargin{:});
        end
        
        function model = train_point(~, X, Y)
            model = fitrgp(X, Y);
        end
        
        function point = predict_point(~, model, X)
            point = model.predict(X);
        end
        
    end
end
