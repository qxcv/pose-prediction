classdef GPR < IndepMarkovLearner
    % Learn k-th order Markov chain with Gaussian Process Regression (GPR)
    methods
        function obj = GPR(varargin)
            obj@IndepMarkovLearner(varargin{:});
        end
        
        function model = train_point(~, X, Y)
            % Train on a few randomly chosen points to speed things up
            num_train = min([size(X, 1), 500]);
            p = randperm(num_train);
            Xp = X(p, :);
            Yp = Y(p);
            model = fitrgp(Xp, Yp, 'BasisFunction', 'pureQuadratic', ...
                'Standardize', 1);
        end
        
        function point = predict_point(~, model, X)
            point = predict(model, X);
        end
        
    end
end