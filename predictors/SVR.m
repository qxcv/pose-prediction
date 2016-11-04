classdef SVR < IndepMarkovLearner
    % Learn k-th order Markov chain with Support Vector Regression (SVR)
    methods
        function obj = SVR(varargin)
            obj@IndepMarkovLearner(varargin{:});
        end
        
        function model = train_point(~, X, Y)
            % Train on a few randomly chosen points to speed things up
            num_train = min([size(X, 1), 2000]);
            p = randperm(num_train);
            Xp = X(p, :);
            Yp = Y(p);
            model = fitrsvm(Xp, Yp, 'KernelFunction', 'rbf');
            % model = fitrsvm(Xp, Yp);
        end
        
        function point = predict_point(~, model, X)
            point = predict(model, X);
        end
        
    end
end