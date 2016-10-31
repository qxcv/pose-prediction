classdef IndepMarkovLearner < Predictor
    % Interface for higher-order Markov models which predict joint-at-a-time
    properties
        taps;
        models;
    end
    
    methods
        function obj = IndepMarkovLearner(offsets, joints, taps)
            if nargin == 0
                super_args = {};
            else
                super_args = {offsets, joints};
            end
            obj@Predictor(super_args{:});
            
            obj.taps = taps;
        end
        
        function train(obj, seqs)
            psize = 2 * obj.njoints;
            nseqs = length(seqs);
            ntaps = length(obj.taps);
            obj.models = cell([njoints, 2, noffsets]);
            noffsets = length(obj.offsets);
            
            % Prepare training data
            X = nan([nseqs, ntaps * psize]);
            for seq=1:nseqs
                pose_data = seqs{seq}.poses(:, :, obj.taps);
                X(seq, :) = pose_data(:);
            end
            
            % Now train
            for joint=1:obj.njoints
                for coord=1:2
                    for off_i=1:noffsets
                        offset = obj.offsets(i);
                        Y = nan([nseqs 1]);
                        for seq=1:nseqs
                            Y(seq) = seqs{seq}.poses(joint, coord, offset);
                        end
                        obj.models{joint, coord, off_i} = ...
                            obj.train_point(X, Y);
                    end
                end
            end
        end
        
        function poses = predict(obj, poses)
            poses = nan([obj.njoints, 2, length(obj.offsets)]);

            % Inputs are the same for all points, as during training
            x = nan([ntaps * psize]);
            pose_data = poses(:, :, obj.taps);
            x(seq, :) = pose_data(:);

            for joint=1:obj.njoints
                for coord=1:2
                    for off_i=1:noffsets
                        model = obj.models{joint, coord, off_i};
                        poses(joint, coord, off_i) = ...
                            obj.train_point(model, x);
                    end
                end
            end
        end
    end
    
    methods(Abstract)
        train_point(obj, X, Y);
        predict_point(obj, model, X);
    end
end
