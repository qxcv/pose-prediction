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
            noffsets = length(obj.offsets);
            obj.models = cell([obj.njoints, 2, noffsets]);
            
            % Prepare training data
            X = nan([nseqs, ntaps * psize]);
            for seq=1:nseqs
                pose_data = seqs{seq}.poses(:, :, obj.taps);
                X(seq, :) = pose_data(:);
            end
            
            % Now train
            % This code is a little weird because I was using a parfor
            % (until I realised that it takes heaps of memory when you have
            % big objects to copy).
            offsets = obj.offsets;
            local_models = obj.models;
            train_point = @obj.train_point;
            % parfor joint=1:obj.njoints
            for joint=1:obj.njoints
                for coord=1:2
                    for off_i=1:noffsets
                        offset = offsets(off_i);
                        Y = nan([nseqs 1]);
                        for seq=1:nseqs
                            Y(seq) = seqs{seq}.poses(joint, coord, offset);
                        end
                        local_models{joint, coord, off_i} = ...
                            train_point(X, Y);
                    end
                end
            end
            obj.models = local_models;
        end
        
        function poses = predict(obj, seq)
            noffsets = length(obj.offsets);
            poses = nan([obj.njoints, 2, length(obj.offsets)]);

            % Inputs are the same for all points, as during training
            pose_data = seq(:, :, obj.taps);
            x = reshape(pose_data(:), 1, []);

            for joint=1:obj.njoints
                for coord=1:2
                    for off_i=1:noffsets
                        model = obj.models{joint, coord, off_i};
                        poses(joint, coord, off_i) = ...
                            obj.predict_point(model, x);
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
