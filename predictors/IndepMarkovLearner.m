classdef IndepMarkovLearner < Predictor
    % Interface for higher-order Markov models which predict joint-at-a-time
    properties
        taps
        models
        pa
    end
    
    methods
        function obj = IndepMarkovLearner(offsets, njoints, taps, pa)
            if nargin == 0
                super_args = {};
            else
                super_args = {offsets, njoints};
            end
            obj@Predictor(super_args{:});
            
            obj.taps = taps;
            obj.pa = pa;
        end
        
        function train(obj, seqs)
            global dtw_T;
            nseqs = length(seqs);
            ntaps = length(obj.taps);
            dtw_T = ntaps-1;
            noffsets = length(obj.offsets);
            obj.models = cell([obj.njoints, 2, noffsets]);
            
            % Upfront copying (into all_X, all_Y, etc.) is to make it
            % easier for PCT to figure out what to send to workers.
            x_dim = ntaps * 2 * 2 - 2;
            all_X = nan([nseqs, x_dim, noffsets, obj.njoints]);
            all_Y = nan([nseqs, noffsets, 2, obj.njoints]);
            for seq=1:nseqs
                seq_poses = seqs{seq}.poses(:, :, :);
                for off_i=1:noffsets
                    % TODO: Normalisation. I think that this may still
                    % screw up if the taps have inconsistent gaps, since
                    % some offsets will be much larger than others.
                    offset = obj.offsets(off_i);
                    all_taps = [obj.taps offset];
                    tapped = seq_poses(:, :, all_taps);
                    [normed, ~] = norm_seq(tapped);
                    pa_offs = parent_offsets(tapped(:, :, 1:end-1), obj.pa);
                    for joint=1:obj.njoints
                        norm_X = normed(joint, :, 1:end-1);
                        j_offs = pa_offs(joint, :, :);
                        all_X(seq, :, off_i, joint) = [norm_X(:)', j_offs(:)'];
                        for coord=1:2
                            norm_Y = normed(joint, coord, end);
                            all_Y(seq, off_i, coord, joint) = norm_Y;
                        end
                    end
                end
            end
            
            local_models = obj.models;
            train_point = @obj.train_point;
            assert(~any(isnan(all_X(:))));
            assert(~any(isnan(all_Y(:))));
            parfor joint=1:obj.njoints
                for coord=1:2
                    for off_i=1:noffsets
                        fprintf('.');
                        Y = all_Y(:, off_i, coord, joint);
                        X = all_X(:, :, off_i, joint);
                        local_models{joint, coord, off_i} = ...
                            train_point(X, Y);
                    end
                end
            end
            fprintf('\n');
            obj.models = local_models;
        end
        
        function poses = predict(obj, seq)
            global dtw_T;
            noffsets = length(obj.offsets);
            ntaps = length(obj.taps);
            dtw_T = ntaps-1;
            poses = nan([obj.njoints, 2, length(obj.offsets)]);

            for off_i=1:noffsets
                % Reconstruct the entire sequence (or at least the
                % tapped/subsampled sequence) so that we can apply
                % normalisation and un-normalisation.
                [tapped, params] = norm_seq(seq(:, :, obj.taps));
                pa_offs = parent_offsets(seq(:, :, obj.taps), obj.pa);
                rec_seq = nan([obj.njoints, 2, ntaps]);
                rec_seq(:, :, 1:end-1) = tapped;
                pose = nan([obj.njoints, 2]);
                for joint=1:obj.njoints
                    pose_data = tapped(joint, :, :);
                    j_offs = pa_offs(joint, :, :);
                    x = [reshape(pose_data(:), 1, []), j_offs(:)'];
                    for coord=1:2
                        model = obj.models{joint, coord, off_i};
                        pose(joint, coord) = ...
                            obj.predict_point(model, x);
                    end
                end
                rec_seq(:, :, end) = pose;
                orig_seq = denorm_seq(rec_seq, params);
                poses(:, :, off_i) = orig_seq(:, :, end);
            end
        end
    end
    
    methods(Abstract)
        train_point(obj, X, Y);
        predict_point(obj, model, X);
    end
end
