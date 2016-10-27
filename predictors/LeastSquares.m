classdef LeastSquares < Predictor
    % Learns a least squares model for predicting at each given offset
    properties
        taps
        weights
    end
    
    methods
        function obj = LeastSquares(offsets, joints, taps)
            if nargin == 0
                super_args = {};
            else
                super_args = {offsets, joints};
            end
            obj@Predictor(super_args{:});
            
            obj.taps = taps;
        end
        
        function train(obj, seqs)
            psize = obj.njoints * 2;
            nseqs = length(seqs);
            ntaps = length(obj.taps);
            noffsets = length(obj.offsets);
            X = nan([nseqs, psize, ntaps]);
            Y = nan([nseqs, psize, noffsets]);
            for i=1:length(seqs)
                seq_X = seqs{i}.poses(:, :, obj.taps);
                X(i, :, :) = reshape(seq_X, [], ntaps);
                seq_Y = seqs{i}.poses(:, :, obj.offsets);
                Y(i, :, :) = reshape(seq_Y, [], noffsets);
            end
            % Turned out to be easier to flatten afterwards
            Xf = reshape(X, [], ntaps);
            Yf = reshape(Y, [], noffsets);
            cvx_begin
                variable W(ntaps, noffsets);
                expression costs(noffsets);
                for o=1:noffsets
                    costs(o) = norm(Xf * W(:, o) - Yf(:, o));
                end
                minimize(sum(costs))
            cvx_end
            obj.weights = W;
        end
        
        function poses = predict(obj, seq)
            in_poses = seq(:, :, obj.taps);
            flat_out = reshape(in_poses, [], length(obj.taps)) * obj.weights;
            poses = reshape(flat_out, [obj.njoints, 2, length(obj.offsets)]);
        end
    end
end
