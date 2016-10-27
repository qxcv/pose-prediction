classdef IkeaDB < handle
    properties
        data
        root
        poses
        is_train
        is_test
        is_val
    end
    
    methods
        function obj = IkeaDB(root)
            if nargin == 0
                root = './';
            end
            obj.root = root;
            
            % Load clip database itself
            obj.data = loadout(fullfile(root, 'IkeaClipsDB'), 'IkeaDB');
            
            % Load all poses; obj.poses will have some gaps (e.g. clip 27
            % is missing)
            vid_ids = unique([obj.data.video_id]);
            nclips = max(vid_ids);
            obj.poses = cell([1, nclips]);
            for i=vid_ids
                fn = sprintf('pose_clip_%i.mat', i);
                path = fullfile(obj.root, 'poses', fn);
                obj.poses{i} = loadout(path, 'pose');
            end
            
            % Subjects 1 and 2 are for training, subject 3 is for
            % validation, subject 4 is for testing.
            subj_ids = [obj.data.person_idx];
            obj.is_train = ismember(subj_ids, [1 2]);
            obj.is_test = ismember(subj_ids, 3);
            obj.is_val = ~(obj.is_train | obj.is_test);
        end
        
        function info = seqinfo(obj, i)
            dbi = obj.data(i);
            start = obj.pframe(dbi.frame_start);
            finish = obj.pframe(dbi.frame_end);
            offsets = nan([1 5]);
            pose_end = finish;
            for i=1:5
                pf = obj.pframe(dbi.(sprintf('predict_frame_%i', i)));
                offsets(i) = 1 - start + pf;
                pose_end = max([finish pf]);
            end
            seq_poses = obj.poses{dbi.video_id}(:, :, start:pose_end);
            info = struct('anno', dbi, 'poses', seq_poses, ...
                'offsets', offsets, 'ntrain', finish - start + 1, ...
                'njoints', size(seq_poses, 1));
        end
        
        function show_pose(obj, seq_id, pose_id, det_pose)
            info = obj.seqinfo(seq_id);
            bb = info.anno.cropbox;
            frame_id = pose_id + obj.pframe(info.anno.frame_start) - 1;
            clip_dir = strrep(info.anno.clip_path, '/data/home/cherian/', '');
            % clip_dir = fullfile('IkeaDataset', 'Frames', info.anno.subset, info.anno.video_name);
            % clip_dir = fullfile('IkeaDataset', 'Frames', ...
            %     info.anno.subset, sprintf('GOPR%04i', info.anno.video_id));
            im_path = fullfile(clip_dir, sprintf('frame_%i.jpg', frame_id));
            if ~exist('det_pose', 'var')
                det_pose = info.poses(:, :, pose_id);
            end
            show_cpm_pose(im_path, x2y2tohw(bb), det_pose);
        end
    end
    
    methods(Access = private, Static)
        function out = pframe(fn)
            out = sscanf(fn, 'frame_%d.jpg');
            assert(isscalar(out));
        end
    end
end
