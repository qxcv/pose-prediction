classdef IkeaDB < handle
    properties(Constant)
        % Parents array
        PA = [1 1 2 3 4 2 6 7];
    end
    
    properties
        data
        root
        poses
        is_train
        is_test
        is_val
        has_anno
        subj_ids
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
            
            % Subjects 9, 11 and 13 are for testing. The rest are for
            % training. I'm going to arbitrarily train on [1, 2, 4, 5, 7, 8
            % 10, 12, 15].
            obj.subj_ids = [obj.data.person_idx];
            obj.is_train = ismember(obj.subj_ids, [1, 2, 4, 6, 7, 8, 10, 12, 15]);
            obj.is_test = ismember(obj.subj_ids, [9 11 13]);
            obj.has_anno = ~cellfun(@isempty, {obj.data.annot_test_poses});
            no_anno = find(obj.is_test & ~obj.has_anno);
            
            if ~isempty(no_anno)
                warning('pp:MissingTestAnno', ...
                    '%d/%d test items have no manual annotation', ...
                    length(no_anno), sum(obj.is_test));
                % display(no_anno);
            end
            obj.is_test = obj.is_test & obj.has_anno;
            obj.is_val = ~(obj.is_train | obj.is_test);
        end
        
        function info = seqinfo(obj, datum_id)
            dbi = obj.data(datum_id);
            start = obj.pframe(dbi.frame_start);
            finish = obj.pframe(dbi.frame_end);
            offsets = nan([1 5]);
            pose_end = finish;
            for i=1:5
                pf = obj.pframe(dbi.(sprintf('predict_frame_%i', i)));
                offsets(i) = 1 - start + pf;
                pose_end = max([finish pf]);
            end
            ntrain = finish - start + 1;
            % Only grab relevant joints (rest are not reliable)
            good_joints = 1:8;
            seq_poses = obj.poses{dbi.video_id}(good_joints, :, start:pose_end);
            % XXX: Shouldn't perform this smoothing here :P
            seq_poses = smooth_poses(seq_poses);
            test_poses = dbi.annot_test_poses;
            if ~isempty(test_poses)
                test_poses = test_poses(good_joints, :, :);
            end
            info = struct('anno', dbi, 'poses', seq_poses, ...
                'offsets', offsets, 'ntrain', ntrain, ...
                'njoints', size(seq_poses, 1), ...
                'test_poses', test_poses);
            
            if obj.is_test(datum_id) && isempty(info.test_poses)
                warning('pp:NoTestAnno', 'No true pose for datum %i', ...
                    datum_id);
            end
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