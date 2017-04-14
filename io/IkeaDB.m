classdef IkeaDB < handle
    % Pieces of data which this needs to run:
    %
    % 1 IkeaDataset/IkeaClipsDB_withactions.mat: from IkeaDataset/ in
    %   /data/home/cherian.
    %
    % 2 IkeaDatset/tmp2/: this is actually a pose directory. Again, linking
    %   ./IkeaDataset to /data/home/cherian/IkeaDataset should grant this.
    %
    % 3 ./ikea_estimated_actions.h5: HDF5 file mapping GOPRXXXX IDs to
    %   collections of action probability vectors produced by an action
    %   recognition network. Scripts to produce this are currently somewhat
    %   ad-hoc, since they depend on a trained network in Anoop's home dir
    %   (which has to be converted to work with Keras, etc.).

    % Joint names (PC): head (1), base of neck (2), right shoulder (3), right
    % elbow (4), right wrist (5), left shoulder (6), left elbow (7), left wrist
    % (8)
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
        has_anno
        subj_ids
        act_names
        vid_clip_ids
        internal_to_gopro_num
        pose_root
    end
    
    methods
        function obj = IkeaDB(root)
            if nargin == 0
                root = './IkeaDataset';
            end
            obj.pose_root = './poses';
            obj.root = root;
            % Load clip database itself
            obj.data = loadout(...
                fullfile('./IkeaClipsDB_with_annot_actions_poses.mat'), ...
                'IkeaDB');
            
            obj.load_poses();
            obj.map_gopro_ids();
            obj.mark_train_test();
            obj.load_true_acts();
            % obj.load_approx_acts();
        end
        
        function actions = seqactions_tmp2_id(obj, tmp2_id)
            video_id = find([obj.data.video_id] == tmp2_id);
            assert(numel(video_id) == 1);
            actions = obj.data(video_id).activity_id;
        end
        
        function info = seqinfo(obj, datum_id)
            dbi = obj.data(datum_id);
            bb_hw = x2y2tohw(dbi.cropbox);
            diam = norm(bb_hw(3:4));
            % Only grab relevant joints (rest are not reliable)
            good_joints = 1:8;
            seq_poses = obj.poses{dbi.video_id}(good_joints, :, :);
            test_poses = obj.unmap_pose(dbi.test_poses, datum_id);
            if ~isempty(test_poses)
                test_poses = test_poses(good_joints, :, :);
            end
            info = struct('anno', dbi, 'poses', seq_poses, 'njoints', ...
                size(seq_poses, 1), 'is_test', dbi.is_test, ...
                'test_pose_inds', dbi.test_pose_inds, 'diam', diam, ...
                'test_poses', test_poses, 'tmp2_id', dbi.video_id);
        end
        
        function show_pose(obj, seq_id, pose_id, det_pose)
            info = obj.seqinfo(seq_id);
            bb = info.anno.cropbox;
            frame_id = pose_id;
            clip_dir = strrep(info.anno.clip_path, '/data/home/cherian/', '');
            im_path = fullfile(clip_dir, sprintf('frame_%i.jpg', frame_id));
            if ~exist('det_pose', 'var')
                det_pose = info.poses(:, :, pose_id);
            end
            show_cpm_pose(im_path, x2y2tohw(bb), det_pose, true);
        end
    end
    
    methods(Access = private, Static)
        function out = pframe(fn)
            out = sscanf(fn, 'frame_%d.jpg');
            assert(isscalar(out));
        end
    end
    
    methods(Access = private)
        % these methods are just called sequentially by the constructor
        % they've only been split out to make it clear what the different
        % bits of code do
        
        function load_poses(obj)
            % Load all poses; obj.poses will have some gaps (e.g. clip 27
            % is missing)
            vid_ids = unique([obj.data.video_id]);
            nclips = max(vid_ids);
            obj.poses = cell([1, nclips]);
            obj.vid_clip_ids = zeros([1, nclips]);
            for i=vid_ids
                fn = sprintf('pose_clip_%i.mat', i);
                path = fullfile(obj.pose_root, fn);
                pose_seq = loadout(path, 'pose');
                seq_id = find([obj.data.video_id] == i);
                assert(numel(seq_id) == 1);
                seq_id = seq_id(1);  % this just gets rid of a warning :)
                remapped = obj.unmap_pose(pose_seq, seq_id);
                obj.poses{i} = remapped;
            end
        end
        
        function real_poses = unmap_pose(obj, poses, seq_id)
            % Put pose into image coordinates, rather than crop-relative
            % coordinates
            if isempty(poses)
                real_poses = poses;
                return
            end
            assert(size(poses, 2) == 2 && ndims(poses) <= 3);
            bb = obj.data(seq_id).cropbox;
            hw_bb = x2y2tohw(bb);
            box_xmin = hw_bb(1);
            box_ymin = hw_bb(2);
            box_width = hw_bb(3);
            box_height = hw_bb(4);
            real_poses = poses;
            real_poses(:,1,:) = box_xmin + real_poses(:,1,:)*box_width/368;
            real_poses(:,2,:) = box_ymin + real_poses(:,2,:)*box_height/368;
            real_poses = int16(round(real_poses));
        end
        
        function map_gopro_ids(obj)
            vid_ids = unique([obj.data.video_id]);
            obj.internal_to_gopro_num = nan([1 max(vid_ids)]);
            for i=1:length(obj.data)
                % This is for matching up with actions (which are mapped
                % from video IDs), joining poses together, etc.
                parts = strsplit(obj.data(i).clip_path, '/');
                vid_id = sscanf(parts{end}, 'GOPR%d');
                assert(0 < vid_id && vid_id < 200);
                obj.vid_clip_ids(i) = vid_id;
                obj.internal_to_gopro_num(obj.data(i).video_id) = vid_id;
            end
        end
        
        function mark_train_test(obj)
            % Train/test split is defined by which videos have test poses
            % associated with them.
            obj.is_test = [obj.data.is_test];
            obj.is_train = ~obj.is_test;
            obj.has_anno = obj.is_test;
        end

        function load_true_acts(obj)
            % Load action annotations.
            all_acts = {};
            for i=1:length(obj.data)
                ids = obj.data(i).activity_id;
                names = obj.data(i).activity_labels;
                assert(length(ids) == length(names));
                for j=1:length(ids)
                    id = ids(j);
                    if id == 0
                        continue
                    end
                    name = names{j};
                    if length(all_acts) >= id && ~isempty(all_acts{id})
                        assert(strcmp(all_acts{id}, name));
                    elseif isempty(name)
                        all_acts{id} = 'n/a'; %#ok<AGROW>
                    else
                        all_acts{id} = name; %#ok<AGROW>
                    end
                end
            end
            obj.act_names = all_acts;
        end
    end
end
