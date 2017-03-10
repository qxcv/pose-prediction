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
        is_val
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
                fullfile(root, 'IkeaClipsDB_withactions.mat'), ...
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
            % Only grab relevant joints (rest are not reliable)
            good_joints = 1:8;
            seq_poses = obj.poses{dbi.video_id}(good_joints, :, :);
            info = struct('anno', dbi, 'poses', seq_poses, 'njoints', ...
                size(seq_poses, 1));
        end
        
%         function video_poses(obj, video_num)
%             % Get all poses for a single video
%             % TODO
%             match_name = sprintf('GOPR%04i', info.anno.video_id);
%             for i=1:length(obj.data)
%                 dbi = obj.data(i);
%             end
%         end
        
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
                obj.poses{i} = loadout(path, 'pose');
            end
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
            % Subjects 9, 11 and 13 are for testing. The rest are for
            % training. I'm going to arbitrarily train on [1, 2, 4, 5, 7, 8
            % 10, 12, 15].
            obj.subj_ids = [obj.data.person_idx];
            obj.is_train = ismember(obj.subj_ids, [1, 2, 4, 6, 7, 8, 10, 12, 15]);
            obj.is_test = ismember(obj.subj_ids, [9 11 13]);
            if isfield(obj.data, 'annot_test_poses')
                obj.has_anno = ~cellfun(@isempty, {obj.data.annot_test_poses});
            else
                warning('pp:AllAnnoMissing', ['Could not find ' ...
                    'annot_test_poses. Will not matter if you don''t ' ...
                    'care about the manually labelled poses.']);
                obj.has_anno = false([1 length(obj.data)]);
            end
            no_anno = find(obj.is_test & ~obj.has_anno);
            
            if ~isempty(no_anno)
                warning('pp:SomeAnnoMissing', ...
                    '%d/%d test items have no manual annotation', ...
                    length(no_anno), sum(obj.is_test));
                % display(no_anno);
            end
            obj.is_test = obj.is_test & obj.has_anno;
            obj.is_val = ~(obj.is_train | obj.is_test);
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

%         function load_approx_acts(obj)
%             % attempt to load approximate actions
%             act_path = './ikea_estimated_actions.h5';
%             if ~exist(act_path, 'file')
%                 warning('pp:NoEstimatedActions', ...
%                     'Could not find estimated actions at %s, skipping', ...
%                     act_path);
%                 return
%             end
%             
%             id_map = containers.Map('KeyType', 'char', 'ValueType', 'char');
%             % TODO: finish this method
%             for datidx=1:length(obj.data)
%                 continue
%                 % gopro_id = obj.internal_to_gopro_num(datidx);
%             end
%         end
    end
end
