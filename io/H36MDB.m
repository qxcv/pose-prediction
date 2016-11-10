classdef H36MDB < handle
    properties (Constant)
        REAL_SUBJECTS = [1 5 6 7 8 9 11];
        % Try to emulate CPM joint structure
        KEEP_JOINTS = [16 14 26:28 18:20];
        % Parents array
        PA = [1 1 2 3 4 2 6 7];
        TEST_SUBJECT = 5;
        VAL_SUBJECT = 9;
    end
    
    properties
        raw
        root
        is_train
        is_test
        is_val
        has_anno
        subj_ids
    end
    
    methods
        function obj = H36MDB(root)
            if nargin == 0
                root = './h3.6m';
            end
            obj.root = root;
            
            obj.raw = [];
            
            for subj_index=1:length(obj.REAL_SUBJECTS)
                fprintf('Loading subject %i/%i\n', subj_index, ...
                    length(obj.REAL_SUBJECTS));
                
                real_subj = obj.REAL_SUBJECTS(subj_index);
                pose_dir = fullfile(obj.root, sprintf('S%i', real_subj), ...
                    'MyPoseFeatures', 'D2_Positions');
                video_dir = fullfile(obj.root, sprintf('S%i', real_subj), ...
                    'Videos');

                % Iterate over each scenario
                pose_dir_listing = dir(pose_dir);
                pose_fns = {pose_dir_listing(3:end).name};
                
                % copy obj.* attributes so that parfor runs smoothly
                parse_fn = @obj.parse_fn;
                seq_data = @obj.seq_data;
                
                for pose_fn_idx=1:length(pose_fns)
                    pose_fn = pose_fns{pose_fn_idx};

                    [action, cam] = parse_fn(pose_fn);

                    vid_path = fullfile(video_dir, ...
                        sprintf('%s.%i.mp4.ogv', action, cam));
                    if strfind(vid_path, ...
                            'S11/Videos/Directions.54138969.mp4.ogv')
                        continue
                    end

                    [poses, frame_times] = seq_data(real_subj, action, cam);
                    poses = poses(obj.KEEP_JOINTS, :, :);
                    
                    seqs = obj.chop_seq(real_subj, action, cam, ...
                        frame_times, poses);
                    obj.raw = [obj.raw, seqs];
                end
            end
            
            obj.subj_ids = [obj.raw.subject];
            obj.is_test = ismember(obj.subj_ids, obj.TEST_SUBJECT);
            obj.is_val = ismember(obj.subj_ids, obj.VAL_SUBJECT);
            obj.is_train = ~(obj.is_test | obj.is_val);
            obj.has_anno = true(size(obj.subj_ids));
        end
        
        function info = seqinfo(obj, datum_id)
            dbi = obj.raw(datum_id);
            offsets = 330:30:450;
            ntrain = 300;
            seq_poses = dbi.poses;
            test_poses = seq_poses(:, :, offsets);
            info = struct('raw', dbi, 'poses', seq_poses, ...
                'offsets', offsets, 'ntrain', ntrain, ...
                'njoints', size(seq_poses, 1), ...
                'test_poses', test_poses);
        end
        
        function show_pose(obj, seq_id, pose_id, det_pose)
            info = obj.seqinfo(seq_id);
            video_path = sprintf(...
                fullfile(obj.root, 'S%i', 'Videos', '%s.%i.mp4.ogv'), ...
                         info.raw.subject, info.raw.action, info.raw.camera);
            im = obj.video_imread(video_path, info.raw.frame_time(pose_id));
            if ~exist('det_pose', 'var')
                det_pose = info.poses(:, :, pose_id);
            end
            show_cpm_pose(im, [], det_pose);
        end
        
        function [poses, frame_times] = seq_data(obj, subject, action, cam)
            video_path = sprintf(...
                fullfile(obj.root, 'S%i', 'Videos', '%s.%i.mp4.ogv'), ...
                         subject, action, cam);
            pose_path = sprintf(...
                fullfile(obj.root, 'S%i', 'MyPoseFeatures', ...
                         'D2_Positions', '%s.%i.cdf'), subject, action, ...
                         cam);
            poses = obj.load_cdf_poses(pose_path);
            frame_times = obj.video_frametimes(video_path);
            % Sometimes there are more frames than poses for some reason.
            % XXX: Investigate the above. Is it just because they are aligning poses
            % for the three cameras? If so, why do some cameras have more frames than
            % others for the same subject and action sequence?
            assert(length(frame_times) >= size(poses, 3));
            if length(frame_times) ~= size(poses, 3)
                min_len = min(length(frame_times), size(poses, 3));
                frame_times = frame_times(1:min_len);
                poses = poses(:, :, 1:min_len);
            end
        end
    end
    
    methods(Static)
        function [action, cam] = parse_fn(fn)
            % Parse filename like 'Eating 2.60457274.mp4' to get action and camera name
            tokens = regexp(fn, '([\w ])+.(\d+)\.(mp4(\.ogv)?|cdf)', 'tokens');
            assert(length(tokens) == 1 && length(tokens{1}) == 3);
            action = tokens{1}{1};
            cam = str2double(tokens{1}{2});
        end

        function poses = load_cdf_poses(cdf_path)
            % Load poses from NetCDF file. Return as a J*2*T matrix.
            var_cell = cdfread(cdf_path, 'Variable', {'Pose'});
            unshaped = var_cell{1};
            new_size = [size(unshaped, 1), 2, size(unshaped, 2) / 2];
            poses = permute(reshape(unshaped, new_size), [3 2 1]);
        end
        
        function times = video_frametimes(video_path)
            % Return time offsets for each frame in an MP4 file
            reader = VideoReader(video_path);
            frame_time = 1/reader.FrameRate;
            % Video frame range is [0, reader.Duration). Of course, the Matlab docs
            % don't *say* that the range is exclusive on one side, because why bother
            % being clear on minor issues like whether the last frame of your video
            % even exists?
            end_time = reader.Duration - frame_time / 2;
            times = 0:frame_time:end_time;
        end
        
        function frame = video_imread(video_path, frame_time)
            % Read single frame from video
            assert(frame_time >= 0 ...
                && (isa(frame_time, 'single') || isa(frame_time, 'double')));

            % VideoReader expects a double for times
            ft_double = double(frame_time);
            reader = VideoReader(video_path, 'CurrentTime', ft_double);
            assert(reader.hasFrame(), 'Time %f out of range for %s', ft_double, video_path);
            frame = reader.readFrame();
        end
        
        function seqs = chop_seq(subj, action, cam, times, poses)
            % Chop a video into disjoint sequences of 450 poses
            nframes = length(times);
            num_seqs = floor(nframes / 450);
            chopped_ids = cell([1 num_seqs]);
            chopped_times = cell([1 num_seqs]);
            chopped_poses = cell([1 num_seqs]);
            for sn=1:num_seqs
                start = 450 * (sn - 1) + 1;
                finish = start + 450 - 1;
                chopped_ids{sn} = start:finish;
                chopped_times{sn} = times(start:finish);
                chopped_poses{sn} = poses(:, :, start:finish);
            end
            seqs = struct('frame_id', chopped_ids, ...
                'frame_time', chopped_times, 'poses', chopped_poses, ...
                'action', action, 'subject', subj, 'camera', cam);
        end
    end
end
