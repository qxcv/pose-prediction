% Makes a dataset for joint pose/action prediction. Gets postprocessed by
% Python NN training script (or something).
startup;

if ~exist('db', 'var') || ~isa(db, 'IkeaDB')
    fprintf('Loading IkeaDB anew\n');
    db = IkeaDB;
else
    fprintf('Using IkeaDB from memory\n');
end

dest_h5 = 'ikea_action_data.h5';
% head & everything down is safe; other stuff is not
good_joints = 1:8;

while exist(dest_h5, 'file')
    prompt = sprintf('File %s already exists. Delete it first? [y/n] ', ...
        dest_h5);
    answer = input(prompt, 's');
    answer = stripws(answer);
    if strcmpi(answer, 'y') || strcmpi(answer, 'yes')
        fprintf('Okay, deleting %s\n', dest_h5);
        delete(dest_h5);
        break;
    elseif strcmpi(answer, 'n') || strcmpi(answer, 'no')
        error('Can''t do anything without deleting file\n');
    else
        fprintf('Unrecognised answer, try "yes"/"y" or "no"/"n"\n');
    end
end

% Save action names
action_ids = [{'n/a'} db.act_names];
act_names_formatted = cell([1 length(action_ids)]);
for i=1:length(action_ids)
    act_names_formatted{i} = sprintf('"%s"', action_ids{i});
end
% As far as I can tell, this is the only way to write string data to a HDF5
% file in Matlab
action_string = ['[' strjoin(act_names_formatted, ', ') ']'];
action_bytes = unicode2native(action_string, 'UTF-8');
h5create(dest_h5, '/action_names', length(action_bytes), 'DataType', 'uint8');
h5write(dest_h5, '/action_names', action_bytes);

h5create(dest_h5, '/parents', length(db.PA), 'DataType', 'int8');
% Use Python indexing to store joint tree (so subtract 1)
h5write(dest_h5, '/parents', int8(db.PA-1));

h5create(dest_h5, '/num_actions', 1, 'DataType', 'int8');
h5write(dest_h5, '/num_actions', int8(length(action_ids)));

h5create(dest_h5, '/eval_condition_length', 1, 'DataType', 'int64');
h5write(dest_h5, '/eval_condition_length', int64(45));

h5create(dest_h5, '/eval_test_length', 1, 'DataType', 'int64');
h5write(dest_h5, '/eval_test_length', int64(75));

h5create(dest_h5, '/eval_seq_gap', 1, 'DataType', 'int64');
h5write(dest_h5, '/eval_seq_gap', int64(4));

% TODO: need to align the frame skip with actual test frames. They're quite
% rare, unfortunately.
h5create(dest_h5, '/frame_skip', 1, 'DataType', 'int64');
h5write(dest_h5, '/frame_skip', int64(3));

% Save action vectors (one action per time) and poses
for i=1:length(db.data)
    info = db.seqinfo(i);
    tmp2_id = info.tmp2_id;
    action_vector = uint8(db.seqactions_tmp2_id(tmp2_id));
    poses = db.poses{tmp2_id};
    poses = poses(good_joints, :, :);
    assert(length(db.PA) == size(poses, 1));
    assert(length(action_vector) == length(poses));

    act_path = sprintf('/seqs/vid%i/actions', tmp2_id);
    h5create(dest_h5, act_path, length(action_vector), ...
        'DataType', 'uint8', 'ChunkSize', length(action_vector));
    h5write(dest_h5, act_path, action_vector);

    pose_path = sprintf('/seqs/vid%i/poses', tmp2_id);
    h5create(dest_h5, pose_path, size(poses), ...
        'DataType', 'int16', 'ChunkSize', size(poses));
    h5write(dest_h5, pose_path, int16(poses));

    anno_poses = info.test_poses;
    if ~isempty(anno_poses)
        % use Python-style indexing for test pose indices
        anno_pose_inds = info.test_pose_inds - 1;
        anno_path = sprintf('/seqs/vid%i/annot_poses', tmp2_id);
        h5create(dest_h5, anno_path, size(anno_poses), ...
            'DataType', 'int16', 'ChunkSize', size(anno_poses));
        h5write(dest_h5, anno_path, int16(anno_poses));
        h5writeatt(dest_h5, anno_path, 'indices', anno_pose_inds);
    end
    
    train_path = sprintf('/seqs/vid%i/is_train', tmp2_id);
    h5create(dest_h5, train_path, 1, 'DataType', 'uint8');
    h5write(dest_h5, train_path, uint8(~info.is_test));
    
    scale_path = sprintf('/seqs/vid%i/scale', tmp2_id);
    scales = info.diam * ones([length(poses) 1]);
    h5create(dest_h5, scale_path, length(scales));
    h5write(dest_h5, scale_path, scales);
end