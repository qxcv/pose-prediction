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
% As far as I can tell, this is the only way to write string data to a HDF5
% file in Matlab
action_bytes = strcell2bytes(action_ids);
h5create(dest_h5, '/action_names', length(action_bytes), 'DataType', 'uint8');
h5write(dest_h5, '/action_names', action_bytes);

% Save joint names
joint_bytes = strcell2bytes(db.JOINT_NAMES);
h5create(dest_h5, '/joint_names', length(joint_bytes), 'DataType', 'uint8');
h5write(dest_h5, '/joint_names', joint_bytes);

% Identifier for downstream stuff to special-case on
h5create(dest_h5, '/dataset_name', length('ikea'), 'DataType', 'uint8');
h5write(dest_h5, '/dataset_name', uint8('ikea'));

h5create(dest_h5, '/parents', length(db.PA), 'DataType', 'int8');
% Use Python indexing to store joint tree (so subtract 1)
h5write(dest_h5, '/parents', int8(db.PA-1));
assert(length(db.PA) == length(db.JOINT_NAMES));

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

% handy for versioning
h5create(dest_h5, '/unix_time', 1, 'DataType', 'double');
h5write(dest_h5, '/unix_time', posixtime(datetime()));

pck_joints_str = '{"shoulders": [2, 5], "wrists": [4, 7], "head": [0], "neck": [1], "elbows": [3, 6]}';
h5create(dest_h5, '/pck_joints', length(pck_joints_str), 'DataType', 'uint8');
h5write(dest_h5, '/pck_joints', uint8(pck_joints_str));

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

    train_path = sprintf('/seqs/vid%i/is_train', tmp2_id);
    h5create(dest_h5, train_path, 1, 'DataType', 'uint8');
    h5write(dest_h5, train_path, uint8(~info.is_test));

    sid_path = sprintf('/seqs/vid%i/subject_id', tmp2_id);
    h5create(dest_h5, sid_path, 1, 'DataType', 'int64');
    h5write(dest_h5, sid_path, int64(info.subject_id));

    scale_path = sprintf('/seqs/vid%i/scale', tmp2_id);
    scales = info.diam * ones([length(poses) 1]);
    h5create(dest_h5, scale_path, length(scales));
    h5write(dest_h5, scale_path, scales);
end

function bytes = strcell2bytes(strcell)
% converts cell array of strings to UTF8-encoded JSON array
ents_formatted = cell([1 length(strcell)]);
for i=1:length(strcell)
    ents_formatted{i} = sprintf('"%s"', strcell{i});
end
string = ['[' strjoin(ents_formatted, ', ') ']'];
bytes = unicode2native(string, 'UTF-8');
end