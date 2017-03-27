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

% Save action vectors (one action per time) and poses
tmp2_ids = unique([db.data.video_id]);
for tmp2_id=tmp2_ids
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
end