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
action_ids = db.act_names;
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
    % TODO: Is this actually aligned correctly?
    [starts, ends, actions] = db.seqactions(tmp2_id);
    if isempty(actions)
        fprintf('Skipping sequence with id=%i (no actions)\n', tmp2_id);
        continue
    end
    poses = db.poses{tmp2_id};
    poses = poses(good_joints, :, :);
    assert(length(db.PA) == size(poses, 1));
    action_vector = int16(zeros([1 length(poses)]));
    for act_i=1:length(starts)
        start = starts(act_i);
        finish = ends(act_i);
        finish = min([finish length(action_vector)]);
        if finish < start
            fprintf('Warning: sequence %i->%i goes over video bound at %i\n', ...
                start, ends(act_i), length(action_vector));
            break
        end
        action_vector(starts(act_i):ends(act_i)) = actions(act_i);
    end
    
    act_path = sprintf('/seqs/vid%i/actions', tmp2_id);
    h5create(dest_h5, act_path, length(action_vector), ...
        'DataType', 'int16', 'ChunkSize', length(action_vector), 'Deflate', 9);
    h5write(dest_h5, act_path, action_vector);
    
    pose_path = sprintf('/seqs/vid%i/poses', tmp2_id);
    h5create(dest_h5, pose_path, size(poses), ...
        'DataType', 'int16', 'ChunkSize', size(poses), 'Deflate', 9);
    h5write(dest_h5, pose_path, int16(poses));
end