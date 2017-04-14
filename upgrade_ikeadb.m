% Make new IkeaDB which contains both action and pose annotations

if ~exist('db_poses', 'var')
    fprintf('Loading dataset with poses\n');
    db_poses_l = load('IkeaDataset/IkeaClipsDB_withannotposes.mat');
    db_poses = db_poses_l.IkeaDB;
end

if ~exist('db_actions', 'var')
    fprintf('Loading dataset with actions\n');
    db_actions_l = load('IkeaDataset/IkeaClipsDB_withactions.mat');
    db_actions = db_actions_l.IkeaDB;
end

annot_struct = extract_annot_poses(db_poses);
% The _withactions and _withannotposes databases actually have different
% structures. _withannotposes is a big struct containing one entry for each
% 30s (?) slice of the dataset. _withactions, on the other hand, contains
% one entry for each video. I think that this code should work okay with
% it.
for i=1:length(db_actions)
    id = db_actions(i).video_id;
    ai = annot_struct([annot_struct.video_id] == id);
    assert(length(ai) <= 1);
    if isempty(ai)
        db_actions(i).test_pose_inds = [];
        db_actions(i).test_poses = [];
        db_actions(i).is_test = false;
    else
        db_actions(i).test_pose_inds = ai.test_pose_inds;
        db_actions(i).test_poses = ai.test_poses;
        db_actions(i).is_test = ~isempty(ai.test_poses);
    end
end

dest = 'IkeaClipsDB_with_annot_actions_poses.mat';
fprintf('Saving to %s\n', dest);
IkeaDB = db_actions;
save(dest, 'IkeaDB');