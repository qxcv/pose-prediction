function anno_struct = extract_annot_poses(db)
%EXTRACT_ANNOT_POSES Extract test poses from IkeaDB to an easier format.
db_ids = unique([db.video_id]);
anno_struct = struct('video_id', num2cell(db_ids));

for i=1:length(anno_struct)
    id = anno_struct(i).video_id;

    % get clip path, just because I can
    paths = unique({db([db.video_id] == id).clip_path});
    assert(length(paths) == 1);
    anno_struct(i).clip_path = paths{1};
    
    % get annotated poses
    id_mask = [db.video_id] == id;
    has_anno_mask = ~cellfun(@isempty, {db.annot_test_poses});
    with_anno = db(id_mask & has_anno_mask);
    
    if isempty(with_anno)
        test_poses = [];
        test_pose_inds = [];
    else
        ind_mat = [with_anno.pred1_frame_idx
                   with_anno.pred2_frame_idx
                   with_anno.pred3_frame_idx
                   with_anno.pred4_frame_idx
                   with_anno.pred5_frame_idx]';
        test_pose_inds = unique(ind_mat(:));
        test_poses = zeros([size(with_anno(1).poses(:, :, 1)), ...
                            length(test_pose_inds)]);
        for indi=1:length(test_pose_inds)
            ind = test_pose_inds(indi);
            flat = find(ind_mat == ind);
            [sample, subframe] = ind2sub(size(ind_mat), flat(1));
            test_poses(:, :, indi) = ...
                with_anno(sample).annot_test_poses(:, :, subframe);
        end
    end

    anno_struct(i).test_pose_inds = test_pose_inds;
    anno_struct(i).test_poses = test_poses;
end
end