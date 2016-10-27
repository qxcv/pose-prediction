function pck_table = format_pcks(all_pcks, pck_thresholds, pck_joints)
% Turn a bunch of pcks produced by pck(), taken at a given set of
% thresholds, into a nice representation using the provided joint map.
joint_names = pck_joints.keys;
num_out_joints = length(joint_names);
accs = cell([1 num_out_joints]);
for joint_idx=1:num_out_joints
    these_joints = pck_joints(joint_names{joint_idx});
    these_pcks = cellfun(@(acc) mean(acc(these_joints)), all_pcks);
    accs{joint_idx} = these_pcks';
end
pck_table = table(pck_thresholds', accs{:}, ...
    'VariableNames', ['Threshold' joint_names]);
end
