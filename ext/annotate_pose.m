% annotate ikea poses.
function annotate_pose()
global IkeaDB;
if isempty(IkeaDB)
    %IkeaDB = parload('/data/home/cherian/IkeaDataset/IkeaClipsDB_withpose.mat', 'IkeaDB');   
    IkeaDB = loadout('/data/home/cherian/IkeaDataset/IkeaClipsDB_withannotposes.mat', 'IkeaDB');
    %IkeaDB=loadout('IkeaClipsDB.mat', 'IkeaDB');
end
test_person_idx = [3 7 8]; cnt = 0;
for t=1:length(IkeaDB)    
    if ~isempty(intersect(IkeaDB(t).person_idx, test_person_idx))
        cnt = cnt + 1;
        fprintf('t=%d/%d cnt=%d/3400\n', t, length(IkeaDB), cnt);
        db = IkeaDB(t); 
        if exist(['/tmp/test_pose_' num2str(IkeaDB(t).video_id) '_' num2str(t) '.mat'], 'file')
            continue;
        end
        
        annotated_pose = annotate_test_poses(db);
        IkeaDB(t).annot_test_poses = annotated_pose;
        save(['/tmp/test_pose_' num2str(IkeaDB(t).video_id) '_' num2str(t) '.mat'], 'annotated_pose');
    end
end
end

function annoted_pose = annotate_test_poses(db)
    parts={'head', 'neck', 'lshol', 'lelb', 'lwri', 'rshol', 'relb', 'rwri'};    
    pose = db.annot_test_poses; % This is used when we alredy have annotated the poses and wanted to verify.
    if isempty(pose)
        fprintf('no poses for frame\n');
        pose = nan([14 2 5]);
        for t=1:5
            pose(:, :, t) = db.(['pred' num2str(t) '_pose']);
        end
    end
    for t=1:5
        test_im = imread(eval(['fullfile(db.clip_path, db.predict_frame_' num2str(t) ')']));
        test_im = imcrop(test_im, x2y2tohw(db.cropbox));
        test_im = imresize(test_im, [368,368], 'antialiasing', false);
        
        %pose = eval(['db.pred' num2str(t) '_pose']); 
        pose_data = pose(1:8,:,t);        
        
        % check and update pose_data.
        
        figure(1); clf(1);
        handles = show_pose(test_im, pose_data);    
        title(['video id ' num2str(db.video_id) ' clip id=' num2str(db.seq_idx)]);
        show_ui_and_annotate();       
        uiwait();
        
        pose(1:8,:,t) = pose_data;        
    end
    annoted_pose = pose;


    function show_ui_and_annotate()
        uicontrol('Style', 'pushbutton', 'String', 'Previous pose (cancel)',...
            'Position', [100 30 50 50],...
            'Callback', @use_previous_annotation);  
        uicontrol('Style', 'pushbutton', 'String', 'New pose (update)',...
            'Position', [200 30 50 50],...
            'Callback', @use_new);       

        function use_previous_annotation(~, ~)
            fprintf('reusing old pose\n');
            uiresume;
        end
    end
        
    function use_new(~, ~)
        fprintf('using new pose\n');
        for j=1:8
            h = handles{j};
            pose_data(j, :) = h.getPosition;
        end
        uiresume;
    end
end

function handles = show_pose(im, pose)
imshow(im);
% colors={'r', 'g','b','c','m','k','w', 'y'};
colors = jet(8);
handles = cell([1 8]);
hold on; 
for t=1:size(pose,1)
    h = impoint(gca, pose(t, 1), pose(t, 2));
    h.setColor(colors(t, :));
    handles{t} = h;
    drawnow;
end
end
