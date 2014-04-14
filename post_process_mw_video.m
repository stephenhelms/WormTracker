clear bad_frames traj
for i=1:size(frame_info,2)
    try
        [frame_info(:,i),bad_frames(:,i)] = identify_bad_frames(frame_info(:,i));
        [frame_info(:,i),traj(i)] = post_process_frames(frame_info(:,i),bad_frames(:,i),...
            pixels_per_um,frameRate);
    catch
        disp(['Problem analyzing worm ', num2str(i)]);
    end
end