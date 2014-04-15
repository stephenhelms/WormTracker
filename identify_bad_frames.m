function [frame_info,bad_frames] = identify_bad_frames(frame_info)

% Identify bad frames
bad_frames = [frame_info.bad_frame] | [frame_info.bad_skeletonization];

% Reject abnormal skeleton lengths (<0.8 or >1.2 the median length)
med_length = median([frame_info(~bad_frames).length]);
bad_frames = bad_frames | ([frame_info.length]/med_length < 0.8) | ...
    ([frame_info.length]/med_length > 1.2);
disp(['Median worm length is ', num2str(med_length), ' um.']);
% Reject abnormal median body widths (<0.5 or >1.5 the median length)
for i=1:length(frame_info)
    if isempty(frame_info(i).width)
        frame_info(i).width = NaN;
    end
end
med_width = nanmedian([frame_info(~bad_frames).width]);
bad_frames = bad_frames | ([frame_info.width]/med_width < 0.5) | ...
    ([frame_info.width]/med_width > 1.5);
disp(['Median worm width is ', num2str(med_width), ' um.']);

idx_bad_frames = find(bad_frames);
disp(['Rejected ', num2str(length(idx_bad_frames)), ' (', ...
    num2str(length(idx_bad_frames)/length(frame_info)*100), '%) out of ', ...
    num2str(length(frame_info)), ' frames.']);

end