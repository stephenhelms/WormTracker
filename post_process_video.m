function [frame_info,bad_frames] = post_process_video(frame_info,use_crossed_frames)

% Identify bad frames
bad_frames = [frame_info.bad_frame] | [frame_info.bad_skeletonization];
% Optional: Reject crossed frames
if ~use_crossed_frames
    bad_frames = bad_frames | [frame_info.crossed_frame];
end
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

% % Identify contiguous segments
% contiguous_segments = [1 length(frame_info)];
% i = 1;
% while i<=length(idx_bad_frames)
%     contiguous_segments(end) = idx_bad_frames(i)-1;
%     while ((i+1) < length(idx_bad_frames)) && ...
%             (idx_bad_frames(i+1) - idx_bad_frames(i) == 1)
%         i = i + 1;
%     end
%     if (idx_bad_frames(i)+1) < length(frame_info)
%         contiguous_segments = [contiguous_segments; ...
%             idx_bad_frames(i)+1,length(frame_info)];
%     end
%     i = i + 1;
% end
% % If all the last set of frames are bad, this is needed to remove that
% % segment
% if any(bad_frames(contiguous_segments(end,1):contiguous_segments(end,2)))
%     contiguous_segments = contiguous_segments(1:(size(contiguous_segments,1)-1),:);
% end
% % To improve chances of making a good choice, re-evaluate the head/tail
% % choice every n frames
% max_lumped_frames = 100;
% % And exclude segments with less than a certain number of frames (hard to
% % make a good choice then)
% min_lumped_frames = 10;
% split_contiguous_segments = [];
% for i=1:size(contiguous_segments,1)
%     n_lump = diff(contiguous_segments(i,:));
%     if n_lump < min_lumped_frames
%         bad_frames(contiguous_segments(i,1):contiguous_segments(i,2)) = 1;
%     elseif n_lump < max_lumped_frames
%         split_contiguous_segments = [split_contiguous_segments;contiguous_segments(i,:)];
%     else
%         for j=1:floor(n_lump/max_lumped_frames)
%             split_contiguous_segments = [split_contiguous_segments; ...
%                 contiguous_segments(i,1) + (j-1)*max_lumped_frames, ...
%                 contiguous_segments(i,1) + j*max_lumped_frames - 1];
%         end
%         if n_lump > floor(n_lump/max_lumped_frames)
%             j = floor(n_lump/max_lumped_frames);
%             split_contiguous_segments = [split_contiguous_segments; ...
%                 contiguous_segments(i,1) + j*max_lumped_frames, ...
%                 contiguous_segments(i,2)];
%         end
%     end
% end
% contiguous_segments = split_contiguous_segments;
% disp(['Found ', num2str(size(contiguous_segments,1)),' contiguous segments.']);
% 
% % Assign head/tail for each contiguous segment
% pixels_per_um = frame_info(contiguous_segments(1,1)).position_x / ...
%     frame_info(contiguous_segments(1,1)).position_x_um;
% for i=1:size(contiguous_segments,1)
%     frame_info = assign_head_tail2(frame_info,contiguous_segments(i,1), ...
%         contiguous_segments(i,2),pixels_per_um);
%     disp(['Assigned ', num2str(i), ' of ', num2str(size(contiguous_segments,1)),' segments.']);
% end

end