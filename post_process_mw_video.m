frame_info(1,1).skeleton_ends_relpos = [NaN,NaN];
frame_info(1,1).head_position_x = NaN;
frame_info(1,1).head_position_x_um = NaN;
frame_info(1,1).head_position_y = NaN;
frame_info(1,1).head_position_y_um = NaN;
frame_info(1,1).tail_position_x = NaN;
frame_info(1,1).tail_position_x_um = NaN;
frame_info(1,1).tail_position_y = NaN;
frame_info(1,1).tail_position_y_um = NaN;
%bad_frames = bad_frames(1:size(frame_info,1),:);
for i=1:size(frame_info,2)
    [frame_info(:,i),bad_frames(:,i)] = post_process_video(squeeze(frame_info(:,i)),0);
end
% for i=1:size(frame_info,2)
%     ii = 1;
%     for j=1:size(frame_info,1)
%         if ~bad_frames(j,i)
%             theta_ensemble{i}(ii,:) = frame_info(j,i).theta;
%             ii = ii + 1;
%         end
%     end
% end
% for i=1:size(frame_info,2)
%     [ev,l] = eig(cov(theta_ensemble{i}));
%     l = flipud(diag(l));
%     ev = fliplr(ev);
%     l_pw(:,i) = l;
%     ev_pw(:,:,i) = ev;
% end