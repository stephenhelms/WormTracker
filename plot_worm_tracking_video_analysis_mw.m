function plot_worm_tracking_video_analysis(f,frame_info,index,extra_frame_info)
% f: Analysis figure handle
% frame_info: The analyzed frames
% index: The current frame
% extra_frame_info: The analysis object with all the data on the current
% (index) frame -- some of this is not stored in frame_info to save memory

N_worms = size(frame_info,2);
n_path_plot = 500; % The number of previous points to plot

figure(f);
clf;
for w=1:N_worms
    subplot(2,N_worms,w); % Subplot 1: Cleaned-up video frame with trajectory overlaid
    % Adjust the image brightness using the selected worm, if available
    low_high = [0 255];
    if frame_info(index,w).found_worm
        low_high = stretchlim(frame_info(index,w).im_worm)*255;
    end
    imshow(extra_frame_info(1,w).im_bgrem,low_high,'InitialMagnification','fit'); hold on;
    if w==1
        title(['Frame ',num2str(index), ' of ', num2str(size(frame_info,1)), ...
            ' (t=', num2str(frame_info(index,w).t,'%10.2f'),'s)']);
    end
    % Plot trajectory, plotting the previous n_path_plot/2 frames and then
    % n_path_plot/2 spread evenly since the start of the video
    if ~isempty([frame_info(1:index,w).position_x])
        t = [frame_info(1:index,w).t];
        x = [frame_info(1:index,w).position_x];
        y = [frame_info(1:index,w).position_y];

        % If we have a lot of points, show the last 500 in detail
        % and evenly sample the rest
        if index > n_path_plot
            idx = [round(linspace(1,index-n_path_plot/2,n_path_plot/2)),...
                (index-n_path_plot+1):index];
            t = t(idx);
            x = x(idx);
            y = y(idx);
        end
        scatter(x,y,4,'b','Marker','.');
    end

    % If there was a detected worm in the frame
    if frame_info(index,w).found_worm
        subplot(2,N_worms,N_worms+w); % Subplot 2: Close-up of worm with boundary and skeletonization
        imshow(imadjust(frame_info(index,w).im_worm),[0 255],'InitialMagnification','fit');
        hold on;
        % Plot boundary
        if ~isempty(frame_info(index,w).outer_boundary)
            plot([frame_info(index,w).outer_boundary(:,2);frame_info(index,w).outer_boundary(1,2)], ...
                [frame_info(index,w).outer_boundary(:,1);frame_info(index,w).outer_boundary(1,1)],'b-');
        end
        % Plot skeleton
        if ~frame_info(index,w).bad_skeletonization & ~isempty(frame_info(index,w).skeleton)
             % Color by angle
             angle_colormap = spring(64);
             n_skeleton_points = size(frame_info(index,w).skeleton,1);
             n_theta = length(frame_info(index,w).theta);
             scatter(frame_info(index,w).skeleton(:,2),frame_info(index,w).skeleton(:,1),8, ...
                    angle_colormap( ...
                    round(interp1(1:n_theta,63*((3/2*pi + frame_info(index,w).theta)/(3*pi))+2,linspace(1,n_theta,n_skeleton_points))), ...
                    :),'filled','Marker','o');
             % Plot centroid and body midpoint
             bb = frame_info(index,w).stats.BoundingBox;
             scatter(frame_info(index,w).x_mid - bb(:,1), frame_info(index,w).y_mid - bb(:,2),12,'b','filled','Marker','s','MarkerEdgeColor','w');
             scatter(frame_info(index,w).position_x - bb(:,1), frame_info(index,w).position_y - bb(:,2),12,'b','filled','Marker','o','MarkerEdgeColor','w');
        end
    end
end

end