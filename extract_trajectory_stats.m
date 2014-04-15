function [frame_info,analysis] = extract_trajectory_stats(frame_info,bad_frames)

fps = 11.5;
n_smooth = 1*round((fps*1)/2);

[N_frames,N_worms] = size(frame_info);

for w=1:N_worms
    analysis.worm(w).t = [frame_info(:,w).t];
    
    % Position
    analysis.worm(w).x = [frame_info(:,w).position_x_um]';
    analysis.worm(w).y = [frame_info(:,w).position_y_um]';
    analysis.worm(w).xm = [frame_info(:,w).x_mid_um]';
    analysis.worm(w).ym = [frame_info(:,w).y_mid_um]';
    
    % Velocity
    analysis.worm(w).vdir = NaN*ones(N_frames,1);
    analysis.worm(w).v = NaN*ones(N_frames,1);
    analysis.worm(w).xy_pred = NaN*ones(N_frames,2);
    % Smooth velocity by local linear interpolation
    for i=ceil(n_smooth/2+1):floor(N_frames-n_smooth/2)
        p = [analysis.worm(w).x((i-n_smooth/2):(i+n_smooth/2)),analysis.worm(w).y((i-n_smooth/2):(i+n_smooth/2))];
        % Local linear fit (p-<p>)=A(t-<t>)
        A = (p-repmat(mean(p),size(p,1),1))' / (analysis.worm(w).t((i-n_smooth/2):(i+n_smooth/2))-repmat(analysis.worm(w).t(i),n_smooth+1,1)');
        % Remove noise by replacing position with the local linear estimate
        analysis.worm(w).xy_pred(i,:) = mean(p);
        % Velocity magnitude and direction come from the slope of the fit
        analysis.worm(w).vdir(i) = atan2(A(2),A(1));
        analysis.worm(w).v(i) = sqrt(A(1)^2 + A(2)^2);
    end
    clear xy_pred p A
    
    % Unsmoothed velocity
    analysis.worm(w).v_nf = [NaN;sqrt(sum([diff(analysis.worm(w).y),diff(analysis.worm(w).x)].^2,2))/(1/11.5)];
    analysis.worm(w).vdir_nf = [NaN;atan2(diff(analysis.worm(w).y),diff(analysis.worm(w).x))];
    
    % Moving average filtered velocity
    analysis.worm(w).vdir_ma = NaN*ones(N_frames,1);
    for i=ceil(n_smooth/2+1):floor(N_frames-n_smooth/2)
        analysis.worm(w).vdir_ma(i) = nanmean(analysis.worm(w).vdir_nf(i-n_smooth/2):(i+n_smooth/2));
    end
    
    % Path curvature
    % Velocity
    analysis.worm(w).curvature = NaN*ones(N_frames,1);
    for i=ceil(n_smooth/2+1):floor(N_frames-n_smooth)
        analysis.worm(w).curvature(i) = analysis.worm(w).vdir(i+n_smooth/2)-analysis.worm(w).vdir(i);
    end
    analysis.worm(w).curvature = atannorm(analysis.worm(w).curvature);
    
    % Smooth velocity by local linear interpolation
    for i=ceil(n_smooth/2+1):floor(N_frames-n_smooth/2)
        p = [analysis.worm(w).x((i-n_smooth/2):(i+n_smooth/2)),analysis.worm(w).y((i-n_smooth/2):(i+n_smooth/2))];
        % Local linear fit (p-<p>)=A(t-<t>)
        A = (p-repmat(mean(p),size(p,1),1))' / (analysis.worm(w).t((i-n_smooth/2):(i+n_smooth/2))-repmat(analysis.worm(w).t(i),n_smooth+1,1)');
        % Remove noise by replacing position with the local linear estimate
        analysis.worm(w).xy_pred(i,:) = mean(p);
        % Velocity magnitude and direction come from the slope of the fit
        analysis.worm(w).vdir(i) = atan2(A(2),A(1));
        analysis.worm(w).v(i) = sqrt(A(1)^2 + A(2)^2);
    end
    
    % Calculate interframe skeleton distance to make head/tail assignment
    % consistent
    analysis.worm(w).interframe_d = NaN*ones(N_frames,2);
    analysis.worm(w).n_from_last_good = NaN*ones(N_frames,1);
    N_angles = [];
    idx_good = find(~bad_frames(:,w));
    for i=1:length(idx_good)
        if ~isempty(frame_info(idx_good(i),w).theta)
            N_angles = length(frame_info(idx_good(i),w).theta);
            break;
        end
    end
    s = (0:N_angles)/N_angles;
    idx_first_good = find(~bad_frames(:,w),1);
    if isempty(idx_first_good)
        continue;
    end
    pixels_per_um = frame_info(idx_first_good,w).position_x/frame_info(idx_first_good,w).position_x_um;
    for i=2:N_frames
        if ~bad_frames(i,w)
            ip = find(~bad_frames(1:(i-1),w),1,'last');
            if isempty(ip)
                continue;
            end
            analysis.worm(w).n_from_last_good(i) = i - ip;
            np = size(frame_info(ip,w).skeleton,1);
            if np<N_angles || any(any(isnan(frame_info(ip,w).skeleton)))
                continue;
            end
            xy_p = spline((0:(np-1))/(np-1),(frame_info(ip,w).skeleton/pixels_per_um - ...
                repmat([frame_info(ip,w).x_mid_um-frame_info(ip,w).stats.BoundingBox(1)/pixels_per_um, ...
                frame_info(ip,w).y_mid_um-frame_info(ip,w).stats.BoundingBox(2)/pixels_per_um],np,1))',s)';
            ni = size(frame_info(i,w).skeleton,1);
            xy = spline((0:(ni-1))/(ni-1),(frame_info(i,w).skeleton/pixels_per_um - ...
                repmat([frame_info(i,w).x_mid_um-frame_info(i,w).stats.BoundingBox(1)/pixels_per_um, ...
                frame_info(i,w).y_mid_um-frame_info(i,w).stats.BoundingBox(2)/pixels_per_um],ni,1))',s)';
            analysis.worm(w).interframe_d(i,1) = sum(sqrt(sum((xy - xy_p).^2,2)))/length(s);
            analysis.worm(w).interframe_d(i,2) = sum(sqrt(sum((flipud(xy) - xy_p).^2,2)))/length(s);
        end
    end
    
    % Flip inverted skeletons and angle measurements
    last_flipped = false;
    for i=2:N_frames
        if isnan(analysis.worm(w).interframe_d(i,1))
            continue;
        end
        if analysis.worm(w).interframe_d(i,2) < analysis.worm(w).interframe_d(i,1)
            last_flipped = ~last_flipped;
            analysis.worm(w).interframe_d(i,:) = fliplr(analysis.worm(w).interframe_d(i,:));
        end
        if last_flipped
            frame_info(i,w).skeleton = flipud(frame_info(i,w).skeleton);
            frame_info(i,w).theta = fliplr(frame_info(i,w).theta);
        end
    end
    
    % Break video into segments with matched skeletons
    max_n_missing = 10;
    max_d = 10/pixels_per_um;
    max_segment_frames = 500;
    min_segment_size = 150;
    ii = 1;
    n_segment = 1;
    analysis.worm(w).segments = zeros(round(N_frames/max_segment_frames),2);
    while ii < N_frames
        analysis.worm(w).segments(n_segment,1) = ii;
        ii = ii + 1;
        % Continue segment until >max_n_missing consecutive bad frames are
        % found, or >max_segment_frames are collected
        n_missing = 0;
        last_missing = false;
        while ii < N_frames && ...
                (ii - analysis.worm(w).segments(n_segment,1)) < max_segment_frames && ...
                (isnan(analysis.worm(w).interframe_d(ii,1)) || min(analysis.worm(w).interframe_d(ii,:))<max_d)
            if bad_frames(ii,w)
                n_missing = n_missing + 1;
                last_missing = true;
                if n_missing > max_n_missing
                    ii = ii + 1;
                    break;
                end
            else
                n_missing = 0;
                last_missing = false;
            end
            ii = ii + 1;
        end
        % Mark end of segment
        analysis.worm(w).segments(n_segment,2) = ii - 1;
        n_segment = n_segment + 1;
    end
    keep_segment = false(size(analysis.worm(w).segments,1),1);
    for n=1:size(analysis.worm(w).segments,1)
        keep_segment(n) = sum(~isnan(analysis.worm(w).interframe_d(analysis.worm(w).segments(n,1):analysis.worm(w).segments(n,2),1))) > min_segment_size;
    end
    analysis.worm(w).segments = analysis.worm(w).segments(keep_segment,:);
    clear keep_segment n_segment last_missing n_missing ii i n
    
    % Calculate skeleton end orientation
    analysis.worm(w).ht_orientation = NaN*ones(N_frames,2);
    for i=2:N_frames
        if ~bad_frames(i,w)
            p = fliplr((frame_info(i,w).skeleton([1,size(frame_info(i,w).skeleton,1)],:) + ...
                repmat(frame_info(i,w).stats.BoundingBox([2,1]),2,1))/pixels_per_um);
            analysis.worm(w).ht_orientation(i,:) = atan2(p(:,2)-analysis.worm(w).y(i),p(:,1)-analysis.worm(w).x(i))';
        end
    end
    
    % Calculate skeleton end velocity
    analysis.worm(w).v_ends_rdir = NaN*ones(N_frames,2);
    analysis.worm(w).v_ends = NaN*ones(N_frames,2);
    analysis.worm(w).xy_ends_pred = NaN*ones(N_frames,4);
    for i=ceil(n_smooth/2+1):floor(N_frames-n_smooth/2)
        p = NaN*ones(n_smooth+1,4);
        for j=round(i-n_smooth/2):round(i+n_smooth/2)
            if ~bad_frames(j,w)
                p(j-(i-n_smooth/2)+1,:) = ([frame_info(j,w).skeleton(1,:), ...
                    frame_info(j,w).skeleton(size(frame_info(j,w).skeleton,1),:)] ...
                    + repmat(frame_info(j,w).stats.BoundingBox([2,1]),1,2))/pixels_per_um;
            end
        end
        A = (p-repmat(nanmean(p),size(p,1),1))' / (analysis.worm(w).t(round(i-n_smooth/2):round(i+n_smooth/2))-repmat(analysis.worm(w).t(i),n_smooth+1,1)');
        analysis.worm(w).xy_ends_pred(i,:) = nanmean(p);
        analysis.worm(w).v_ends_rdir(i,:) = [atan2(A(2),A(1)),atan2(A(4),A(3))] - analysis.worm(w).vdir(i);
        analysis.worm(w).v_ends(i,:) = [sqrt(A(1)^2 + A(2)^2),sqrt(A(3)^2 + A(4)^2)];
    end
    clear xy_pred A p i j
    
    % Assign head
    [frame_info,analysis] = assign_head_tail3(frame_info,analysis,bad_frames,w);
end
analysis = analyze_posture(frame_info,analysis,false);

end