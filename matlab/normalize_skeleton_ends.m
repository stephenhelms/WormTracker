function [frames,interframe_d] = normalize_skeleton_ends(frames,bad_frames,pixels_per_um)

N_frames = length(frames);

interframe_d = NaN*ones(N_frames,2);
n_from_last_good = NaN*ones(N_frames,1);

% Determine number of body angles that were measured
N_angles = [];
idx_good = find(~bad_frames);
for i=1:length(idx_good)
    if ~isempty(frames(idx_good(i)).theta)
        N_angles = length(frames(idx_good(i)).theta);
        break;
    end
end
s = (0:N_angles)/N_angles;
idx_first_good = find(~bad_frames,1);
if isempty(idx_first_good)
    throw(MException('normalize_skeleton_ends:No postural data'));
end

% Calculate an error metric for both possible orientations:
% The sum of the distance between matched skeleton points
for i=2:N_frames
    if ~bad_frames(i)
        ip = find(~bad_frames(1:(i-1)),1,'last');
        if isempty(ip)
            continue;
        end
        n_from_last_good(i) = i - ip;
        np = size(frames(ip).skeleton,1);
        if np<N_angles || any(any(isnan(frames(ip).skeleton)))
            continue;
        end
        xy_p = spline((0:(np-1))/(np-1),(frames(ip).skeleton/pixels_per_um - ...
            repmat([frames(ip).x_mid_um-frames(ip).stats.BoundingBox(1)/pixels_per_um, ...
            frames(ip).y_mid_um-frames(ip).stats.BoundingBox(2)/pixels_per_um],np,1))',s)';
        ni = size(frames(i).skeleton,1);
        xy = spline((0:(ni-1))/(ni-1),(frames(i).skeleton/pixels_per_um - ...
            repmat([frames(i).x_mid_um-frames(i).stats.BoundingBox(1)/pixels_per_um, ...
            frames(i).y_mid_um-frames(i).stats.BoundingBox(2)/pixels_per_um],ni,1))',s)';
        interframe_d(i,1) = sum(sqrt(sum((xy - xy_p).^2,2)))/length(s);
        interframe_d(i,2) = sum(sqrt(sum((flipud(xy) - xy_p).^2,2)))/length(s);
    end
end

% Flip inverted skeletons and angle measurements
last_flipped = false;
for i=2:N_frames
    if isnan(interframe_d(i,1))
        continue;
    end
    if interframe_d(i,2) < interframe_d(i,1)
        last_flipped = ~last_flipped;
        interframe_d(i,:) = fliplr(interframe_d(i,:));
    end
    if last_flipped
        frames(i).skeleton = flipud(frames(i).skeleton);
        frames(i).theta = fliplr(frames(i).theta);
    end
end

end