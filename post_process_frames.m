function [frames,output] = post_process_frames(frames,bad_frames,pixels_per_um,fps)
% Tries to match the head and tail in successive frames

N_frames = length(frames);

% Measure position, velocity, speed, and bearing
dt = 1/fps;
output.t = (0:(N_frames-1))*dt;
output.x = [reshape([frames.position_x_um],[],1), ...
    reshape([frames.position_y_um],[],1)];
output.x(bad_frames,:) = NaN;
output.xm = [reshape([frames.x_mid_um],[],1), ...
    reshape([frames.y_mid_um],[],1)];
output.xm(bad_frames,:) = NaN;
output.v = [NaN,NaN;(output.x(3:end,:)-output.x(1:end-2,:))/2/dt;NaN,NaN];
output.v(bad_frames,:) = NaN;
output.s = sqrt(sum(output.v.^2,2));
output.s(bad_frames) = NaN;
output.phi = atan2(output.v(:,2),output.v(:,1));
output.phi(bad_frames) = NaN;

% Arrange postural data in a consistent way
[frames,interframe_d] = normalize_skeleton_ends(frames,bad_frames,pixels_per_um);
output.interframe_d = interframe_d;

% Calculate skeleton end orientation
ht_orientation = NaN*ones(N_frames,2);
for i=2:N_frames
    if ~bad_frames(i) & ~isnan(output.interframe_d(i,1))
        p = fliplr((frames(i).skeleton([1,size(frames(i).skeleton,1)],:) + ...
            repmat(frames(i).stats.BoundingBox([2,1]),2,1))/pixels_per_um);
        ht_orientation(i,:) = atan2(p(:,2)-output.x(i,2),p(:,1)-output.x(i,1))';
    end
end
output.ht_orientation = ht_orientation;

% Determine number of body angles that were measured
N_angles = [];
idx_good = find(~bad_frames);
for i=1:length(idx_good)
    if ~isempty(frames(idx_good(i)).theta)
        N_angles = length(frames(idx_good(i)).theta);
        break;
    end
end

% Calculate skeleton end velocity
x_ends = NaN(N_frames,4);
for j=1:N_frames
    if size(frames(j).skeleton,1) > N_angles && ~isnan(output.interframe_d(j,1))
        x_ends(j,:) = ([frames(j).skeleton(1,:), ...
                frames(j).skeleton(size(frames(j).skeleton,1),:)] ...
                + repmat(frames(j).stats.BoundingBox([2,1]),1,2))/pixels_per_um;
    end
end
output.x_ends = x_ends;
output.v_ends = [NaN,NaN,NaN,NaN; ...
    (x_ends(3:end,:)-x_ends(1:end-2,:))/2/dt; ...
    NaN,NaN,NaN,NaN];
output.s_ends = [sqrt(sum(output.v_ends(:,1:2).^2,2)),...
    sqrt(sum(output.v_ends(:,3:4).^2,2))];
output.phi_ends = [atan2(output.v_ends(:,2),output.v_ends(:,1)), ...
    atan2(output.v_ends(:,4),output.v_ends(:,3))];
output.dphi_ends = atannorm(output.phi_ends-repmat(output.phi,1,2));

% Break video into segments
output.segments = break_video_into_segments(frames,bad_frames,pixels_per_um,interframe_d);

% Assign head in each segment
[frames,output] = assign_head_tail(frames,output,bad_frames);

% Make head-dependent measurements
output.psi = output.ht_orientation(:,1);
output.dpsi = atannorm(output.psi - output.phi);

% Store info about worm
output.im_worm = imadjust(frames(find(~bad_frames,1)).im_worm);
output.length = nanmedian([frames(~bad_frames).length]);
output.width = nanmedian([frames(~bad_frames).width]);

end