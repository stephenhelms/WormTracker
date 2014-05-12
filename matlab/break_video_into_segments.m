function segments = break_video_into_segments(frames,bad_frames,pixels_per_um,interframe_d)
% Breaks video into a small number of blocks of consecutive data for
% head/tail assignment

N_frames = length(frames);

% Segment matching parameters
max_n_missing = 10;
max_d = 10/pixels_per_um;
max_segment_frames = 500;
min_segment_size = 150;

% Break video into segments with matched skeletons
ii = 1;
n_segment = 1;
segments = zeros(round(N_frames/max_segment_frames),2);
while ii < N_frames
    segments(n_segment,1) = ii;
    ii = ii + 1;
    % Continue segment until >max_n_missing consecutive bad frames are
    % found, or >max_segment_frames are collected
    n_missing = 0;
    last_missing = false;
    while ii < N_frames && ...
            (ii - segments(n_segment,1)) < max_segment_frames && ...
            (isnan(interframe_d(ii,1)) || min(interframe_d(ii,:))<max_d)
        if bad_frames(ii)
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
    segments(n_segment,2) = ii - 1;
    n_segment = n_segment + 1;
end

% Check whether each segment meets criteria (min segment size)
keep_segment = false(size(segments,1),1);
for n=1:size(segments,1)
    keep_segment(n) = sum(~isnan(interframe_d(segments(n,1):segments(n,2),1))) > ...
        min_segment_size;
end
segments = segments(keep_segment,:);

end