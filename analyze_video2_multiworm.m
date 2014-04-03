function frame_info = analyze_video(video,video_out,frameRate,pixels_per_um,...
    crop,thresh,bg_disk_radius,W_worm,L_worm,show_analysis)
% crop should be a N worm box x rectangle dimensions
N_worms = size(crop,1);
N_frames = video.NumberOfFrames;

t_frame_est = ones(1,N_frames)*NaN;

% These limits will be used for accepting or rejecting thresholded objects
% as worms
min_expected_worm_area = (0.7*W_worm)*(0.6*L_worm)*pixels_per_um^2;
max_expected_worm_area = (1.3*W_worm)*(1.4*L_worm)*pixels_per_um^2;

% Open the analysis progress figure
if show_analysis
    f_analysis = figure('Position',[ 26   150   800   500]);
    open(video_out); % Open the output video
end

% Determine how many body points to measure
if L_worm*pixels_per_um > 100
    numBodyPoints = 50;%100;
elseif L_worm*pixels_per_um > 50
    numBodyPoints = 50;
elseif L_worm*pixels_per_um > 35
    numBodyPoints = 30;
elseif L_worm*pixels_per_um > 15
    numBodyPoints = 15;
else
    error('Insufficient resolution for postural analysis.');
end

% The worm shape will be filtered using worm-sized disk
clse = strel('disk',round(W_worm*pixels_per_um/2));

disp('Analyzing video frames...');
n_batch = 8*10; % This is the number of frames loaded at once. I have an 8-core processor, and 80 frames is ~350 MB so this works well
% Initialize data structure
frame_info(1:N_frames,1:N_worms) = struct('position_x',NaN,'position_x_um',NaN, ...
    'position_y',NaN,'position_y_um',NaN, 't',num2cell(repmat((1:N_frames)'/frameRate,1,N_worms)),...
    'im_worm',[],'bw_worm',[],'found_worm',false,'bad_frame',false,'bad_worm_at_edge',false,'bad_skeletonization',false, ...
    'length',NaN,'theta',NaN(1,numBodyPoints),'bw_skeleton',[],'skeleton_endpoints',[], ...
    'stats',struct('Centroid',NaN(1,2),'FilledArea',NaN,'BoundingBox',NaN(1,4),'Solidity',NaN),...
    'skeleton',[],'width',[],'outer_boundary',[],'inner_boundary',[],'inner_dist',[],'closest_inner_point',[], ...
    'x_mid',NaN,'y_mid',NaN,'x_mid_um',NaN,'y_mid_um',NaN,'outer_boundary_angle',[],'crossed_frame',false);
% And temporary structure
analysis(1:n_batch,1:N_worms) = struct('found_worm',false,'im_gray',[],'im_bgrem',[],...
        'bw_thresh',[],'bw_cleaned',[],'cc_cleaned',[],'bw_worm',[],'im_worm',[],'crossed_frame',false,'stats', ...
        struct('Centroid',NaN(1,2),'FilledArea',NaN,'BoundingBox',NaN(1,4),'Solidity',NaN));
% Iterate through the video frames
for i=1:n_batch:N_frames
    % Load up to n_batch frames at a time
    n_endgrab = i + n_batch - 1;
    if n_endgrab > N_frames
        n_endgrab = N_frames;
    end
    % These frames are read as RGB, so save memory by cropping, immediately
    % converting them back to grayscale and deleting the RGB version
    tic;
    raw_frames = read(video,[i,n_endgrab]);
    test_frame = rgb2gray(raw_frames(:,:,:,1));
    frames = zeros([size(test_frame,1),size(test_frame,2),size(raw_frames,4)],'uint8');
    parfor j=1:size(raw_frames,4)
            frames(:,:,j) = rgb2gray(raw_frames(:,:,:,j));
    end
    clear raw_frames test_frame
    disp(['Reading ', num2str(size(frames,3)), ' frames took ', num2str(toc), ' s.']);
    temp = whos('frames');
    disp(['Frames take up ', num2str(temp.bytes/1000/1000), ' MB of memory.']);
    
    % Iteratively preprocess frames in parallel
    tic;
    parfor j=1:size(frames,3)
        for k=1:N_worms
            % Process frame to identify connected components
            processed_frame = process_img2(imcrop(frames(:,:,j),crop(k,:)),thresh,...
                bg_disk_radius,clse,10,10);

            analysis(j,k).im_gray = processed_frame.im_gray;
            analysis(j,k).bw_cleaned = processed_frame.bw_cleaned;
            analysis(j,k).bw_thresh = processed_frame.bw_thresh;
            analysis(j,k).im_bgrem = processed_frame.im_bgrem;
            analysis(j,k).cc_cleaned = processed_frame.cc_cleaned;

            % Find worm
            analysis(j,k) = select_worm(analysis(j,k),min_expected_worm_area,max_expected_worm_area);
        end
    end
    tpre = toc/size(frames,3);
    
    % Sequentially analyze the worm position and shape in each frame
    for j=1:size(frames,3)
        tic;
        for k=1:N_worms
            % Transfer variables from the temporary structure (needed for the
            % parfor loop) to the frame_info structure
            frame_info(i+j-1,k).found_worm = analysis(j,k).found_worm;
            if frame_info(i+j-1,k).found_worm
                frame_info(i+j-1,k).position_x = analysis(j,k).stats.Centroid(1);
                frame_info(i+j-1,k).position_x_um = frame_info(i+j-1,k).position_x/pixels_per_um;
                frame_info(i+j-1,k).position_y = analysis(j,k).stats.Centroid(2);
                frame_info(i+j-1,k).position_y_um = frame_info(i+j-1,k).position_y/pixels_per_um;
                frame_info(i+j-1,k).im_worm = analysis(j,k).im_worm;
                frame_info(i+j-1,k).bw_worm = analysis(j,k).bw_worm;
                frame_info(i+j-1,k).stats = analysis(j,k).stats;
            end

            frame_info(i+j-1,k).bad_frame = false;
            frame_info(i+j-1,k).bad_worm_at_edge = false;
            frame_info(i+j-1,k).bad_skeletonization = false;
            % Check whether a worm was found
            if ~analysis(j,k).found_worm
                frame_info(i+j-1,k).bad_frame = true;
                frame_info(i+j-1,k).length = NaN;
                frame_info(i+j-1,k).theta = [];
                frame_info(i+j-1,k).stats = [];
                frame_info(i+j-1,k).bw_worm = [];
                frame_info(i+j-1,k).skeleton = [];
            % Check whether worm is touching the border
            elseif any(analysis(j,k).stats.BoundingBox(1:2) < 2) || ...
                any(size(analysis(j,k).bw_cleaned) - ...
                (analysis(j,k).stats.BoundingBox(1:2) + analysis(j,k).stats.BoundingBox(3:4)) < 2);
                frame_info(i+j-1,k).bad_frame = true; % If so, throw out the frame
                frame_info(i+j-1,k).bad_worm_at_edge = true;
                frame_info(i+j-1,k).length = NaN;
                frame_info(i+j-1,k).theta = [];
                frame_info(i+j-1,k).stats = [];
                frame_info(i+j-1,k).bw_worm = [];
                frame_info(i+j-1,k).skeleton = [];
                frame_info(i+j-1,k).bad_skeletonization = true;
            % If all good, do the analysis
            else
                % Skeletonize
                frame_info(i+j-1,k)=skeletonize(frame_info(i+j-1,k));
                % Find boundaries and measure width
                frame_info(i+j-1,k) = find_boundaries(frame_info(i+j-1,k),pixels_per_um,W_worm);
                % Problems could arise during skeletonization and
                % boundary-finding, so make sure everything is still good
                if ~frame_info(i+j-1,k).bad_frame
                    % Correct crossed shapes
                    % Disabling crossed shape correction for now
                    if size(frame_info(i+j-1,k).skeleton_endpoints,1) == 0
                        %frame_info(i+j-1,k) = cut_coiled_worm(frame_info(i+j-1,k),W_worm,pixels_per_um);
                        frame_info(i+j-1,k).crossed_frame = true;
                        frame_info(i+j-1,k).bad_frame = true;
                    elseif size(frame_info(i+j-1,k).skeleton_endpoints,1) == 1
                        %frame_info(i+j-1,k) = cut_crossed_worm(frame_info(i+j-1,k),W_worm,pixels_per_um);
                        frame_info(i+j-1,k).crossed_frame = true;
                        frame_info(i+j-1,k).bad_frame = true;
                    end
                    % Check again that everything is ok -- the skeleton
                    % sometimes is overcorrected or not fixed
                    if size(frame_info(i+j-1,k).skeleton_endpoints,1) ~= 2
                        frame_info(i+j-1,k).bad_skeletonization = true;
                    end
                    if ~frame_info(i+j-1,k).bad_skeletonization
                        % Measure the body midpoint
                        frame_info(i+j-1,k) = measure_body_midpoint(frame_info(i+j-1,k),pixels_per_um);

                        % Find the tangent angles along the ordered skeleton
                        [frame_info(i+j-1,k).theta,~,L]=...
                            wormThetaFun(frame_info(i+j-1,k).skeleton,numBodyPoints);
                        % Store the length of the worm (in um)
                        frame_info(i+j-1,k).length = L / pixels_per_um;

                        % Check for bad skeletonization due to implausible
                        % lengths
                        if frame_info(i+j-1,k).length < 0.5*L_worm || frame_info(i+j-1,k).length > 2*L_worm
                            frame_info(i+j-1,k).bad_skeletonization = true;
                        end
                    end
                else
                    frame_info(i+j-1,k).length = NaN;
                    frame_info(i+j-1,k).theta = [];
                    frame_info(i+j-1,k).stats = [];
                    frame_info(i+j-1,k).bw_worm = [];
                    frame_info(i+j-1,k).skeleton = [];
                    frame_info(i+j-1,k).bad_skeletonization = true;
                end
            end
        end
        if show_analysis
            % Plot progress
            plot_worm_tracking_video_analysis_mw(f_analysis,frame_info,i+j-1,analysis(j,:));
        end
        t_analysis = tpre + toc;
        disp(['Analyzing frame took ', num2str(t_analysis), ' s.']);
        if show_analysis
            % Output analysis to video
            tic;
            writeVideo(video_out, getframe(f_analysis));
            t_write = toc;
            disp(['Writing frame to video took ', num2str(t_write), ' s.']);
        else
            t_write = 0;
        end
        t_frame_est(i+j-1) = t_analysis + t_write;
        disp(['Estimated to finish in ', num2str((nanmean(t_frame_est))*(N_frames-(i+j))/60/60), ' h.']);
    end
    % Save results at end of each batch
    %save('temp_img_analysis.mat','frame_info');
end
if show_analysis
    % Close the video to save the file
    close(video_out);
end

end