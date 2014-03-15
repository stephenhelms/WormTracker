%% Load video
videoFile = ‘your_video.avi’;
video = VideoReader(videoFile);
frameRate = 11.5; % should be changed to match the frame rate of your camera
figure; % This takes a while sometimes, so this just lets you know it is done
%% Read first frame
first_frame = rgb2gray(read(video,1));
%% Measure known distance
known_distance = 10000; % in um, enter a value you can estimate from the video.
% alternatively, use the length of a worm (~1000 um) and click on the ends of a worm

f = figure;
imshow(first_frame,'InitialMagnification','fit');
title(['Click on known points separated by ', num2str(known_distance/1000), ' mm.']);
[x,y] = ginput(2);
close(f);
pixels_per_um = sqrt((y(2)-y(1))^2 + (x(2)-x(1))^2)/known_distance

%% Pick crop boundaries
f = figure;
subplot(2,4,1:4); %If you have something other than 4 regions, change this
imshow(first_frame,'InitialMagnification','fit');
title(['Drag to select the cropped area']);
crop = [0 0 0 0];
for i=1:4 % Also change this
    figure;
    [first_frame_crop,crop(i,:)] = imcrop(first_frame);
    figure(f);
    subplot(2,4,4+i);
    imshow(first_frame_crop,'InitialMagnification','fit');
end

%% Estimate background
N_sample = 50;
background = estimate_background(video,N_sample);
figure;
for i=1:size(crop,1)
    bg_crop{i} = imcrop(background,crop(i,:));
end
figure;
first_frame_corrected = imcomplement(imsubtract(background,first_frame));
subplot(1,3,1);
imshow(first_frame);
title('First Frame');
subplot(1,3,2);
imshow(background);
title('Background');
subplot(1,3,3);
imshow(first_frame_corrected);
title('First Frame, Corr.');

%% Determine appropriate threshold
thresh = 0.90;
figure;
h = [];
h(1) = subplot(1,2,1);
imshow(first_frame_corrected);
title('First Frame, Corrected');
h(2) = subplot(1,2,2);
imshow(im2bw(first_frame_corrected,thresh));
title('Thresholded');
linkaxes(h);
% Would be nice to update this with an actual program that let you click
% the threshold up and down

%% Worm parameters - used for morphology operations and quality control
% Try to find a worm automatically and measure the length and width
[L_worm,W_worm] = estimate_worm_length_width(imcrop(first_frame,crop(1,:)),bg_crop{1},thresh,pixels_per_um)
% If this fails, manually enter estimates of worm length and width
%L_worm = 1000;
%W_worm = 35;

% The worm shape will be filtered using worm-sized disk
clse = strel('disk',round(W_worm*pixels_per_um/2));

%% Save analysis video to:
analysisVideoName = 'test.avi';
video_out = VideoWriter(analysisVideoName);
video_out.FrameRate = frameRate*2; % 2X speed

%% Analyze video
outputName = ’test_analysis.mat’;
save(outputName);
frame_info = analyze_video2_multiworm(video,video_out,frameRate,pixels_per_um,crop,thresh,bg_crop,clse,W_worm,L_worm,1); % For faster analysis, set the last parameter to 0 — this turns off plotting
close(video_out);
save(outputName);
post_process_mw_video
save(outputName);