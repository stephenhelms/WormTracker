%% Load video
videoFile = 'U:\Helms\2014-03-11_lena_test_chambers_xvid.avi';
video = VideoReader(videoFile);
frameRate = 11.5; % should be changed to match the frame rate of your camera
figure; % This takes a while sometimes, so this just lets you know it is done
%% Read first frame
first_frame = rgb2gray(read(video,1));
%% Measure known distance
known_distance = 25500; % in um, the width of the slide
% alternatively, use the length of a worm (~1000 um) and click on the ends of a worm

f = figure;
imshow(first_frame,'InitialMagnification','fit');
title(['Click on known points separated by ', num2str(known_distance/1000), ' mm.']);
[x,y] = ginput(2);
close(f);
pixels_per_um = sqrt((y(2)-y(1))^2 + (x(2)-x(1))^2)/known_distance

%% Pick crop boundaries
f = figure;
n_regions = 4;%If you have something other than 4 regions, change this
crop = [0 0 0 0];
for i=1:n_regions % Also change this
    [first_frame_crop,crop(i,:)] = imcrop(first_frame);
end

%% Draw food circles
f = figure;
food_region = zeros(n_regions,3);
for i=1:n_regions
    imshow(imcrop(first_frame,crop(i,:)));
    h = imellipse(gca,[0 0 100 100]);
    setFixedAspectRatioMode(h,true);
    wait(h);
    pos = getPosition(h);
    food_region(i,:) = [pos(1)+pos(3)/2,pos(2)+pos(4)/2,pos(3)/2];
    hold on;
    plot(food_region(i,1),food_region(i,2),'ro');
    plot(food_region(i,1)+[0 food_region(i,3)],food_region(i,2)*[1 1],'r-');
end

%% Configure background removal filter
disk_radius = 5;
first_frame_corrected = imcomplement( ...
    imbothat(first_frame,strel('disk',disk_radius)));
figure; imshow(imadjust(first_frame_corrected));

%% Determine appropriate threshold
thresh = 0.88;
figure;
h = [];
h(1) = subplot(1,3,1);
imshow(first_frame);
title('First Frame');
h(2) = subplot(1,3,2);
imshow(first_frame_corrected);
title('First Frame, Corrected');
h(3) = subplot(1,3,3);
imshow(im2bw(first_frame_corrected,thresh));
title('Thresholded');
linkaxes(h);
% Would be nice to update this with an actual program that let you click
% the threshold up and down

%% Worm parameters - used for morphology operations and quality control
% Try to find a worm automatically and measure the length and width
i = 3;
[L_worm,W_worm] = estimate_worm_length_width(imcrop(first_frame,crop(i,:)),...
    disk_radius,thresh,pixels_per_um)
% If this fails, manually enter estimates of worm length and width
%L_worm = 1000;
%W_worm = 35;

%% Save analysis video to:
analysisVideoName = 'test.avi';
video_out = VideoWriter(analysisVideoName);
video_out.FrameRate = frameRate*2; % 2X speed

%% Analyze video
outputName = 'test_analysis.mat';
save(outputName);
% For faster analysis, set the last parameter to 0 — this turns off plotting
frame_info = analyze_video2_multiworm(video,video_out,frameRate,pixels_per_um,...
    crop,thresh,disk_radius,W_worm,L_worm,0);
close(video_out);
save(outputName);
post_process_mw_video
save(outputName);