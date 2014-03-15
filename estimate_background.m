function background = estimate_background(video,N_sample)
% estimate_background estimates the unchanging background in a video by
% averaging all the frames, evenly sampled N_sample times

background = zeros(video.Height,video.Width,'uint8');
N_frames = video.NumberOfFrames;
ii = 0;
t = zeros(1,N_sample);
for i=1:round(N_frames/N_sample):N_frames
    tic;
    frame = rgb2gray(read(video,i));
    background = imadd(background,imdivide(frame,(N_sample+1)));
    ii = ii + 1;
    t(ii) = toc;
    disp([num2str(ii/(N_sample+1)*100),'% complete']);
    disp([num2str(mean(t(1:ii))*((N_sample+1)-ii)), ' s until completion.']);
end

% Paranoid check to make sure I sampled as many times as I should have
if abs(ii - (N_sample+1)) > 0.1
    background = immultiply(background,(N_sample+1)/ii);
end

end

