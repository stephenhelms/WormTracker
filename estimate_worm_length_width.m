function [L_worm,W_worm] = estimate_worm_length_width(frame,bg,thresh,pixels_per_um)

processed_frame = process_img2(frame,thresh,...
                bg,strel('disk',2),10,10);

min_expected_worm_area = 30*300*pixels_per_um^2*0.7;
max_expected_worm_area = 150*2000*pixels_per_um^2*1.3;
processed_frame = select_worm(processed_frame,min_expected_worm_area,max_expected_worm_area);
processed_frame=skeletonize(processed_frame);
processed_frame = find_boundaries(processed_frame,pixels_per_um,30);
figure;
subplot(1,2,1);
imshow(imadjust(processed_frame.im_worm),'InitialMagnification','fit');
subplot(1,2,2);
imshow(processed_frame.bw_worm,'InitialMagnification','fit');

[~,~,L]=wormThetaFun(processed_frame.skeleton,25);
L_worm = L / pixels_per_um;
W_worm = processed_frame.width;

end