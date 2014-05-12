
%%
figure;
for i=1:size(frame_info,2)
    subplot(1,size(frame_info,2),i);
    imshow(imcrop(first_frame,crop(i,:)))
    hold on;
    ellipse(food_region(i,3),food_region(i,3),0,food_region(i,1),food_region(i,2), ...
        'r');
    plot(traj(i).x(:,1)*pixels_per_um,traj(i).x(:,2)*pixels_per_um,'b.','MarkerSize',2);
end

%%
figure;
for i=1:size(frame_info,2)
    subplot(1,size(frame_info,2),i);
    hist(traj(i).s,200);
    xlim([0 500]);
    axis square
    xlabel('Speed (um/s)');
    ylabel('Counts');
    box off
end

%%
theta_ensemble = NaN(size(frame_info,2),size(frame_info,1),50);
for i=1:size(frame_info,2)
    for j = 1:size(frame_info,1)
        if ~bad_frames(j,i)
            theta_ensemble(i,j,:) = frame_info(j,i).theta;
        end
    end
end
%%
C = nancov(reshape(theta_ensemble,[],50));
[ev,l] = eig(C);
l = flipud(diag(l));
ev = fliplr(ev);
figure;
plot(cumsum(l)./sum(l),'k.-');
xlabel('Postures');
ylabel('Fraction Captured Variance');
%%
figure;
for i=1:4
    subplot(4,1,i);
    plot(ev(:,i),'k-');
end
%%
figure;
for i=1:size(frame_info,2)
    subplot(1,size(frame_info,2),i);
    A = squeeze(theta_ensemble(i,:,:))*ev(:,1:3);
    scatter3(A(:,1),A(:,2),A(:,3),2,'k','.')
end