function output = measure_body_midpoint(output,pixels_per_um)
% Finds the midpoint of the body
% output: frame_info or analysis structure
% pixels_per_um: conversion to pixels

x=output.skeleton(:,1);x=reshape(x,1,length(x));
y=output.skeleton(:,2);y=reshape(y,1,length(y));
points=length(output.skeleton);

if points <= 2 % Not enough to analyze
    return;
end

% Calculate distance along skeleton
s=zeros(1,points);
for ix=2:points
    s(ix)=s(ix-1)+ sqrt((x(ix)-x(ix-1))^2+(y(ix)-y(ix-1))^2);
end
xy = [x;y];
curve_spline=spline(s,xy);
% Find the midpoint (50% distance)
backbonemid=ppval(curve_spline,max(s)/2);
% Store the position
output.x_mid = (output.stats.BoundingBox(1) + backbonemid(2));
output.x_mid_um = output.x_mid / pixels_per_um;
output.y_mid = (output.stats.BoundingBox(2) + backbonemid(1));
output.y_mid_um = output.y_mid / pixels_per_um;

end