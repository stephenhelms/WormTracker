function [theta,dS,wormLength,mu]=wormThetaFun(backbone,num_bodypoints)
%given the worm backbone for the tangent angles at num_bodypoints (equal
%arclengths along the body)

x=backbone(:,1);x=reshape(x,1,length(x));
y=backbone(:,2);y=reshape(y,1,length(y));
points=length(backbone);

s=zeros(1,points);
for i=2:points
    s(i)=s(i-1)+ sqrt((x(i)-x(i-1))^2+(y(i)-y(i-1))^2);
end
xy = [x;y];
wormLength=s(end);

%now resample at equal arc-lengths
%a cubic spline interpolation
curve_spline=spline(s,xy);
dS=max(s)/(num_bodypoints+1);
scale=0:dS:max(s);
spline_position=ppval(curve_spline,scale);
x_spline=spline_position(1,:);
y_spline=spline_position(2,:);
theta=zeros(1,num_bodypoints);
%use a symmetric derivative
for i=2:length(scale)-1
  theta(i-1)=atan2((x_spline(i+1)-x_spline(i-1))/2, (y_spline(i+1)-y_spline(i-1))/2);
end
theta=unwrap(theta);
mu = mean(theta);
%theta=unwrap(atan2(diff(x_spline),diff(y_spline)));
theta=theta-mu;