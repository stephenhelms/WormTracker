function output = skeletonize(output)

% Generate skeleton by a series of morphological operations
output.bw_skeleton = bwmorph(output.bw_worm,'skeleton',Inf); % Skeletonization
output.bw_skeleton = bwmorph(output.bw_skeleton,'thin',Inf); % Thinning-fixes problems in which there are >2 connectivity pixels
output.bw_skeleton = output.bw_skeleton & ~bwmorph(output.bw_skeleton,'endpoints'); % Fixes simple spur issues in which an end is branched
output.bw_skeleton = bwmorph(output.bw_skeleton,'thin',Inf); % Same as before
% Convert bw image of skeleton to positions
[x,y] = ind2sub(size(output.bw_skeleton), ...
        find(bwmorph(output.bw_skeleton,'endpoints')));
output.skeleton_endpoints = [x,y];
ii = 0;
n_ends = 2;
n_branch = length(find(bwmorph(output.bw_skeleton,'branchpoints'))); % Identify branch points
% Continue refining skeleton until there are exactly 2 branch+endpoints
max_iter = 100;
while size(output.skeleton_endpoints,1) > (n_ends - n_branch) & ii < max_iter
    output.bw_skeleton = bwmorph(output.bw_skeleton,'spur'); % Remove 1 pixel from ends
    output.bw_skeleton = bwmorph(output.bw_skeleton,'thin',Inf); % Thin to correct for >2-conn pixels
    [x,y] = ind2sub(size(output.bw_skeleton), ...
        find(bwmorph(output.bw_skeleton,'endpoints')));
    output.skeleton_endpoints = [x,y];
    ii = ii + 1;
    n_branch = length(find(bwmorph(output.bw_skeleton,'branchpoints')));
end

% Order the skeleton
output.skeleton = order_skeleton(output.bw_skeleton);

% Check for obvious problems
if size(output.skeleton,1) < 10 || ii == max_iter
    output.bad_frame = true;
    output.bad_skeletonization = true;
end

end