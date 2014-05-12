function output = find_boundaries(output,pixels_per_um,W_worm)
% Finds the boundaries of the thresholded worm and calculates the width and
% additional properties

% Identify boundaries of worm
boundaries = bwmorph(output.bw_worm,'remove',Inf); % Remove interior pixels
result = bwmorph(bwmorph(boundaries,'thin',Inf),'spur',Inf); % Correct for >2-conn pixels
output.bw_worm(imfill(boundaries,'holes') & ~imfill(result,'holes')) = false;
boundaries = result;

% If worm was just barely touching, the boundary may have connectivity-3
% pixels, which will disrupt our analysis. Clean the boundary by removing
% these pixels

%figure;
%imshow(boundaries,'InitialMagnification','fit');
%[~,output,boundaries] = remove_touching_boundary_points(output,boundaries);
if is_boundary_touching(output,boundaries)
    % Clean up any 1-pixel diagonal lines which cause problems with
    % the boundary
    output.bw_worm = ~bwmorph(~output.bw_worm,'diag');
    boundaries = bwmorph(output.bw_worm,'remove',Inf);
    
    % Check again
    if is_boundary_touching(output,boundaries)
        % Use the stronger algorithm to clean up the boundary
        [modified,output,boundaries] = remove_touching_boundary_points(output,boundaries);
        %if modified
        %    imshow(boundaries,'InitialMagnification','fit');
        %end
        
        % Repeat as needed up to 10 times, at which point declare the frame
        % bad
        ii = 1;
        while modified & is_boundary_touching(output,boundaries)
        %    imshow(boundaries,'InitialMagnification','fit');
            [modified,output,boundaries] = remove_touching_boundary_points(output,boundaries);
            ii = ii + 1;
            if ii > 10
                output.bad_frame = true;
                break;
            end
        end
    end
end

% Measure the width of the worm
width = measure_width(output);
output.width = nanmedian(width)/pixels_per_um;
if any(width*pixels_per_um > W_worm*1.5)
    output.bad_frame = true;
end

% If there are two boundaries, the larger is the outer boundary
cc = bwconncomp(boundaries);
stats = regionprops(cc,'FilledArea');
[~,order] = sort([stats.FilledArea],'descend');
if isempty(order)
    output.bad_frame = true;
    return;
end
l = labelmatrix(cc);
outer = ismember(l,order(1));
output.outer_boundary = order_skeleton(outer); % Orders the boundary pixels
if length(stats) > 1
    if any(any(imfill(l==order(1),'holes') & imfill(l==order(2),'holes')))
        inner = ismember(l,order(2));
        output.inner_boundary = order_skeleton(inner); % Orders the interior boundary pixels
    else % Two non-overlapping areas detected
        output.bad_frame = true;
        return;
    end
end

% Compute (1) the minimal distance from the outer boundary points to the inner
% boundary and (2) the curvature 
N_points = size(output.outer_boundary,1);
for i=1:N_points
    if length(stats) > 1
        % Distance
        [output.inner_dist(i),output.closest_inner_point(i)] = ...
            min(sqrt(sum((repmat(output.outer_boundary(i,:), ...
            size(output.inner_boundary,1),1) ...
            - output.inner_boundary).^2,2)));
    end
    
    % Curvature, using two points away in each direction
    np = 2;
    ip = i - np;
    if ip < 1
        ip = N_points + ip - np;
    end
    in = i + np;
    if in > N_points
        in = np + in - N_points;
    end
    output.outer_boundary_angle(i) = acosd(dot(output.outer_boundary(ip,:)-output.outer_boundary(i,:), ...
        output.outer_boundary(in,:)-output.outer_boundary(i,:)) / ...
        (sqrt(dot(output.outer_boundary(ip,:)-output.outer_boundary(i,:), ...
        output.outer_boundary(ip,:)-output.outer_boundary(i,:)))* ...
        sqrt(dot(output.outer_boundary(in,:)-output.outer_boundary(i,:), ...
        output.outer_boundary(in,:)-output.outer_boundary(i,:)))));
end

    function tf = is_boundary_touching(output,boundaries)
        % Boundary is touching if the skeleton has a branch but only one
        % boundary is found
       cc = bwconncomp(boundaries);
       tf = size(output.skeleton_endpoints,1) < 2 & cc.NumObjects == 1;
    end
    function [modified,output,boundaries] = remove_touching_boundary_points(output,boundaries)
        % Find skeleton branchpoints
        branchpoints = find(bwmorph(output.bw_skeleton,'branchpoints'),1);
        if isempty(branchpoints) % If no branchpoints, try opening to clear
            % barely touching regions
            result = imopen(output.bw_worm,strel('disk',2));
            modified = any(result(:) ~= output.bw_worm(:));
            output.bw_worm = result;
        else
            % Close image with small disk, find removed regions
            thin_regions = output.bw_worm & ~imopen(output.bw_worm,strel('disk',2));

            cc = bwconncomp(thin_regions);
            stats = regionprops(cc,'Centroid','PixelIdxList');

            % Measure distance to branchpoint
            [xy(1),xy(2)] = ind2sub(size(output.bw_worm),branchpoints);
            d = zeros(length(stats),1);
            for i=1:length(stats)
                d(i) = sqrt(sum((xy - stats(i).Centroid).^2,2));
            end
            % Touching region is the thin region closest to the branchpoint
            [~,i_remove] = min(d);
            output.bw_worm(stats(i_remove).PixelIdxList) = 0;

            modified = true;
        end
                
        if modified
            % Clean up thresholded image
            output.bw_worm = ~bwmorph(~output.bw_worm,'diag');
            output.bw_worm = bwmorph(output.bw_worm,'clean');
            output.bw_worm = bwmorph(output.bw_worm,'spur');
            output = skeletonize(output); % Reskeletonize
            boundaries = bwmorph(output.bw_worm,'remove',Inf); % Boundaries
            boundaries = bwmorph(bwmorph(boundaries,'thin',Inf),'spur',Inf); % Correct for >2-conn pixels
        end
     end

    function [width,angle] = measure_width(output)
        width = NaN*ones(size(output.skeleton,1),1);
        angle = NaN*ones(size(output.skeleton,1),1);
        for i=2:(size(output.skeleton,1)-1)
            angle(i) = atan2(output.skeleton(i+1,2)-output.skeleton(i-1,2), ...
                output.skeleton(i+1,1)-output.skeleton(i-1,1));
            % Measure width perpendicular to skeleton
            dp = [cos(angle(i)+pi/2),sin(angle(i)+pi/2)];
            xy = output.skeleton(i,:);
            last_xy = xy;
            xy_round = round(xy);
            width(i) = 1;
            while all(xy_round > 1) & all(size(output.bw_worm)-xy_round > 1) & ...
                    output.bw_worm(xy_round(1),xy_round(2))
                xy = xy + dp/10;
                xy_round = round(xy);
                if any(abs(xy_round - last_xy)) > 0.1
                    if output.bw_worm(xy_round(1),xy_round(2))
                        width(i) = width(i) + 1;
                    end
                end
                last_xy = xy_round;
            end
            while all(xy_round) > 1 & all(size(output.bw_worm)-xy_round > 1) & ...
                    output.bw_worm(xy_round(1),xy_round(2))
                xy = xy - dp/10;
                xy_round = round(xy);
                if any(abs(xy_round - last_xy)) > 0.1
                    if output.bw_worm(xy_round(1),xy_round(2))
                        width(i) = width(i) + 1;
                    end
                end
                last_xy = xy_round;
            end
        end
    end 

end