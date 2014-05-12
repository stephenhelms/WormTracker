function backbone=order_skeleton(skeleton)

if ~any(skeleton(:))
    backbone = [];
    return
end

skeleton_indices=find(skeleton>0);
%find the x,y points of the skeleton
[skeleton_x,skeleton_y]=ind2sub(size(skeleton),skeleton_indices);

% Start at an endpoint, if one exists
endpoint = find(bwmorph(skeleton,'endpoints'),1);
if isempty(endpoint)
    endpoint = 1; % Otherwise just use the upperleftmost point
else
    [i,j] = ind2sub(size(skeleton),endpoint);
    endpoint = find(skeleton_x == i & skeleton_y == j,1);
    if isempty(endpoint)
        endpoint = 1;
    end
end

x=skeleton_x(endpoint);
y=skeleton_y(endpoint);

%now we use the origin position to order the points along the skeleton
backbone_indices=endpoint;
max_runs = length(skeleton_indices);
ii = 0;
while length(backbone_indices)<length(skeleton_indices) & ii <= max_runs;
    ii = ii + 1;
    i=skeleton_x(backbone_indices(end));
    j=skeleton_y(backbone_indices(end));
    adjacent_skeleton = find(skeleton_x>=i-1 & skeleton_x<=i+1 & ...
        skeleton_y>=j-1 & skeleton_y<=j+1);
    remaining_adjacent=setdiff(adjacent_skeleton,backbone_indices);
    if isempty(remaining_adjacent)
        break;
    end
    % Pick the upperleftmost adjacent pixel
    adjacent_ordered_row = i + [0 -1 -1 -1 0 1 1 1];
    adjacent_ordered_col = j + [-1 -1 0 1 1 1 0 -1];
    next_index = [];
    pos = [skeleton_x(remaining_adjacent),skeleton_y(remaining_adjacent)];
    for i=1:8
        i_match = find(pos(:,1) == adjacent_ordered_row(i) & ...
            pos(:,2) == adjacent_ordered_col(i));
        if ~isempty(i_match)
            next_index = remaining_adjacent(i_match);
            break;
        end
    end
    backbone_indices=cat(2,backbone_indices,next_index);
end
backbone=[skeleton_x(backbone_indices),skeleton_y(backbone_indices)];

end
