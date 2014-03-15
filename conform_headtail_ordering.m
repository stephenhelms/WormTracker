function output = conform_headtail_ordering(output,prevHeadTail)
% output: the frame_info(index) structure
% prevHeadTail: a 2x2 matrix in which the columns are the x and y positions
% of the two ends

d = zeros(2,2);
% Calculate the location of the skeleton ends (stored relative to the
% bounding box) relative to the body midpoint
skeleton_ends_relpos = repmat(output.stats.BoundingBox(1:2),2,1) + ...
    output.skeleton_endpoints - repmat([output.x_mid,output.y_mid],2,1);

% Distance from each endpoint:
% Columns: Endpoint 1 of prev frame, endpoint 2 of prev frame
% Rows: Endpoint 1 of current frame, endpoint 2 of current frame
d(:,1) = sqrt(sum((skeleton_ends_relpos - ...
    repmat(prevHeadTail(1,:),2,1)).^2,2));
d(:,2) = sqrt(sum((skeleton_ends_relpos - ...
    repmat(prevHeadTail(2,:),2,1)).^2,2));

% Assign the smallest distance and force the other to the remaining
[~,i_best_col] = min(min(d,[],1));
[~,i_best_row] = min(d(:,i_best_col));
if i_best_col ~= i_best_row % H/T are flipped
    % Flip skeleton
    output.skeleton = flipud(output.skeleton);
end

end