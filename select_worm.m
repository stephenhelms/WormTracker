function output = select_worm(output,min_area,max_area)
% Selects a worm of appropriate size from a thresholded video frame.
% output: the frame_info or analysis structure
% min_area: the minimum area
% max_area: the maximum area

% Calculate properties of connected components in the image
stats = regionprops(output.cc_cleaned, 'Centroid','FilledArea','BoundingBox','Solidity');

% Find the biggest one with area between min_area and max_area
[~,order] = sort([stats.FilledArea],'descend');
stats = stats(order);
possible_worms = [stats.FilledArea] > min_area & ...
    [stats.FilledArea] < max_area;
idx = find(possible_worms,1,'first');

% If one is found, select it and crop the image to it
if ~isempty(idx)
    output.found_worm = 1;
    output.stats = stats(idx);
    output.bw_worm = imcrop(ismember(labelmatrix(output.cc_cleaned), order(idx)),output.stats.BoundingBox);
    output.im_worm = imcrop(output.im_bgrem,output.stats.BoundingBox);
else
    output.found_worm = 0;
end

end