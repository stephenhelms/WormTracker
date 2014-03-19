function output = process_img2(img,thresh,bg,clse,compactness_thresh,hole_area_thresh)
% PROCESS_IMG Process color image of worms by removing the background,
% thresholding, cleaning up the worm shapes.
% img: The raw video frame
% thresh: Grayscale threshold
% bg: A bg image that will be subtracted
% clse: A structuring element used for morphological closing (should be
% worm-diameter disk)
% compactness_thresh: A threshold for identifying holes in worms due to
% looped structures based on a measure of roundness
% hole_area_thresh: A threshold for identifying holes in worms due to
% looped structures based on area

% Convert to grayscale
if length(size(img)) == 3
    output.im_gray = rgb2gray(img);
else
    output.im_gray = img;
end
% Remove background - this assumes the image has a light background
if ~isempty(bg)
    output.im_bgrem = imcomplement(imsubtract(bg,output.im_gray));
else
    background = imbothat(output.im_gray,strel('disk',100));
    output.im_bgrem = 255 - (background - output.im_gray);
end

%Use a low pass gaussian filter to get a better outline of the worm.
%Otherwise the transparency of the worm impairs analysis
% Parameters are optimized for Leon's lab's videos
gauss = [7,7];
sigma = 10;
GaussFilter = fspecial('gaussian', gauss, sigma);
output.im_bgrem = imfilter(output.im_bgrem, GaussFilter);

% Threshold image
output.bw_thresh = im2bw(output.im_bgrem,thresh);
% Fix morphological defects in the thresholding by closing
bw_cleaned_close = imclose(~output.bw_thresh,clse);
% Fix morphological defects by filling holes
bw_cleaned_holes = ~imfill(~output.bw_thresh,'holes');

% Combine the fixed images using holes of appropriate compactness and area
% to fill the closed image
cc = bwconncomp(output.bw_thresh & ~bw_cleaned_holes);
stats = regionprops(cc,'Perimeter','Area','PixelIdxList');
compactness = [stats.Perimeter].^2./[stats.Area];
merge = ~bw_cleaned_close;
for i=1:length(compactness)
    if compactness(i) > compactness_thresh & stats(i).Area > hole_area_thresh
        merge(stats(i).PixelIdxList) = 1;
    end
end
% Fix barely touching regions
output.bw_cleaned = bwmorph(~bwmorph(~bwmorph(~merge,'majority'),'diag'),'open');

% Find connected components
output.cc_cleaned = bwconncomp(output.bw_cleaned);

end