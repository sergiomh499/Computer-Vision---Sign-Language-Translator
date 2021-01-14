function [features, setLabels] = helperExtractHOGFeaturesFromImageSet(imds, hogFeatureSize1, cellSize)
% Extract HOG features from an imageDatastore.

setLabels = imds.Labels;
numImages = numel(imds.Files);
features  = zeros(numImages, hogFeatureSize1, 'single');

% Process each image and extract features
for j = 1:numImages
    img = readimage(imds, j);
    img_bw = rgb2gray(img);
    img_bin = imbinarize(img_bw);
    img(:,:,1) = abs(1 - double(img_bin)).*double(img(:,:,1));
    img(:,:,2) = abs(1 - double(img_bin)).*double(img(:,:,2));
    img(:,:,3) = abs(1 - double(img_bin)).*double(img(:,:,3));
    features(j, :) = extractHOGFeatures(img,'CellSize',cellSize);
end
end