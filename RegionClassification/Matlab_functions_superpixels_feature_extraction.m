function [L, N] = applySLIC(imageName) 
    %% Function to apply the SLIC superpixels in a new image
    
    % read image
    I = imread(imageName); 
    
    % get image dimensions
    ImgDim = size(I);

    img_size = size(I,1)*size(I,2);
    
    % desired number of superpixels basaed on image size
    K = ceil(img_size/sp_area);
    
    % application of SLIC superpixels
    % L - matrix with superpixels' labels
    % N - number of superpixels returned
    [L,N] = superpixels(I,K);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
function [Ih, Is, Iv, IL] = getColourChannels(imageName)

   %% Function to get the HSV and L (intensity) channels from image
   
   % read image
   I = imread(imageName); 
   
   % convert image to HSV colourspace
   Ihsv = rgb2hsv(I);
   Ih = Ihsv(:,:,1);
   Is = Ihsv(:,:,2);
   Iv = Ihsv(:,:,3);
   
   % convert image to Lab colourspace
   Ilab = rgb2lab(I);
   IL = Ilab(:,:,1);
   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [croppedImage] = cropSuperpixel(L, idx)

    %% Function to get a small cropped image corresponding to a superpixel to extract texture features
    
    % L - matrix with superpixels' labels
    % idx - desired superpixel
    
    % Get cropped image of the desired superpixel 
    [px, py] = find(L, idx);
    topLine = min(py);
    bottomLine = max(py);
    leftColumn = min(px);
    rightColumn = max(px);
    width = bottomLine - topLine + 1;
    height = rightColumn - leftColumn + 1;
    croppedImage = imcrop(I2, [topLine, leftColumn, width, height]);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [k1] = HueFeature(Ih, L, idx)

    %% Function to get the mean Hue (H colourchannel) from the pixels corresponding to a specific superpixel
    
    % Ih - The hue (H) colour channel
    % L - matrix with superpixels' labels
    % idx - desired superpixel

    spH = Ih(L == idx);
    % Mean Hue
    k1 = mean(spH);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [k2] = SaturationFeature(Is, L, idx)

    %% Function to get the mean Saturation (S colourchannel) from the pixels corresponding to a specific superpixel
    
    % Is - The saturation (S) colour channel
    % L - matrix with superpixels' labels
    % idx - desired superpixel
    
    spS = Is(L == idx);
    % Mean Saturation
    k2 = mean(spS);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [k3] = ValueFeature(Iv, L, idx)

    %% Function to get the mean Value (V colourchannel) from the pixels corresponding to a specific superpixel
    
    % Iv - The Value (V) colour channel
    % L - matrix with superpixels' labels
    % idx - desired superpixel
    
    spV = Iv(L == idx);
    % Mean Value
    k3 = mean(spV);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [k4] = SumFeature(IL, L, idx)

    %% Function to get the sum of intensities (L colourchannel) from the pixels corresponding to a specific superpixel
    
    % IL - The Lightness (Intensity - L) colour channel
    % L - matrix with superpixels' labels
    % idx - desired superpixel
    
    spL = IL(L == idx);
    % sum
    k4 = sum(spL(:));
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [k5] = MaxFeature(IL, L, idx)

    %% Function to get the maximum intensity (L colourchannel) from the pixels corresponding to a specific superpixel
    
    % IL - The Lightness (Intensity - L) colour channel
    % L - matrix with superpixels' labels
    % idx - desired superpixel
    
    spL = IL(L == idx);
    % max
    k5 = max(spL(:));  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [k6] = MinFeature(IL, L, idx)

    %% Function to get the minimum intensity (L colourchannel) from the pixels corresponding to a specific superpixel
    
    % IL - The Lightness (Intensity - L) colour channel
    % L - matrix with superpixels' labels
    % idx - desired superpixel
    
    spL = IL(L == idx);
    % max
    k6 = min(spL(:));  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [k7] = ContrastFeature(IL, L, idx)

    %% Function to get the contrast from the pixels corresponding to a specific superpixel
    
    % IL - The Lightness (Intensity - L) colour channel
    % L - matrix with superpixels' labels
    % idx - desired superpixel
    
    spL = IL(L == idx);
    % contrast
    k7 = max(spL(:)) - min(spL(:));  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [k8] = StandardDeviationFeature(IL, L, idx)

    %% Function to get the standard deviation from the pixels corresponding to a specific superpixel
    
    % IL - The Lightness (Intensity - L) colour channel
    % L - matrix with superpixels' labels
    % idx - desired superpixel
    
    spL = IL(L == idx);
    % Standard deviation
    k8 = std(spL);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [k9] = EntropyFeature(croppedImage)

    %% Function to get the entropy as calculated from MATLAB's entropy function for a specific superpixel
    
    % croppedImage - The cropped image around a superpixel given by the
    % "cropSuperpixel" function
    
    % Entropy
    k9 = entropy(croppedImage);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [k10] = HaralickFeatures(croppedImage)

    %% Function to get Haralick texture features for a specific superpixel
    
    % croppedImage - The cropped image around a superpixel given by the
    % "cropSuperpixel" function
    
    % This Matlab package is needed:
    % https://uk.mathworks.com/matlabcentral/fileexchange/58769-haralicktexturefeatures
    
    % Haralick
    glcm = graycomatrix(croppedImage, 'offset', [2 0], 'Symmetric', true);
    k10 = haralickTextureFeatures(glcm);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [k11] = LBPFeatures(croppedImage)

    %% Function to get Local Binary Patterns texture features for a specific superpixel
    
    % croppedImage - The cropped image around a superpixel given by the
    % "cropSuperpixel" function
    
    % LBP
    k11 = extractLBPFeatures(croppedImage,'Normalization', 'None');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [k12] = STFAFeatures(croppedImage)

    %% Function to get Segmentation-based Fractal Texture Analysis texture features for a specific superpixel
    
    % croppedImage - The cropped image around a superpixel given by the
    % "cropSuperpixel" function
    
    % This Matlab package is needed:
    % https://uk.mathworks.com/matlabcentral/fileexchange/37933-alceufc-sfta
    
    % STFA 
    k12 = sfta(croppedImage,4);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dataStd, mu, sgm] = StandardiseZScores(data)

    %% Function to standardise the features with mean equals to 0 and standard deviation equals to 1
    
    % The means and standard deviations used for centering and scaling are
    % stored 
    
    % The matrix dataStd has the same dimensions with the data matrix
    
    for i=1:size(data,2)-1
        [dataStd(:,i), mu(i), sgm(i)] = zscore(data(:,i));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [model] = trainModel(labels, data)

    %% Function to train SVM model
    
    % The libsvm library (version 3.22) is needed
    % https://www.csie.ntu.edu.tw/~cjlin/libsvm/#download
    % https://github.com/cjlin1/libsvm
    
    % labels - the labels used for training
    
    % experiment with the weights in each of the classes
    
    model = libsvmtrain(labels,data,'-c 10 -w1 1 -w2 1 -w3 5 -w4 1 -w5 1');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
