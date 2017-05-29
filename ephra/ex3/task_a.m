pkg load image;

%--------------------------------------------------------------------------
% read an imagefile using a GUI
% 
% Inputs: 
%    - text: String with information for user
% Outputs:
%   - file: name of the file
%   - path: path of the file
%   - image: 2d-array of the image; if the chosen image hat more than one
%     channel, the arithmetic mean of all channels wil be computed 
%   - s: size of the image
function [file, path, image, s] = read_image(text)

    % open a dialogue to pick afile
    [file, path] = uigetfile('*.*', text);
 
    % read training image
    image = imread([path,file]);
    
    % convert image to grayscale image [0,...,1]
    image = mat2gray(mean( image, 3 ));
    s = size(image);
end

## Task A a)
# read image and convert to greyscale 

[f, p, img_grey, shape] = read_image("Choose image");

figure("Name","Grey Image");
imshow(img_grey);

## Task A b)
# Apply GoG_filtering (code reused from assignment 2)
source("assignment2.m");

[GoG_x, GoG_y] = GoG_filtering(img_grey, 0.5);

#figure('Name', 'GoG X');
#imshow(GoG_x);

#figure("Name","GoG Y");
#imshow(GoG_y);

## Task A c)
# Compute the gradient magnitude
img_mag = sqrt(GoG_x.^2 + GoG_y.^2);

#figure("Name","Magnitude");
#imshow(img_mag);

## Task A d)
# Convert magnitude image to binary image
img_bw = im2bw(img_mag, 0.07);

figure("Name","Binary Image with Tresh=0.07");
imshow(img_bw);

## Task A e)
# Hough Line Detection
