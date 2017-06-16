function task_a

% just close the windows please
  close all
% thanks

%% Task A

% Task A.a
% Read the image as grey value image with double precision
  img = imread('taskA.png');
  img = im2double(rgb2gray(img)); 
  
  figure('Name', 'Initial Image');
  imshow(img);
  title('initial image');
  
  figure('Name', 'Initial Image - logarithmic centered image spectra');
  colormap(jet);
  imagesc(log(abs(fftshift(fft2(img)))));  
  title('logarithmic centered image spectra');
  
% Task A.b
% Add some noise to the image
  img_noise = imnoise(img, 'gaussian', 0.01);
  figure('Name', 'Image with gaussian noise');
  imshow(img_noise);
  
% Task A.c
% Filter the noisy image with a 2d Gaussian filter
% Using a sigma greater than 1 leads to way smoother results in the final
% calculated images, since choosing a relatively low sigma results in a
% smaller gaussian kernel. The final smoothing is highly dependent on the
% used gog kernel. If the kernel is chosen to low, the prior added noise is
% not smooth and rather emphasized.
  [gog, radius] = Create_GoG_Kernel(1.6);
  figure('Name', 'GoG filter');
  imshow(gog, [], 'InitialMagnification', 'fit');

% calculate the smoothed image
  [img_smooth, filter_fft, image_fft] = Smooth_in_frequency_domain(img_noise, gog, radius);
  
  figure('Name', 'Smoothing Image & Filter');
  subplot(211);
  imagesc(log(abs(fftshift(image_fft))));
  colormap(jet); colorbar
  title('Image fft2');
    
  subplot(212);
  imagesc(log(abs(filter_fft)));
  colormap(jet); colorbar
  title('Filter fft2');
  
  figure('Name', 'Frequency smoothed image');
  imshow(img_smooth);
  title('Smooth image');
  
%% Task A.d
 % already covered during the assignments single tasks
  
  
end

function [smooth_img, filter_fft, image_fft] = Smooth_in_frequency_domain(image_src, gog_filter, radius)
  
% get matrix and filter size
  [image_rows, image_cols] = size(image_src);
  
% create the new filter image
  filter_image = zeros(image_rows, image_cols);
  filter_image(1:radius*2+1,1:radius*2+1) = gog_filter;
  
% shift the filter to center the kernel according to the slides
% visualization
  filter_shifted = circshift(filter_image, [-radius, -radius]);
  
% apply fft2 to image source & created filter image
  image_fft = fft2(image_src);
  filter_fft = fft2(filter_shifted);

% multiply filtered images
  result = image_fft.*filter_fft;

% inverse fft2 the image
  smooth_img = ifft2(result);

end


%-------------------------------------------------------------------------
% Function for a self-made 2-dimensioanl gaussian derivative filter.
% Input: sigma, radius
%   sigma = 1.0 leads to a 7-element filter
% Output: Filter kernel, radius
function [filter, r] = Create_GoG_Kernel(sigma)

  r = round(3*sigma)
  x_coord = -r:r;
  y_coord=(-r:r).';
  n = numel(x_coord);
  
  x = repmat(x_coord,n,1);
  y = repmat(y_coord,1,n);
 
  left = 1 / (2 * pi * sigma^2);
  right = exp( - (x.^2 + y.^2) ./ (2.0*sigma^2));
  
  filter = left.*right;
end