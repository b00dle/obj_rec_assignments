function task_a

% just close the windows please
close all
% thanks

%% Task B.a, b
% Read the image as grey value image with double precision
img_train_gr = load_image_grey('trainingB.png');
thresh = graythresh(img_train_gr);
img_train_bw = im2bw(img_train_gr, thresh);

figure('Name', 'BW Image');
imshow(img_train_bw);
title('Training Binary Image');

%% Task B.c
% Build a Fourier-descriptor & extract the boundaries
bnd = bwboundaries(img_train_bw);
D_f = build_fourier_descriptor(bnd{1}, 24);

%% Task B.d, e, f
% Find the shape in both test images
t_img_1gr = load_image_grey('test1B.jpg');
thresh = graythresh(t_img_1gr);
t_img_1bw = im2bw(t_img_1gr, thresh);

%% find trained shape in test image 1
bt1 = find_shape_in_image(D_f, t_img_1bw, 0.2, 24);

figure('Name', 'Test1 Results');
imshow(t_img_1bw); hold on;
plot(bt1(:,2), bt1(:,1), 'r', 'LineWidth', 2);
title('found object in test image 1');


%% find trained shape in test image 2
t_img_2gr = load_image_grey('test2B.jpg');
t_img_2bw = im2bw(t_img_2gr, 0.27);

bt2 = find_shape_in_image(D_f, t_img_2bw, 0.8, 24);

figure('Name', 'Test1 Results');
imshow(t_img_2bw); hold on;
plot(bt2(:,2), bt2(:,1), 'r', 'LineWidth', 2);
title('found object in test image 1');


end

% -------------------------------------------------------------------------
function boundaries = find_shape_in_image(descriptor, binary_image, threshold, num_features)
% find the desired shape in an image using a fourier descriptor

b = bwboundaries(binary_image);

boundaries = [];
for i = 1:length(b)
  
  % get the current element and build a fourier descriptor
  cur = b{i};
  cur_Df = build_fourier_descriptor(cur, num_features);
  
  % calculate the euclidean distance of the current boundary descriptor and
  % the trained one
  d = norm(cur_Df - descriptor);
  
  % if the distance is smaller than the given threshold return the current
  % boundaries, else just return an empty object
  if d < threshold
    boundaries = cur;
  end
  
end

end
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
function D_f = build_fourier_descriptor(boundary_array, num_features)
% builds a fourier descriptor for a certain training image
%  binary_image: bw input image

% build the complex vector
D = zeros(1,length(boundary_array));
for i = 1:length(boundary_array)
  cur = boundary_array(i,:);
  D(i) = cur(1,2) + 1j * cur(1,1);
end
D = D';

% DFT on D
D_f = fft(D);

% extract the first 24 elements with the lowest frequency
% if there are less than 24 elements pad the remaining space with zeroes
if length(D_f) < num_features
  D_f = D_f(1:length(D_f),1);
  D_f(length(D_f):num_features,1) = 0;
else
  D_f = D_f(1:num_features,1);
end

% remove first element representing the shapes centroid
D_f = D_f(2:(length(D_f)),1);

% normalize D_f
D_f = D_f / abs(D_f(1));

% remove orientation encoded in phase information
D_f = abs(D_f);

end
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
function image_grey = load_image_grey(filename)
% loads the given image and returns a binary mask
img = imread(filename);
image_grey = im2double(rgb2gray(img));
end
% -------------------------------------------------------------------------



















