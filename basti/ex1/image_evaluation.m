%% Assignment 1 - Image analysis and object recognition
% Ephraim Schott 110960
% Hagen Hiller 110514
% Sebastian Stickert 110882

function image_evaluation(image_path)
  % load image package
%   pkg load image;
  
  % load image, plot and show histogram
  I = load_greyscale(image_path);
  figure('Name', 'original image');
  subplot(2,1,1);
  imshow(I);
  subplot(2,1,2);
  imhist(I);
  %% A) b. NOTES TO HISTOGRAM:
  % - spectrum of color values very narrow
  % - mostly values lie between 150 - 215 of intesity 
  % - global maximum is approximately 160
  
  % stretch coontrast and plot result
  I_stretched = stretch_contrast(I);
  figure('Name', 'contrast stretched image')
  subplot(2,1,1);
  imshow(I_stretched);
  subplot(2,1,2);
  imhist(I_stretched);
  %% A) d. CHANGES NOTED
  % - the spectrum of intensity values has been enhanced
  % - histogram stretched
  % - image now makes full use of possible color encodings
  % - details pop out more and appearance is generally more crisp
  
  % convert to bw mask
  I_bw = ~apply_threshold(I_stretched, 0.35);
  figure('Name', 'bw-mask');
  imshow(I_bw);
  %% B) c. TOWARDS DIFFERENT THRESHOLDS
  % - very low values ([0, 0.1]) tend to convert most features to white
  %   color
  % - very high values ([0.9, 1.0]) convert most features to black color
  % - a mid ranged value ([0.45, 0.55]) shows a fairly evenly distributed 
  %   amount of black and white colors
  % - Regardless of the threshold chosen, the multiple small features
  %   located in the areas around the lake, seem to have a partially 
  %   overlapping intensity value range. Thus excluding coloring for these
  %   areas, while uniformly coloring the lake area is impossible using 
  %   only linear black-white conversion
  % - additionally we had to invert the resulting image mask to produce a 
  %   result comparable to the example shown in the exercise slides 
  %   (see slide 18)
  % - This is due to the fact that, for threshold values chosen to result
  %   the lake in white color, most surrounding areas will be converted to
  %   white as well.
  % - any approaches of using the graythresh function to automatically 
  %   compute a threshold did not produce maximally useful masks
  
  % sucessive opening and closing
  I_bw_mod = open_close_successive(I_bw, 3);
  figure('name', 'image after successive open close');
  imshow(I_bw_mod);

  % self written erosion function
  I_bw_erode = erode(I_bw);
  figure('name', 'erode image');
  imshow(I_bw_erode);
  
  % in-built erosion
  I_bw_erode_alt = erode_alt(I_bw);
  figure('name', 'in-built erosion'), imshow(I_bw_erode_alt);
  
  % comparison of built-in erosion and self written by subtracting the
  % images from each other to get the difference
  difference = I_bw_erode - I_bw_erode_alt;
  figure('Name', 'image differences from built-in to self written');
  imshow(difference);
  
  % count the pixel difference
  diff_num = count_positive_elements(difference);
  disp(diff_num);
  
  %% C.d Are there differences in the results? Why?
  % Since the self-implemented approach is very naive there area 56307
  % pixel difference to the matlab implemented functions result. This could
  % result because of further optimization within matlabs function. The
  % self written function has no variable size of the structuring element
  % and is fixed to a size of 1 pixel surrounding the current images pixel.
  % Furthermore the use of just local information could be enhanced in
  % order to find and eliminate outliers.
  

  % in-built dilation
  %   I_bw_dilate_alt = dilate_alt(I_bw);
  %   figure(8), imshow(I_bw_dilate_alt);

  % overlay computation
  I_bw_gray = uint8(I_bw_erode*255);
  I_combined = imadd(I, I_bw_gray);
  figure(9), imshow(I_combined);
  
  %% E. Are the results satisfactory? 
  % Regarding the self written erosion function it is easy to see,
  % that we have a lot of artifacts in our eroded image.
  % These artefacts occur in the river and in the city.
  % Comparing our results to the in-built erosion function we can clearly
  % see that it sepererated fore- and background with less artifacts.
  %
  % What are the limitations of this approach for
  % separating back- and foreground?
  % The seperating approach has problems with dark roofs, as they appear in the
  % same color as the river.
  % When we try to seperate fore- and background, we do this by defining 
  % a threshold. When the colors in the fore- and background 'overlap', we will 
  % have artifacts as a result. These artifacts could be minimized to some 
  % degree by eroding and dilationing.
  
end

%--------------------------------------------------------------------

function image = load_greyscale(image_path)

  % how to include mean??
  image = rgb2gray(imread(image_path));
  
end

%--------------------------------------------------------------------

% stretches the contrast of the input image along some thresholds
function new_image = stretch_contrast(image_src)

    min_value = double(min(image_src(:)));
    max_value = double(max(image_src(:)));
    
    s = size(image_src);
    
    result = zeros(s(1), s(2));
    % calculcate contrast ratio for each pixel
    for i = 1:s(1) % loop over x
        for j = 1:s(2) % loop over y
          pixel_value = (double(image_src(i,j)) - min_value) ...
            / (max_value - min_value);                    
          result(i,j) = pixel_value;
        end
    end
   
    new_image = result;

end

%--------------------------------------------------------------------

% Alternative contrast stretching function.
% Uses in-built implementation.
function new_image = stretch_contrast_alt(image_src)

  new_image = imadjust(image_src, stretchlim(image_src),[]);

end

%--------------------------------------------------------------------

function bw_image = apply_threshold(image_src, value)

  bw_image = im2bw (image_src, im2double(value));
  % automatic computation of the threshold value
  % bw_image = im2bw (image_src, graythresh (image_src(:), 'concavity'));

end

%--------------------------------------------------------------------

function mod_image = open_close_successive(image_src, times)

  % Create a disk-shaped structuring element with a radius of 5 pixels.
  % (see https://de.mathworks.com/help/images/ref/imopen.html)
  se = strel('disk', 5, 0);
  
  mod_image = image_src;
  for i = 1:times
    mod_image = imclose(imopen(mod_image, se), se);
  end 
  
end

%--------------------------------------------------------------------

% erode current image mask

function mod_image = erode(image_src)

  % get the images size
  s = size(image_src);
  
  % preallocate memery for the output image
  output = zeros(s(1),s(2));
  
  % loop over the image matrix
  for r = 2:s(1)-1 % loop over x
    for c = 2:s(2)-1 % loop over y
      
      % get the current structuring element and check whether its valid
      s_element = image_src(r-1:r+1, c-1:c+1);      
      if check_erode_structuring_element(s_element)
        output(r,c) = 1;
      end
      
    end
  end
  
  mod_image = output;

end

%--------------------------------------------------------------------


% helper function checking the center-pixel surrounding structure element
function is_valid = check_erode_structuring_element(matrix_src)

  % get matrix size
  s = size(matrix_src);

  % iterate over the matrix and sum up all center-surrounding elements
  res = 0;
  for r = 1:s(1)
    for c = 1:s(2)
      if (r ~= 2) || (c ~= 2)
        res = res + matrix_src(r,c);
      end

    end
  end
  
  % if all elements are true, return true for the current structuring
  % element
  if res == 8
    is_valid = true;
  else
    is_valid = false;
  end

end

%--------------------------------------------------------------------

% utilizes in-built erosion function (see imerode)
function mod_image = erode_alt(image_src)
  
  % Create a disk-shaped structuring element with a radius of 5 pixels.
  % (see https://de.mathworks.com/help/images/ref/imopen.html)
  se = strel('disk', 5, 0);
  mod_image = imerode(image_src, se);

end

%--------------------------------------------------------------------

%count number of positive (1) elements in a matrix
function count = count_positive_elements(image_src)
  
  % get matrix sizes
  s = size(image_src);

  % iterate over the matrix and sum up all center-surrounding elements
  res = 0;
  for r = 1:s(1)
    for c = 1:s(2)
      if image_src(r,c) == 1
        res = res + 1;
      end
    end
  end
  
  count = res;

end

%--------------------------------------------------------------------

% utilizes in-built erosion function (see imdilate)
function mod_image = dilate_alt(image_src)
  
  % Create a disk-shaped structuring element with a radius of 5 pixels.
  % (see https://de.mathworks.com/help/images/ref/imopen.html)
  se = strel('disk', 5, 0);
  mod_image = imdilate(image_src, se);
  
end

