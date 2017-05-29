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

function [H, o_idx_vec, p_idx_vec] = hough_voting_array(img_thresh)
  [nr, nc] = size(img_thresh);
  p_max = int32(sqrt(nr^2 + nc^2));
  H = zeros(2*p_max + 1, 180);
  p_idx_vec = 1:2*p_max+1;
  o_idx_vec = 1:180;
  
  for x = 1:nc
    for y = 1:nr
      # check if pixel is edge point
      if img_thresh(y, x)
        for o_i = 1:180
          o = o_i - 91;
          p = x*cos(o) + y*sin(o);
          p_i = int32(p)+p_max;
          H(p_i, o_i) += 1;
        end          
      end
    end  
  end

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
# H = hough_voting_array(img_bw);

[H,theta,rho] = hough(img_bw);






## Task A f)
figure("Name", "Hough voting array");
image(H);
# pbaspect([1 1 1]);
axis equal;
axis image;

## Task A g)
# 
# source("houghpeaks_octave.m");

# [r, c, hnew] = houghpeaks(H, numpeaks, threshold, nhood)
[r, c, h_peaks] = houghpeaks(H);

figure("Name", "Houghpeaks Result");
imshow(h_peaks);

#[H, R] = hough_line( I, angles)
#[H_lines, R_lines] = hough_line( img_bw);
lines = houghlines(img_bw);

figure, imshow(img_grey), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
