% pkg load image;

function task_a
  %% Task A a)
  % read image and convert to greyscale
  [f, p, img_grey, shape] = read_image('Choose image');

  figure('Name','Grey Image');
  imshow(img_grey);

  %% Task A b)
  % Apply GoG_filtering (code reused from assignment 2)
  [GoG_x, GoG_y] = GoG_filtering(img_grey, 0.5);

  %% Task A c)
  % Compute the gradient magnitude
  img_mag = sqrt(GoG_x.^2 + GoG_y.^2);

  %% Task A d)
  % Convert magnitude image to binary image
  img_bw = im2bw(img_mag, 0.07);

  figure('Name','Binary Image with Tresh=0.07');
  imshow(img_bw);

  %% Task A e)
  % Hough Line Detection
  [H, R, T] = hough_voting_array(img_bw);

  %% Task A f)
  figure('Name', 'Hough Voting Array');
  imshow(H, [], 'XData', T, 'YData', R, 'InitialMagnification', 'fit');
  xlabel('\theta'), ylabel('\rho');
  axis on, axis normal;

  %% Task A g) & h)
  P = houghpeaks(H,32);
  figure('Name', 'Hough Voting Array and Peaks');
  imshow(H, [], 'XData', T, 'YData', R, 'InitialMagnification', 'fit');
  xlabel('\theta'), ylabel('\rho');
  axis on, axis normal, hold on;
  plot(T(P(:,2)), R(P(:,1)), 's', 'color', 'white');

  %% Task A i) & j)
  lines = houghlines(img_bw, T, R, P, 'FillGap', 5, 'MinLength', 7);
  figure('Name', 'Hough lines on grey scale image.'), imshow(img_grey), hold on
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
end


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

function [H, R, T] = hough_voting_array(img_thresh)
  [nr, nc] = size(img_thresh);
  r_max = int32(sqrt(nr^2 + nc^2));
  H = zeros(2*r_max + 1, 180);
  R = double(-r_max:r_max);
  T = -90:89;
  
  for x = 1:nc
    for y = 1:nr
      % check if pixel is edge point
      if img_thresh(y, x)
        for t_i = 1:180
          t = t_i - 91;
          t_rad = degtorad(t);
          r = x*cos(t_rad) + y*sin(t_rad);
          r_i = int32(r)+r_max;
          H(r_i, t_i) = H(r_i, t_i) + 1;
        end          
      end
    end  
  end

end

%--------------------------------------------------------------------------
% function for GoG filtering
% Inputs: Image, sigma
% Outputs: Filter results in x- and y-direction (GoG_x, GoG_y)
function [GoG_x, GoG_y] = GoG_filtering(I, sigma)

    % get filter masks and radius of filter
    [ filter_x, filter_y, r] = GoG_filter(sigma);
    
    % size of image
    s = size(I);
    
    % result arrays
    GoG_x = I*0.0;
    GoG_y = I*0.0;
    
    % loop over each pixel (except the edges)
    for i = r+1:(s(1)-r);
        for j = r+1:(s(2)-r);
            
            % current image 'chip'
            chip = I(i-r:i+r,j-r:j+r);
            
            % store filter outputs
            GoG_x(i,j) = sum(sum(filter_x .* chip));
            GoG_y(i,j) = sum(sum(filter_y .* chip));
        end
    end
end

%-------------------------------------------------------------------------
% Function for a self-made 2-dimensioanl gaussian derivative filter.
% Input: sigma
%   sigma = 1.0 leads to a 7-element filter
% Outputs: Filter kernel in x- and y-direction (filter_x, filter_y), and
%          radius r of the filter kernel
function [filter_x, filter_y, r] = GoG_filter(sigma)
    % --- Calculate its size and build filter
    r = round(3*sigma); x_coord = -r:r; y_coord = (-r:r).'; 
    n = numel(x_coord);
    %n = numberofelements(x_coord);
    
    x = repmat(x_coord,n,1);
    y = repmat(y_coord,1,n);

    filter_x = -(x)./(2.0*pi*sigma^4).*exp( -(x.^2 + y.^2)./(2.0*sigma^2) );
    filter_y = filter_x';
end
