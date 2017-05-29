function test
    main('input_ex3.jpg');
end

function [ img, s ] = img_read( text )

    [file, path] = uigetfile('*.*', text);
 
    % read training image
    img = imread([path,file]);
    
    % convert image to grayscale image [0,...,1]
    img = mat2gray(mean( img, 3 ));
    s = size(img);

end

function [H, o_idx_vec, p_idx_vec] = hough_voting_array(img_thresh)
  [nr, nc] = size(img_thresh);
  p_max = int32(sqrt(nr^2 + nc^2));
  H = zeros(2*p_max + 1, 180);
  p_idx_vec = 1:2*double(p_max)+1;
  o_idx_vec = 1:180;
  
  for x = 1:nc
    for y = 1:nr
      % check if pixel is edge point
      %if img_thresh(y, x)
        for o_i = 1:180
          o = o_i - 91;
          p = x*cos(o) + y*sin(o);
          p_i = int32(p)+p_max;
          H(p_i, o_i) = H(p_i, o_i) + 1;
        end          
      %end
    end  
  end
end

function y = deg2rad(x)
 y = x * pi/180;
end

function [H, o_idx_vec, p_idx_vec] = hough_ephra(img_thresh)
  [nr, nc] = size(img_thresh);
  p_max = int32(sqrt(nr^2 + nc^2));
  disp(p_max);
  H = zeros(2*p_max + 1, 180);
  p_idx_vec = 1:2*double(p_max);
  o_idx_vec = 1:180;
  
  for x = 1:nc
    for y = 1:nr
      % check if pixel is edge point
      if img_thresh(y, x)
        for o_i = 1:180
          o = o_i - 91;
          rad_o = deg2rad(o);
          p = x*cos(rad_o) + y*sin(rad_o);
          p_i = int32(p)+p_max;
          H(p_i, o_i) = H(p_i, o_i) + 1;
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
            
            % current image "chip"
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
    %n = numel(x_coord);
    n = numel(x_coord);
    
    x = repmat(x_coord,n,1);
    y = repmat(y_coord,1,n);

    filter_x = -(x)./(2.0*pi*sigma^4).*exp( -(x.^2 + y.^2)./(2.0*sigma^2) );
    filter_y = filter_x';
end

function [r, c, hnew] = houghpeaks_uni(h, numpeaks, threshold, nhood)
    %HOUGHPEAKS Detect peaks in Hough transform.
    %   [R, C, HNEW] = HOUGHPEAKS(H, NUMPEAKS, THRESHOLD, NHOOD) detects
    %   peaks in the Hough transform matrix H.  NUMPEAKS specifies the
    %   maximum number of peak locations to look for.  Values of H below
    %   THRESHOLD will not be considered to be peaks.  NHOOD is a
    %   two-element vector specifying the size of the suppression
    %   neighborhood.  This is the neighborhood around each peak that is
    %   set to zero after the peak is identified.  The elements of NHOOD
    %   must be positive, odd integers.  R and C are the row and column
    %   coordinates of the identified peaks.  HNEW is the Hough transform
    %   with peak neighborhood suppressed. 
    %
    %   If NHOOD is omitted, it defaults to the smallest odd values >=
    %   size(H)/50.  If THRESHOLD is omitted, it defaults to
    %   0.5*max(H(:)).  If NUMPEAKS is omitted, it defaults to 1. 

    %   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
    %   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
    %   $Revision: 1.5 $  $Date: 2003/11/21 13:34:50 $
    disp('HELLOO');
    if nargin < 4
       nhood = size(h)/50;
       % Make sure the neighborhood size is odd.
       nhood = max(2*ceil(nhood/2) + 1, 1);
    end
    if nargin < 3
       threshold = 0.5 * max(h(:));
    end
    if nargin < 2
       numpeaks = 1;
    end

    done = false;
    hnew = h; r = []; c = [];
    while ~done
       [p, q] = find(hnew == max(hnew(:)));
       p = p(1); q = q(1);
       if hnew(p, q) >= threshold
          r(end + 1) = p; c(end + 1) = q;

          % Suppress this maximum and its close neighbors.
          p1 = p - (nhood(1) - 1)/2; p2 = p + (nhood(1) - 1)/2;
          q1 = q - (nhood(2) - 1)/2; q2 = q + (nhood(2) - 1)/2;
          [pp, qq] = ndgrid(p1:p2,q1:q2);
          pp = pp(:); qq = qq(:);

          % Throw away neighbor coordinates that are out of bounds in
          % the rho direction.
          badrho = find((pp < 1) | (pp > size(h, 1)));
          pp(badrho) = []; qq(badrho) = [];

          % For coordinates that are out of bounds in the theta
          % direction, we want to consider that H is antisymmetric
          % along the rho axis for theta = +/- 90 degrees.
          theta_too_low = find(qq < 1);
          qq(theta_too_low) = size(h, 2) + qq(theta_too_low);
          pp(theta_too_low) = size(h, 1) - pp(theta_too_low) + 1;
          theta_too_high = find(qq > size(h, 2));
          qq(theta_too_high) = qq(theta_too_high) - size(h, 2);
          pp(theta_too_high) = size(h, 1) - pp(theta_too_high) + 1;

          % Convert to linear indices to zero out all the values.
          hnew(sub2ind(size(hnew), pp, qq)) = 0;

          done = length(r) == numpeaks;
       else
          done = true;
       end
    end
end






function [result] = main( filepath )
    [img_grey, s] = img_read(filepath);
    disp(s);
    [GoG_x, GoG_y] = GoG_filtering(img_grey, 0.5);

    %% Task A c)
    % Compute the gradient magnitude
    img_mag = sqrt(GoG_x.^2 + GoG_y.^2);

    %figure("Name","Magnitude");
    %imshow(img_mag);

    %% Task A d)
    % Convert magnitude image to binary image
    img_bw = im2bw(img_mag, 0.07);

    figure('Name','Binary Image with Tresh=0.07');
    imshow(img_bw);

    [H,theta,rho] = hough_ephra(img_bw);
    disp('2 Theta');
    disp(size(theta ));
    disp('2 rho');
    disp(size(rho));

    
   

    %% Task A f)
    figure('Name', 'Hough voting array');
    image(H);
    axis equal;
    axis image;
    
    h_peaks = houghpeaks(H,1);
    figure('Name', 'Houghpeaks Result');
    imshow(h_peaks);
    disp(size(h_peaks));
    %disp(h_peaks);
    
    lines = houghlines(img_bw,theta, rho, h_peaks);

    figure('Name', 'Houghlines Result');
    imshow(img_grey), hold on
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


    result = H;


end

