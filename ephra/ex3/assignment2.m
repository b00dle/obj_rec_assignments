
% sample solution for exercise 2 in the course 
% "Image Analysis and Object Recognition"

% Input: An imagefile with one or more channels

% The code coducts the following tasks:
% - Applys GoG filtering on input image
% - Computes region candidates for foerstner interest points (not the points)
% - Vusually compares the identified foerstner regions to harris points

function foerstner

    close all;
    % dialog for input image selection
    %----------------------------------
    [file, path, Image, s_image] = read_image('Select image file');
    
    f = figure('name', 'Result of Foerstnter operator'); 
    subplot(1,4,1); imshow(Image, []); title('Original image');

%----------------------------------------------------------------------

    sigma = 0.5;
    % apply GoG filtering
    [GoG_x, GoG_y] = GoG_filtering(Image, sigma);

%    f = figure('name', 'GoG Results'); 
%    subplot(1,2,1); imshow(GoG_x,[]); title('GoG x');
%    subplot(1,2,2); imshow(GoG_y,[]); title('GoG y');
%    f = figure('name', 'Autocorrelation components'); 
%    subplot(1,3,1); imshow(GoG_x.^2,[]); title('Ix�');
%    subplot(1,3,2); imshow(GoG_y.^2,[]); title('Iy�');
%    subplot(1,3,3); imshow(GoG_y.*GoG_x,[]); title('Iyx');    
    
    Gradient_GoG = sqrt(GoG_x.^2 + GoG_y.^2);
    subplot(1,4,2); imshow(Gradient_GoG); title('GoG-output: Gradient magnitudes');

%----------------------------------------------------------------------

    % foerstner-operator
    radius = 2; % 2 --> 5x5 mask
    % threshold for cornerness
    w = 0.004; 
    % threshold for roundness
    q = 0.5; 
    [Interest_point_mask, W, Q] = foerstner_points(GoG_x, GoG_y, radius , w, q);

    %----------------------------------------------------------------------
    
    figure(f);
    subplot(1,4,3); imshow(Interest_point_mask); title('Result of Foerstner operator');
    
    % overlay of result and input rgb
    Image = mat2gray( imread([path,file]));
    
    % position of all interest points
    ind = find(Interest_point_mask );
    
    % mark these positions in the red channel as 1
    red = Image(:,:,1); % the red channel
    red(ind) = 1; % set all interest point candidates in red channel = 1
    Image(:,:,1) = red; % overwrite original image channel
    % set all other channels to zero at this position --> interest point
    % candidates will appear in red
    Image(:,:,2) = ~Interest_point_mask .* Image(:,:,2);
    Image(:,:,3) = ~Interest_point_mask .* Image(:,:,3);
    
    % plot the identified local maxima
    subplot(1,4,4); imshow(Image); title('Result of Foerstner operator, r: foerstner regions, y: local max');

    % identify local maxima
    peaks = imregionalmax(W.*Q);
    % get indices of maxima
    [peaks_row, peaks_col] = find(peaks);
    % plot them
    hold on
    plot(peaks_col, peaks_row, 'b+');
    hold off
end

%--------------------------------------------------------------------------
% Derivation of foerstner point candidates in an image
% Inputs: - Gradient of Gassians result in x- and y-direction
%         - radius of the window in which the values for autocorrelation matrix
%           are computed
%         - thresholds w_min and q_min
% Output: Binary mask with interest point CANDIDATES (not the points itself)
function [Interest_point_mask, w, q] = foerstner_points(GoG_x, GoG_y, r, w_min, q_min)

    % Values for autocorrelation matrix
    IxIx = GoG_x.^2; IyIy = GoG_y.^2; IxIy = GoG_x.*GoG_y;
    % imagesize
    s = size(GoG_x);
    
    w = GoG_x*0.0;
    q = GoG_x*0.0;
    w_klt = GoG_x*0.0;
    
    % loop over all pixels
    for i = (1+r):(s(1)-r);
        for j = (1+r):(s(2)-r);

            % Autocorrelation matrix M
            I_xx = sum(sum(IxIx(i-r:i+r,j-r:j+r)));
            I_yy = sum(sum(IyIy(i-r:i+r,j-r:j+r)));
            I_xy = sum(sum(IxIy(i-r:i+r,j-r:j+r)));
            M = [I_xx, I_xy; I_xy, I_yy];
        
            tr = trace(M);
            de = det(M);
            
            % area of error ellipse
            %w(i,j) = de / tr;
            w(i,j) = tr/2 - sqrt( (tr/2)^2 - de );
             
            % roundness of ellipse
            q(i,j) = (4.0*de) / (tr^2);
            
            % test inf or nan?
            if isnan( q(i,j) ) | isinf( q(i,j) )
                w(i,j) = 0.0; q(i,j) = 0.0;
            end
        end
    end
    
    % w > 0
    ind = find(w < 0);
    w(ind) = 0;

    % 0 < q < 1
    ind = find(q < 0);
    q(ind) = 0;
    ind = find(q > 1);
    q(ind) = 1;
    
    f = figure('name', 'Intermediate-Results of foerstnter operator'); 
    subplot(1,2,1); imshow(w); title('Result of w'); colormap(jet);
    subplot(1,2,2); imshow(q); title('Result of q'); colormap(jet);
    
    % get the binary mask for candidate regions
    Interest_point_mask = w > w_min & q > q_min;
    % set all other pixels to zewo in w and q
    w = w .* Interest_point_mask;
    q = q .* Interest_point_mask;
    
    f = figure('name', 'Maxima regions of w and q'); 
    subplot(1,2,1); imshow(w); title('Max regions in w'); colormap(jet);
    subplot(1,2,2); imshow(q); title('Max regions in q'); colormap(jet);
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
    n = numel(x_coord);
    #n = numberofelements(x_coord);
    
    x = repmat(x_coord,n,1);
    y = repmat(y_coord,1,n);

    filter_x = -(x)./(2.0*pi*sigma^4).*exp( -(x.^2 + y.^2)./(2.0*sigma^2) );
    filter_y = filter_x';
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
   

