% Lab 1: Image as a 2D Signal (Sampling, Quantization, Histograms, Enhancement)
%
% This toolbox-free version avoids imhist, montage, imshow, imadjust,
% imresize, rgb2gray, and mat2gray. It uses base MATLAB only.

close all; clear; clc;

%% 0) Load and inspect an image
if exist('peppers.png','file')
    I_rgb = imread('peppers.png');
elseif exist('cameraman.tif','file')
    I_rgb = repmat(imread('cameraman.tif'),1,1,3); % fallback
else
    imgs = [dir('*.png'); dir('*.jpg'); dir('*.jpeg'); dir('*.tif'); dir('*.bmp')];
    if ~isempty(imgs)
        I_rgb = imread(imgs(1).name);
        if size(I_rgb,3)==1, I_rgb = repmat(I_rgb,1,1,3); end
    else
        error('Could not find peppers.png or cameraman.tif. Please place an image in the folder.');
    end
end

figure; showrgb(I_rgb); title('Original RGB');

% Convert to grayscale without rgb2gray
if size(I_rgb,3)==3
    I = to_grayscale_uint8(I_rgb);
else
    I = I_rgb;
    if ~isa(I,'uint8'), I = im2uint8_local(I); end
end

figure; showgray(I); title('Grayscale');

% Basic info
fprintf('Class: %s | Range: [%g, %g] | Size: %d x %d\n', ...
    class(I), double(min(I(:))), double(max(I(:))), size(I,1), size(I,2));

%% 1) Quantization and dynamic range
I8 = I;                             % 8-bit (0..255)
I6 = uint8(floor(double(I)/4)*4);   % ~6 bits (step 4)
I4 = uint8(floor(double(I)/16)*16); % ~4 bits (step 16)

figure;
subplot(1,3,1); showgray(I8); title('8-bit');
subplot(1,3,2); showgray(I6); title('~6-bit');
subplot(1,3,3); showgray(I4); title('~4-bit');

%% 2) Histogram and contrast stretching
figure;
subplot(1,2,1);
plot_hist_uint8(I);
title('Histogram - original');

% Normalize to [0,1] like mat2gray
I_norm = mat2unit_local(I); % double in [0,1]

% Contrast stretching equivalent to imadjust(I,[0.2 0.8],[0 1])
I_stretch = linear_stretch(I, 0.2, 0.8); % double in [0,1]

subplot(1,2,2);
plot_hist_uint8(im2uint8_local(I_stretch));
title('Histogram - stretched');

figure;
subplot(1,3,1); showgray(I);         title('Original');
subplot(1,3,2); showgray(I_norm);    title('Normalized [0,1]');
subplot(1,3,3); showgray(I_stretch); title('Contrast-stretched');

%% 3) Gamma correction - nonlinear amplitude scaling
% gamma < 1 brightens, gamma > 1 darkens
I_gamma_low  = gamma_adjust(I, 0.6); % double in [0,1]
I_gamma_high = gamma_adjust(I, 1.6); % double in [0,1]

figure;
subplot(1,3,1); showgray(I);             title('Original');
subplot(1,3,2); showgray(I_gamma_low);   title('Gamma = 0.6');
subplot(1,3,3); showgray(I_gamma_high);  title('Gamma = 1.6');

%% 4) Sampling and aliasing - downsample then upsample
scale   = 0.1; % 10 percent of size
I_small = resize_nearest_scale(I, scale);                      % nearest neighbor downsample
I_back  = resize_nearest_to_size(I_small, size(I,1), size(I,2)); % nearest neighbor upsample

figure;
subplot(1,3,1); showgray(I);       title('Original');
subplot(1,3,2); showgray(I_small); title('Downsampled 10 percent');
subplot(1,3,3); showgray(I_back);  title('Upscaled back - aliasing artifacts');

%% 5) OPTIONAL: Moire demo
% If you have a striped or high-frequency texture image, repeat Section 4
% and observe interference patterns - moire.

%% 6) Short reflections - for your README or report
% 1) Relate bit depth to visible banding or posterization you observed.
% 2) How contrast stretching changes the histogram and the visibility of details.
% 3) Why aggressive downsampling causes aliasing. Reference Nyquist.

%% -------- Local helper functions --------
function Iu8 = im2uint8_local(I)
    % Convert numeric array to uint8, scaling if needed
    if isa(I,'uint8')
        Iu8 = I;
    else
        I = double(I);
        if max(I(:)) <= 1 && min(I(:)) >= 0
            Iu8 = uint8(round(I*255));
        else
            mn = min(I(:)); mx = max(I(:));
            if mx == mn
                Iu8 = uint8(zeros(size(I)));
            else
                Iu8 = uint8(round((I - mn) * 255 / (mx - mn)));
            end
        end
    end
end

function G = to_grayscale_uint8(RGB)
    % Rec. 601 luma coefficients
    RGBd = im2double_local(RGB);
    Gd = 0.2989*RGBd(:,:,1) + 0.5870*RGBd(:,:,2) + 0.1140*RGBd(:,:,3);
    G = uint8(round(255*clip01(Gd)));
end

function Id = im2double_local(I)
    if isa(I,'double')
        Id = I;
    elseif isa(I,'single')
        Id = double(I);
    elseif isa(I,'uint8')
        Id = double(I)/255;
    elseif isa(I,'uint16')
        Id = double(I)/65535;
    else
        Id = double(I);
        mn = min(Id(:)); mx = max(Id(:));
        if mx == mn
            Id = zeros(size(I));
        else
            Id = (Id - mn) / (mx - mn);
        end
    end
end

function J = mat2unit_local(I)
    I = double(I);
    mn = min(I(:)); mx = max(I(:));
    if mx == mn
        J = zeros(size(I));
    else
        J = (I - mn) / (mx - mn);
    end
end

function J = linear_stretch(Iu8, low, high)
    % Equivalent to imadjust(I,[low high],[0 1]) producing double in [0,1]
    X = double(Iu8)/255;
    J = (X - low) / (high - low);
    J = clip01(J);
end

function J = gamma_adjust(Iu8, g)
    X = double(Iu8)/255;
    J = clip01(X).^g; % double in [0,1]
end

function Y = clip01(X)
    Y = min(max(X,0),1);
end

function plot_hist_uint8(I)
    % Plot histogram for uint8 or double image by converting to uint8
    if ~isa(I,'uint8')
        I = im2uint8_local(I);
    end
    counts = histcounts(I(:), 0:256);
    bar(0:255, counts, 1, 'EdgeColor', 'none');
    xlim([0 255]);
    xlabel('Intensity'); ylabel('Count');
end

function showgray(A)
    % Display grayscale using base MATLAB
    if size(A,3) == 3
        A = to_grayscale_uint8(A);
    end
    imagesc(A);
    axis image off; colormap gray;
end

function showrgb(A)
    % Display RGB using base MATLAB
    if size(A,3) == 1
        showgray(A);
        return;
    end
    image(A); % base MATLAB truecolor display
    axis image off;
end

function B = resize_nearest_scale(A, scale)
    % Nearest neighbor resize by a scale factor without imresize
    if scale <= 0, error('Scale must be positive'); end
    H = size(A,1); W = size(A,2);
    Hn = max(1, round(H*scale));
    Wn = max(1, round(W*scale));
    B = resize_nearest_to_size(A, Hn, Wn);
end

function B = resize_nearest_to_size(A, Hn, Wn)
    % Nearest neighbor resize to target size without imresize
    H = size(A,1); W = size(A,2);
    xi = round(linspace(1, W, Wn)); xi(xi<1)=1; xi(xi>W)=W;
    yi = round(linspace(1, H, Hn)); yi(yi<1)=1; yi(yi>H)=H;
    if ndims(A)==2
        B = A(yi, xi);
    else
        C = size(A,3);
        B = zeros(Hn, Wn, C, class(A));
        for c = 1:C
            B(:,:,c) = A(yi, xi, c);
        end
    end
end
