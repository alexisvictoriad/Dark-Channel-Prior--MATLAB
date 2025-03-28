function J = dehaze_parallel(I, omega, t0, win_size)
    % I: Hazy input image
    % omega: Dark channel prior parameter (typically 0.95)
    % t0: Minimum transmission value (typically 0.1)
    % win_size: Window size for dark channel (typically 15)

    % Convert the image to double
    I = im2double(I);

    % Estimate the atmospheric light
    A = estimate_atmospheric_light(I);

    % Estimate the transmission map
    t = estimate_transmission_parallel(I, A, omega, win_size);

    % Refine the transmission map using guided filter
    t = guided_filter(I, t, win_size);

    % Recover the scene radiance
    J = recover_radiance(I, t, A, t0);
end

function A = estimate_atmospheric_light(I)
    % Estimate the atmospheric light in the image
    dark_channel = min(I, [], 3);
    [~, idx] = max(dark_channel(:));
    [i, j] = ind2sub(size(dark_channel), idx);
    A = I(i, j, :);
end

function t = estimate_transmission_parallel(I, A, omega, win_size)
    % Estimate the transmission map using parallel processing
    norm_I = bsxfun(@rdivide, I, A);
    dark_channel = min(norm_I, [], 3);
    
    % Initialize the transmission map
    t = zeros(size(dark_channel));
    
    % Precompute the indices for the local patches
    [rows, cols] = size(dark_channel);
    row_indices = cell(rows, 1);
    col_indices = cell(cols, 1);
    
    for i = 1:rows
        row_indices{i} = max(1, i-win_size):min(rows, i+win_size);
    end
    
    for j = 1:cols
        col_indices{j} = max(1, j-win_size):min(cols, j+win_size);
    end
    
    % Parallelize the dark channel computation
    parfor i = 1:rows
        for j = 1:cols
            local_patch = dark_channel(row_indices{i}, col_indices{j});
            t(i, j) = 1 - omega * min(local_patch(:));
        end
    end
end

function t_refined = guided_filter(I, t, win_size)
    % Refine the transmission map using guided filter
    r = win_size;
    eps = 0.001;
    t_refined = imguidedfilter(t, I, 'NeighborhoodSize', [r r], 'DegreeOfSmoothing', eps);
end

function J = recover_radiance(I, t, A, t0)
    % Recover the scene radiance
    t = max(t, t0);
    J = bsxfun(@minus, I, A);
    J = bsxfun(@rdivide, J, t);
    J = bsxfun(@plus, J, A);
end

% Load the hazy image
I = imread('HazyImage.jpg');

% Set parameters
omega = 0.95;  % Dark channel prior parameter
t0 = 0.1;     % Minimum transmission value
win_size = 15; % Window size for dark channel

% Call the parallel dehaze function
J = dehaze_parallel(I, omega, t0, win_size);

% Display the results
figure;
subplot(1, 2, 1);
imshow(I);
title('Hazy Image');

subplot(1, 2, 2);
imshow(J);
title('Dehazed Image');