clear; clc; close all;

%% Hard coded parameters
ratio = 0.64;
cropRatio = 0.6;
targetSize = 512;
marginSize = 72;
lambda_d = 1;
lambda_n = 1;
lambda_r = 0.5;
startPoint = 1;
endPoint = 350;

%% Add path
addpath(genpath('./Bk_matlab') );

%% Load the data
root = '/media/zhl/DATA/BRDFOriginDataset';
matDirs = dir(fullfile(root, 'Material__*') );

for n = startPoint : min(length(matDirs ), endPoint)
    matDir = matDirs(n).name;
    matDir = fullfile(root, matDir);
    
    fprintf('%d/%d: %s\n', n, min(length(matDirs ), endPoint ), matDir );
    if exist(fullfile(matDir, 'diffuse.jpg'), 'file')
        diffuseName = fullfile(matDir, 'diffuse.jpg');
        normalName = fullfile(matDir, 'normal.jpg');
        roughName = fullfile(matDir, 'rough.jpg');
        
        diffuseImOrig = imread(diffuseName );
        normalImOrig = imread(normalName );
        roughImOrig = imread(roughName );
        
        if length(size(roughImOrig ) ) == 2
            roughImOrig = cat(3, roughImOrig, roughImOrig, roughImOrig);
        end
           
        imSize = int32(targetSize / ratio);
        diffuseIm = imresize(diffuseImOrig, [imSize, imSize] );
        roughIm = imresize(roughImOrig, [imSize, imSize] );
        normalIm = imresize(normalImOrig, [imSize, imSize] );
        
        diffuseIm = single(diffuseIm) / 255.0;
        roughIm = single(roughIm ) / 255.0;
        normalIm = single(normalIm ) / 255.0;
        
        %% Compute the edge of the three maps
        [height, width, ~] = size(diffuseIm );
        diffuse_x = zeros(height, width);
        diffuse_y = zeros(height, width);
        for m = 1 : 3
            [diff_x, diff_y] = gradient(diffuseIm(:, :, m) );
            diffuse_x = diffuse_x + abs(diff_x);
            diffuse_y = diffuse_y + abs(diff_y);
        end
        
        normal_x = zeros(height, width);
        normal_y = zeros(height, width);
        for m = 1 : 3
            [norm_x, norm_y] = gradient(normalIm(:, :, m) );
            normal_x = normal_x + abs(norm_x);
            normal_y = diffuse_y + abs(diff_y);
        end
        
        rough_x = zeros(height, width);
        rough_y = zeros(height, width);
        for m = 1 : 3
            [r_x, r_y] = gradient(normalIm(:, :, m) );
            rough_x = rough_x + abs(r_x);
            rough_y = rough_y + abs(r_y);
        end
        
        %% Find the best cropping
        diffuse_x = imresize(imresize(diffuse_x, 0.125), 8);
        diffuse_y = imresize(imresize(diffuse_y, 0.125), 8);
        normal_x = imresize(imresize(normal_x, 0.125), 8);
        normal_y = imresize(imresize(normal_y, 0.125), 8);
        rough_x = imresize(imresize(rough_x, 0.125), 8);
        rough_y = imresize(imresize(rough_y, 0.125), 8);
        
        diffuse_x_cum = cumsum(diffuse_x, 1);
        diffuse_y_cum = cumsum(diffuse_y, 2);
        normal_x_cum = cumsum(normal_x, 1);
        normal_y_cum = cumsum(normal_y, 2);
        rough_x_cum = cumsum(rough_x, 1);
        rough_y_cum = cumsum(rough_y, 2);
        
        % Find the best path through integral graph
        minR = targetSize + marginSize;
        minC = targetSize + marginSize;
        maxR = height - marginSize;
        maxC = width - marginSize;
        
        minCost = 1e10;
        bestR = minR;
        bestC = minC;
        for r = minR : maxR
            for c = minC : maxC
                rr = r - targetSize+1;
                cc = c - targetSize+1;
                cost_xu = lambda_d * (diffuse_y_cum(rr, c) - diffuse_y_cum(rr, cc)) ...
                    + lambda_n * (normal_y_cum(rr, c) - normal_y_cum(rr, cc) ) ...
                    + lambda_r * (rough_y_cum(rr, c) - rough_y_cum(rr, cc) );
                cost_xd = lambda_d * (diffuse_y_cum(r, c) - diffuse_y_cum(r, cc) ) ...
                    + lambda_n * (normal_y_cum(r, c) - normal_y_cum(r, cc) ) ...
                    + lambda_r * (rough_y_cum(r, c) - rough_y_cum(r, cc) );
                cost_yl = lambda_d * (diffuse_x_cum(r, cc) - diffuse_x_cum(rr, cc) ) ...
                    + lambda_n * (normal_x_cum(r, cc) - normal_x_cum(rr, cc) ) ...
                    + lambda_r * (rough_x_cum(r, cc) - rough_x_cum(rr, cc) );
                cost_yr = lambda_d * (diffuse_x_cum(r, c) - diffuse_x_cum(rr, c) ) ...
                    + lambda_n * (normal_x_cum(r, c) - normal_x_cum(rr, c) ) ...
                    + lambda_r * (rough_x_cum(r, c) - rough_x_cum(rr, c) );
                cost = cost_xu + cost_xd + cost_yl + cost_yr;
                if cost < minCost
                    minCost = cost;
                    bestR = r;
                    bestC = c;
                end
            end
        end
        rs = bestR - targetSize - marginSize + 1;
        re = bestR + marginSize;
        cs = bestC - targetSize - marginSize + 1;
        ce = bestC + marginSize;
        
        diffuseCrop = diffuseIm(rs:re, cs:ce, :);
        normalCrop = normalIm(rs:re, cs:ce, :);
        roughCrop = roughIm(rs:re, cs:ce, :);
        [heightCrop, widthCrop, ~] = size(diffuseCrop );
        
        %% Determin the potential margin size
        minMarginSize = marginSize - int32(marginSize * cropRatio);
        maxMarginSize = marginSize + int32(marginSize * cropRatio);
        
        %% Graph cut to find the right and left seams
        minCost = 1e10;
        bestLabeling = nan;
        bestMarginSize = nan;
        for t = minMarginSize:maxMarginSize
            diffuse_1 = diffuseCrop(:, 1 : 2*t+1, :);
            diffuse_2 = diffuseCrop(:, end-2*t : widthCrop, :);
            normal_1 = normalCrop(:, 1 : 2*t+1, :);
            normal_2 = normalCrop(:, end-2*t : end, :);
            rough_1 = roughCrop(:, 1 : 2*t+1, :);
            rough_2 = roughCrop(:, end-2*t : end, :);
            
            % Compute the unary loss
            [rows, cols, ~] = size(diffuse_1 );
            cols = cols + 1;
            unaryLoss = zeros(2, rows, cols);
            unaryLoss(1, :, 1) = 1e10;
            unaryLoss(2, :, cols) = 1e10;
            
            % Compute the smoothness;
            [d_lr_12, d_lr_21, d_ud_12, d_ud_21] = computeSmooth_lr(diffuse_1, diffuse_2);
            [n_lr_12, n_lr_21, n_ud_12, n_ud_21] = computeSmooth_lr(normal_1, normal_2);
            [r_lr_12, r_lr_21, r_ud_12, r_ud_21] = computeSmooth_lr(rough_1, rough_2);
            
            s_lr_12 = lambda_d * d_lr_12 + lambda_n * n_lr_12 + lambda_r * r_lr_12;
            s_lr_21 = lambda_d * d_lr_21 + lambda_n * n_lr_21 + lambda_r * r_lr_21;
            s_ud_12 = lambda_d * d_ud_12 + lambda_n * n_ud_12 + lambda_r * r_ud_12;
            s_ud_21 = lambda_d * d_ud_21 + lambda_n * n_ud_21 + lambda_r * r_ud_21;
            
            % Build the graph
            graph = BK_Create(rows * cols );
            BK_SetUnary(graph, reshape(unaryLoss, [2, rows * cols]) );
            [x_grid, y_grid] = meshgrid(1:cols, 1:rows );
            xy_grid = (x_grid-1) * rows + y_grid;
            
            l_grid = xy_grid(:, 1:end-1);
            r_grid = xy_grid(:, 2:end);
            lr_grid = [l_grid(:), r_grid(:)];
            
            u_grid = xy_grid(1:end-1, 2:end-1);
            d_grid = xy_grid(2:end, 2:end-1);
            ud_grid = [u_grid(:), d_grid(:)];
            
            s_lr = [s_lr_12(:), s_lr_21(:)];
            s_ud = [s_ud_12(:), s_ud_21(:)];
            
            smoothMat = zeros(size(lr_grid, 1) + size(ud_grid, 1), 6);
            smoothMat(:, 1:2) = [lr_grid; ud_grid];
            smoothMat(:, 4:5) = [s_lr; s_ud];
            
            BK_SetPairwise(graph, smoothMat);
            cost = BK_Minimize(graph );
            labeling = BK_GetLabeling(graph );
            BK_Delete(graph );
            
            if cost < minCost
                minCost = cost;
                bestMarginSize = t;
                bestLabeling = labeling;
            end
        end
        
        labeling = reshape(bestLabeling, [heightCrop, bestMarginSize*2+2]);
        labeling = labeling(:, 2:end-1);
        labeling = cat(3, labeling, labeling, labeling) - 1;
        labeling = imfilter(single(labeling), fspecial('gaussian', [9, 9], 3), 'replicate', 'same');
        diffuse_1 = diffuseCrop(:, 1 : 2*bestMarginSize, :);
        diffuse_2 = diffuseCrop(:, end-2*bestMarginSize+1 : end, :);
        normal_1 = normalCrop(:, 1 : 2*bestMarginSize, :);
        normal_2 = normalCrop(:, end-2*bestMarginSize+1 : end, :);
        rough_1 = roughCrop(:, 1 : 2*bestMarginSize, :);
        rough_2 = roughCrop(:, end-2*bestMarginSize+1 : end, :);
        
        diffuseNew = diffuse_1 .* (1 - labeling) + diffuse_2 .* labeling;
        normalNew = normal_1 .* (1 - labeling) + normal_2 .* labeling;
        roughNew = rough_1 .* (1 - labeling) + rough_2 .* labeling;
        
        diffuseCrop = diffuseCrop(:, 2*bestMarginSize+1:end-2*bestMarginSize, :);
        normalCrop = normalCrop(:, 2*bestMarginSize+1:end-2*bestMarginSize, :);
        roughCrop = roughCrop(:, 2*bestMarginSize+1:end-2*bestMarginSize, :);
        
        diffuseCrop = cat(2, diffuseNew(:,end-bestMarginSize+1:end, :), ...
            diffuseCrop, diffuseNew(:, 1:bestMarginSize, :) );
        normalCrop = cat(2, normalNew(:,end-bestMarginSize+1:end, :), ...
            normalCrop, normalNew(:, 1:bestMarginSize, :) );
        roughCrop = cat(2, roughNew(:,end-bestMarginSize+1:end, :), ...
            roughCrop, roughNew(:, 1:bestMarginSize, :) );
        
        %% Graph cut to find the up and down seams
        [heightCrop, widthCrop, ~] = size(diffuseCrop);
        minCost = 1e10;
        bestLabeling = nan;
        bestMarginSize = nan;
        for t = minMarginSize:maxMarginSize
            diffuse_1 = diffuseCrop(1 : 2*t+1, :, :);
            diffuse_2 = diffuseCrop(end-2*t : end, :, :);
            normal_1 = normalCrop(1 : 2*t+1, :, :);
            normal_2 = normalCrop(end-2*t : end, :, :);
            rough_1 = roughCrop(1 : 2*t+1, :, :);
            rough_2 = roughCrop(end-2*t : end, :, :);
            
            % Compute the unary loss
            [rows, cols, ~] = size(diffuse_1 );
            rows = rows + 1;
            unaryLoss = zeros(2, rows, cols);
            unaryLoss(1, 1, :) = 1e10;
            unaryLoss(2, rows, :) = 1e10;
            
            % Compute the smoothness;
            [d_lr_12, d_lr_21, d_ud_12, d_ud_21] = computeSmooth_ud(diffuse_1, diffuse_2);
            [n_lr_12, n_lr_21, n_ud_12, n_ud_21] = computeSmooth_ud(normal_1, normal_2);
            [r_lr_12, r_lr_21, r_ud_12, r_ud_21] = computeSmooth_ud(rough_1, rough_2);
            
            s_lr_12 = lambda_d * d_lr_12 + lambda_n * n_lr_12 + lambda_r * r_lr_12;
            s_lr_21 = lambda_d * d_lr_21 + lambda_n * n_lr_21 + lambda_r * r_lr_21;
            s_ud_12 = lambda_d * d_ud_12 + lambda_n * n_ud_12 + lambda_r * r_ud_12;
            s_ud_21 = lambda_d * d_ud_21 + lambda_n * n_ud_21 + lambda_r * r_ud_21;
            
            % Compute the hard constraint
            s_lr_constrain = zeros(2*t, 2) + 1e10;
            
            % Build the graph
            graph = BK_Create(rows * cols );
            BK_SetUnary(graph, reshape(unaryLoss, [2, rows * cols]) );
            [x_grid, y_grid] = meshgrid(1:cols, 1:rows );
            xy_grid = (x_grid-1) * rows + y_grid;
            
            l_grid = xy_grid(2:end-1, 1:end-1);
            r_grid = xy_grid(2:end-1, 2:end);
            lr_grid = [l_grid(:), r_grid(:)];
            
            u_grid = xy_grid(1:end-1, :);
            d_grid = xy_grid(2:end, :);
            ud_grid = [u_grid(:), d_grid(:)];
            
            l_edge_grid = xy_grid(2:end-1, 1);
            r_edge_grid = xy_grid(2:end-1, cols);
            lr_edge_grid = [l_edge_grid(:), r_edge_grid(:)];
            
            s_lr = [s_lr_12(:), s_lr_21(:)];
            s_ud = [s_ud_12(:), s_ud_21(:)];
            
            smoothMat = zeros(size(lr_grid, 1) + size(ud_grid, 1) + size(lr_edge_grid, 1), 6);
            smoothMat(:, 1:2) = [lr_grid; ud_grid; lr_edge_grid];
            smoothMat(:, 4:5) = [s_lr; s_ud; s_lr_constrain];
            
            BK_SetPairwise(graph, smoothMat);
            cost = BK_Minimize(graph );
            labeling = BK_GetLabeling(graph );
            BK_Delete(graph);
            
            if cost < minCost
                minCost = cost;
                bestMarginSize = t;
                bestLabeling = labeling;
            end
        end
        
        labeling = reshape(bestLabeling, [bestMarginSize*2+2, widthCrop]);
        labeling = labeling(2:end-1, :);
        labeling = cat(3, labeling, labeling, labeling) - 1;
        labeling = imfilter(single(labeling), fspecial('gaussian', [9, 9], 3), 'replicate', 'same');
        diffuse_1 = diffuseCrop(1 : 2*bestMarginSize, :, :);
        diffuse_2 = diffuseCrop(end-2*bestMarginSize+1 : end, :, :);
        normal_1 = normalCrop(1 : 2*bestMarginSize, :, :);
        normal_2 = normalCrop(end-2*bestMarginSize+1 : end, :, :);
        rough_1 = roughCrop(1 : 2*bestMarginSize, :, :);
        rough_2 = roughCrop(end-2*bestMarginSize+1 : end, :, :);
        
        diffuseNew = diffuse_1 .* (1 - labeling) + diffuse_2 .* labeling;
        normalNew = normal_1 .* (1 - labeling) + normal_2 .* labeling;
        roughNew = rough_1 .* (1 - labeling) + rough_2 .* labeling;
        
        diffuseCrop = diffuseCrop(2*bestMarginSize+1:end-2*bestMarginSize, :, :);
        normalCrop = normalCrop(2*bestMarginSize+1:end-2*bestMarginSize, :, :);
        roughCrop = roughCrop(2*bestMarginSize+1:end-2*bestMarginSize, :, :);
        
        diffuseCrop = cat(1, diffuseNew(end-bestMarginSize+1:end, :, :), ...
            diffuseCrop, diffuseNew(1:bestMarginSize, :, :) );
        normalCrop = cat(1, normalNew(end-bestMarginSize+1:end, :, :), ...
            normalCrop, normalNew(1:bestMarginSize, :, :) );
        roughCrop = cat(1, roughNew(end-bestMarginSize+1:end, :, :), ...
            roughCrop, roughNew(1:bestMarginSize, :, :) );
        
        %% Save the results
        diffuseRes = imresize(diffuseCrop, [targetSize, targetSize] );
        normalRes = imresize(normalCrop, [targetSize, targetSize] );
        roughRes = imresize(roughCrop, [targetSize, targetSize] );
        
        dstRoot = fullfile(matDir, 'tiled');
        system(['mkdir ', dstRoot] );
        diffuseNameNew = fullfile(dstRoot, 'diffuse_tiled.png');
        normalNameNew = fullfile(dstRoot, 'normal_tiled.png');
        roughNameNew = fullfile(dstRoot, 'rough_tiled.png');
        imwrite(diffuseRes, diffuseNameNew );
        imwrite(normalRes, normalNameNew );
        imwrite(roughRes, roughNameNew );
        
        %% Visualize the results by generating 3x3 maps
        diffuseSmall = imresize(diffuseRes, [100, 100] );
        normalSmall = imresize(normalRes, [100, 100] );
        roughSmall = imresize(roughRes, [100, 100] );
        
        diffuseTiled = repmat(diffuseSmall, [3, 3, 1] );
        normalTiled = repmat(normalSmall, [3, 3, 1] );
        roughTiled = repmat(roughSmall, [3, 3, 1]);
        
        diffuseNameDemo = fullfile(dstRoot, 'diffuse_demo.jpg');
        normalNameDemo = fullfile(dstRoot, 'normal_demo.jpg');
        roughNameDemo = fullfile(dstRoot, 'rough_demo.jpg');
        
        imwrite(diffuseTiled, diffuseNameDemo );
        imwrite(normalTiled, normalNameDemo );
        imwrite(roughTiled, roughNameDemo );
    end
end
