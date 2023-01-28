function prepare_testset()

    path_original = '../datasets/original_data';
    degradation = 'BI'; % BI, BD, DN
    dataset = {'Set5'}; % 'Set14', 'BSD100', 'Urban100', 'Manga109'
    ext = {'*.png'};

    if strcmp(degradation, 'BI')
        scale_all = [2, 3, 4];
    else
        scale_all = 3;
    end

    for idx_set = 1:length(dataset)
        fprintf('Processing %s:\n', dataset{idx_set});
        filepaths = [];
        for idx_ext = 1:length(ext)
            filepaths = cat(1, filepaths, dir(fullfile(path_original, dataset{idx_set}, ext{idx_ext}))); % ./original_data/Set5/*.png
        end
        for idx_im = 1:length(filepaths)
            name_im = filepaths(idx_im).name;
            fprintf('%d. %s: ', idx_im, name_im);
            im_ori = imread(fullfile(path_original, dataset{idx_set}, name_im)); % ./original_data/Set5/0001.png
            if size(im_ori, 3) == 1
                im_ori = cat(3, im_ori, im_ori, im_ori);
            end
            for scale = scale_all
                fprintf('x%d ', scale);
                im_HR = modcrop(im_ori, scale);
                if strcmp(degradation, 'BI')
                    im_LR = imresize(im_HR, 1/scale, 'bicubic');
                elseif strcmp(degradation, 'BD')
                    im_LR = imresize_BD(im_HR, scale, 'Gaussian', 1.6); % sigma=1.6
                elseif strcmp(degradation, 'DN')
                    randn('seed', 0); % For test data, fix seed. But, DON'T fix seed, when preparing training data.
                    im_LR = imresize_DN(im_HR, scale, 30); % noise level sigma=30
                end
                % folder
                folder_HR = fullfile('../datasets/', dataset{idx_set}, 'HR', ['x', num2str(scale)]); % ./Set5/HR/x2
                folder_LR = fullfile('../datasets/', dataset{idx_set}, ['LR', degradation], ['x', num2str(scale)]); % ./Set5/LRBI/x2
                if ~exist(folder_HR)
                    mkdir(folder_HR)
                end
                if ~exist(folder_LR)
                    mkdir(folder_LR)
                end
                % fn
                fn_HR = fullfile('../datasets/', dataset{idx_set}, 'HR', ['x', num2str(scale)], name_im);
                fn_LR = fullfile('../datasets/', dataset{idx_set}, ['LR', degradation], ['x', num2str(scale)], name_im);
                imwrite(im_HR, fn_HR, 'png');
                imwrite(im_LR, fn_LR, 'png');
            end
            fprintf('\n');
        end
        fprintf('\n');
    end
end


function imgs = modcrop(imgs, modulo)
    if size(imgs,3)==1
        sz = size(imgs);
        sz = sz - mod(sz, modulo);
        imgs = imgs(1:sz(1), 1:sz(2));
    else
        tmpsz = size(imgs);
        sz = tmpsz(1:2);
        sz = sz - mod(sz, modulo);
        imgs = imgs(1:sz(1), 1:sz(2),:);
    end
end


function [LR] = imresize_BD(im, scale, type, sigma)
    if nargin ==3 && strcmp(type,'Gaussian')
        sigma = 1.6;
    end
    if strcmp(type,'Gaussian') && fix(scale) == scale
        if mod(scale,2)==1
            kernelsize = ceil(sigma*3)*2+1;
            if scale==3 && sigma == 1.6
                kernelsize = 7;
            end
            kernel  = fspecial('gaussian',kernelsize,sigma);
            blur_HR = imfilter(im,kernel,'replicate');
            if isa(blur_HR, 'gpuArray')
                LR = blur_HR(scale-1:scale:end-1,scale-1:scale:end-1,:);
            else
                LR = imresize(blur_HR, 1/scale, 'nearest');
            end
        elseif mod(scale,2)==0
            kernelsize = ceil(sigma*3)*2+2;
            kernel = fspecial('gaussian',kernelsize,sigma);
            blur_HR = imfilter(im, kernel,'replicate');
            LR= blur_HR(scale/2:scale:end-scale/2,scale/2:scale:end-scale/2,:);
        end
    else
        LR = imresize(im, 1/scale, type);
    end
end

function ImLR = imresize_DN(ImHR, scale, sigma)
    % ImLR and ImHR are uint8 data
    % downsample by Bicubic
    ImDown = imresize(ImHR, 1/scale, 'bicubic'); % 0-255
    ImDown = single(ImDown); % 0-255
    ImDownNoise = ImDown + single(sigma*randn(size(ImDown))); % 0-255
    ImLR = uint8(ImDownNoise); % 0-255
end
