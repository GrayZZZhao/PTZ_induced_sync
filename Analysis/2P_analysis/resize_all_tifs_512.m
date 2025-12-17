function resize_all_tifs_512()
% resize_all_tifs_512
% 从用户选择的根目录开始递归：
%   - 找到所有 .tif 文件（包括子文件夹）
%   - 将每个 TIFF（可能是多帧 stack）的每一帧 resize 为 512x512
%   - 结果统一输出到 root\resize 文件夹下，文件名与原始文件相同
%     若重名，则在文件名后自动加 _01, _02 等后缀
%
% 使用：
%   在 MATLAB 命令行执行：
%       resize_all_tifs_512
%   弹出对话框中选择：
%       D:\suite2p_output\analysis

    rootPath = uigetdir(pwd, '请选择包含 TIF 的根目录（例如 D:\suite2p_output\analysis）');
    if rootPath == 0
        fprintf('用户取消选择，函数结束。\n');
        return;
    end

    fprintf('根目录: %s\n', rootPath);

    outRoot = fullfile(rootPath, 'resize');
    if ~exist(outRoot, 'dir')
        mkdir(outRoot);
    end

    % 从根目录递归开始
    processFolder(rootPath, rootPath, outRoot);

    fprintf('\n=== 所有 TIF 的 resize 操作完成，结果保存在: %s ===\n', outRoot);
end


function processFolder(folder, rootPath, outRoot)
    % 避免遍历输出目录自身
    if strcmp(folder, outRoot)
        return;
    end

    % 1) 处理当前文件夹中的 *.tif
    tifs = dir(fullfile(folder, '*.tif'));
    for k = 1:numel(tifs)
        srcFull = fullfile(folder, tifs(k).name);

        % 在输出目录中构造目标文件路径（扁平化到一个目录）
        destFull = fullfile(outRoot, tifs(k).name);

        % 若目标文件已存在，则加后缀避免覆盖
        if exist(destFull, 'file')
            [~, nameOnly, ext] = fileparts(tifs(k).name);
            idx = 1;
            while exist(destFull, 'file')
                destFull = fullfile(outRoot, sprintf('%s_%02d%s', nameOnly, idx, ext));
                idx = idx + 1;
            end
        end

        fprintf('Resize: %s\n   -> %s\n', srcFull, destFull);
        try
            resizeOneTiff(srcFull, destFull, [512 512]);
        catch ME
            fprintf('  [错误] 处理失败: %s\n  原因: %s\n', srcFull, ME.message);
        end
    end

    % 2) 递归子文件夹
    d = dir(folder);
    for i = 1:numel(d)
        if d(i).isdir && ~strcmp(d(i).name, '.') && ~strcmp(d(i).name, '..')
            sub = fullfile(folder, d(i).name);
            processFolder(sub, rootPath, outRoot);
        end
    end
end


function resizeOneTiff(srcFull, destFull, targetSize)
% 将 srcFull 指向的 TIFF（可为多帧）resize 为 targetSize，并写入 destFull
% 使用 Tiff 类，支持 int16 / uint16 / float 等

    info = imfinfo(srcFull);
    nFrames = numel(info);
    if nFrames == 0
        warning('文件无帧: %s', srcFull);
        return;
    end

    % 读取第一帧，获取数据类型
    img1 = imread(srcFull, 1);
    imgClass = class(img1);

    % 先根据第一帧的类型构造 tagstruct
    tagstruct.ImageLength      = targetSize(1);
    tagstruct.ImageWidth       = targetSize(2);
    tagstruct.Photometric      = Tiff.Photometric.MinIsBlack;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.SamplesPerPixel  = 1;
    tagstruct.Compression      = Tiff.Compression.None;

    switch imgClass
        case 'uint8'
            tagstruct.BitsPerSample = 8;
            tagstruct.SampleFormat  = Tiff.SampleFormat.UInt;
        case 'uint16'
            tagstruct.BitsPerSample = 16;
            tagstruct.SampleFormat  = Tiff.SampleFormat.UInt;
        case 'int16'
            tagstruct.BitsPerSample = 16;
            tagstruct.SampleFormat  = Tiff.SampleFormat.Int;
        case 'single'
            tagstruct.BitsPerSample = 32;
            tagstruct.SampleFormat  = Tiff.SampleFormat.IEEEFP;
        case 'double'
            tagstruct.BitsPerSample = 64;
            tagstruct.SampleFormat  = Tiff.SampleFormat.IEEEFP;
        otherwise
            error('暂不支持的数据类型: %s', imgClass);
    end

    % 如果已有同名输出文件，先删除（上层已经避免同名，这里保险起见）
    if exist(destFull, 'file')
        delete(destFull);
    end

    % 尝试使用 BigTIFF ('w8')，如果你的 MATLAB 不支持，可以把 'w8' 改成 'w'
    t = Tiff(destFull, 'w8');

    for f = 1:nFrames
        img = imread(srcFull, f);

        % resize 到目标大小
        if ~isequal(size(img,1), targetSize(1)) || ~isequal(size(img,2), targetSize(2))
            % 保持原数据类型，避免 double → int16 溢出
            imgResized = imresize(img, targetSize, 'Method', 'bilinear');
        else
            imgResized = img;
        end

        if f == 1
            % 第一帧：设置 tag 并写入
            t.setTag(tagstruct);
            t.write(imgResized);
        else
            % 后续帧：新开一个 directory
            t.writeDirectory();
            t.setTag(tagstruct);
            t.write(imgResized);
        end
    end

    t.close();
end
