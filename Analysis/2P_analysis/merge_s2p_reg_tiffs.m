function merge_s2p_reg_tiffs()
% merge_s2p_reg_tiffs
% 选择一个根目录，递归遍历其下所有子文件夹：
%   - 查找形如 *_reg_001.tif, *_reg_002.tif, ... 的文件
%   - 认为它们是同一个原始文件被拆分的块
%   - 按前缀（_reg_ 之前的部分）分组，并在每个文件夹内合并
%   - 合并结果命名为：<前缀>_reg_merged.tif
%
% 使用：
%   直接在 MATLAB 命令行运行：
%       merge_s2p_reg_tiffs
%   然后在弹出的对话框中选择根目录（例如 D:\suite2p_output）

    rootPath = uigetdir(pwd, '请选择 suite2p_output 的根文件夹');
    if rootPath == 0
        fprintf('用户取消选择，函数结束。\n');
        return;
    end

    fprintf('根目录: %s\n', rootPath);
    walkFolder(rootPath);
    fprintf('\n=== 全部合并完成 ===\n');
end


function walkFolder(curFolder)
    % 在当前文件夹内尝试合并 *_reg_*.tif
    mergeInFolder(curFolder);

    % 递归子文件夹
    d = dir(curFolder);
    for i = 1:numel(d)
        if d(i).isdir && ~strcmp(d(i).name, '.') && ~strcmp(d(i).name, '..')
            sub = fullfile(curFolder, d(i).name);
            walkFolder(sub);
        end
    end
end


function mergeInFolder(folder)
    % 查找当前文件夹下所有 *_reg_*.tif
    pattern = fullfile(folder, '*_reg_*.tif');
    files = dir(pattern);
    if isempty(files)
        return; % 没有需要合并的文件
    end

    names = {files.name};
    keys  = cell(size(names));  % 前缀，如 'baseline' / 'ptz'
    nums  = zeros(size(names)); % 序号，如 1, 2, 3

    % 从文件名中解析前缀和编号
    % 例如： baseline_reg_001.tif -> 前缀 baseline, 编号 1
    for i = 1:numel(names)
        [~, nm] = fileparts(names{i});  % 去掉 .tif
        tok = regexp(nm, '^(.*)_reg_(\d+)$', 'tokens', 'once');
        if isempty(tok)
            keys{i} = '';
            nums(i) = NaN;
        else
            keys{i} = tok{1};                % 前缀
            nums(i) = str2double(tok{2});    % 编号
        end
    end

    valid = ~cellfun(@isempty, keys);
    if ~any(valid)
        return;
    end

    names = names(valid);
    keys  = keys(valid);
    nums  = nums(valid);

    fprintf('\n[Folder] %s\n', folder);

    % 按前缀分组（一个前缀对应一组块文件）
    [uniqKeys, ~, ic] = unique(keys);

    for k = 1:numel(uniqKeys)
        thisKey = uniqKeys{k};
        idx = find(ic == k);   % 属于该前缀的一组

        if numel(idx) <= 1
            % 只有一个 _reg_XXX.tif，不一定是拆分的，这里选择跳过
            continue;
        end

        % 按编号排序（001, 002, 003, ...）
        [~, ord] = sort(nums(idx));
        idx = idx(ord);

        fileList = fullfile(folder, names(idx));

        % 输出文件名：<前缀>_reg_merged.tif
        outName = fullfile(folder, sprintf('%s_reg_merged.tif', thisKey));

        if exist(outName, 'file')
            fprintf('  已存在，跳过: %s\n', outName);
        else
            fprintf('  合并 %d 个块 → %s\n', numel(fileList), outName);
            mergeFileList(fileList, outName);
        end
    end
end


function mergeFileList(fileList, outName)
% 将 fileList 中的多个 *_reg_XXX.tif 合并为一个 BigTIFF（多页）
% 使用 Tiff 类，支持 int16 / uint16 / float 等数据类型

    % 如果已有同名文件，先删掉
    if exist(outName, 'file')
        delete(outName);
    end

    firstPage = true;
    t = [];

    for i = 1:numel(fileList)
        info = imfinfo(fileList{i});
        nFrames = numel(info);

        for f = 1:nFrames
            img = imread(fileList{i}, f);

            if firstPage
                % 创建 BigTIFF 文件
                % 如果你的 MATLAB 不支持 'w8'，可以改成 'w'
                t = Tiff(outName, 'w8');  % BigTIFF 模式

                tagstruct.ImageLength      = size(img, 1);
                tagstruct.ImageWidth       = size(img, 2);
                tagstruct.Photometric      = Tiff.Photometric.MinIsBlack;
                tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
                tagstruct.SamplesPerPixel  = 1;
                tagstruct.Compression      = Tiff.Compression.None;

                % 根据数据类型设置位深和 SampleFormat
                switch class(img)
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
                        error('暂不支持的数据类型: %s', class(img));
                end

                t.setTag(tagstruct);
                t.write(img);
                firstPage = false;

            else
                % 后续帧：新建一个 directory
                t.writeDirectory();

                tagstruct.ImageLength = size(img, 1);
                tagstruct.ImageWidth  = size(img, 2);
                % 其余 tag（BitsPerSample / SampleFormat 等）沿用第一次的设置
                t.setTag(tagstruct);
                t.write(img);
            end
        end
    end

    if ~isempty(t)
        t.close();
    end
end
