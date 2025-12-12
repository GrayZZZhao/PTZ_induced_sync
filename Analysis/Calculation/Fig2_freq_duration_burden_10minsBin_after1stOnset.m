%% ================== 基本设置 ==================
T = struct2table(dataByMouse_first8);
nMice = height(T);

% 目标路径（注意没有多余空格）
baseDir = '/Users/grayzzzzzz/Desktop/PTZ figure/Gray_added/Data_analysis/Fig2_Burden/Fig2_Frequency_duration/10minsbin';

if ~exist(baseDir, 'dir')
    mkdir(baseDir);
end
fprintf('Base directory for saving: %s\n', baseDir);

% 时间 bin：0–10,10–20,20–30,30–40,40–50 (unit: min)
edges_min   = 0:10:50;                 % [0 10 20 30 40 50]
binWidth    = 10;
nBins       = numel(edges_min) - 1;    % 5
binCenters  = edges_min(1:end-1) + binWidth/2;  % 5,15,25,35,45

% bin 的标签
binLabels = cell(nBins,1);
for b = 1:nBins
    binLabels{b} = sprintf('%02d-%02d', edges_min(b), edges_min(b+1));
end

% genotype 列表
genoList = unique(T.group, 'stable');
nGeno    = numel(genoList);

%% ================== 1. Frequency（events/min，10-min bins） ==================

freqMat = nan(nMice, nBins);   % 行 = mouse, 列 = 时间段

for iMouse = 1:nMice
    t = T.IED_time{iMouse};        % 秒
    if isempty(t) || all(isnan(t))
        continue;
    end
    
    t = sort(t(:));
    t0      = t(1);                % 第一个 IED 作为 0 点
    t_shift = t - t0;              % 秒
    t_min   = t_shift / 60;        % 分钟
    
    counts = histcounts(t_min, edges_min);
    freqMat(iMouse,:) = counts / binWidth;   % events/min
end

% 导出 Frequency：每个 sheet = genotype，行 = 时间段，列 = Mouse1..Mouse5
freqFile = fullfile(baseDir, 'Fig2_Frequency_10minBins.xlsx');
fprintf('Saving frequency to: %s\n', freqFile);

for g = 1:nGeno
    genoName = genoList{g};
    
    idxG    = strcmp(T.group, genoName);
    freqG   = freqMat(idxG, :);      % [nMouseG x nBins]
    nMouseG = size(freqG,1);
    
    nRep = 5;                        % 固定 5 列
    if nMouseG >= nRep
        freqG = freqG(1:nRep, :);
    else
        freqG = [freqG; nan(nRep-nMouseG, nBins)];
    end
    
    freqG = freqG.';                 % [nBins x 5] 行 = 时间段
    
    tabFreq = array2table(freqG, ...
        'VariableNames', {'Mouse1','Mouse2','Mouse3','Mouse4','Mouse5'});
    tabFreq = addvars(tabFreq, binCenters(:), ...
        'Before', 1, 'NewVariableNames', 'TimeBin_min');
    
    writetable(tabFreq, freqFile, 'Sheet', genoName);
end

%% ================== 2. Duration（event-level，不平均） ==================

rowsPerGeno_dur = cell(nGeno,1);
tabPoolDur      = cell(nGeno,1);

totalDur_valid       = 0;
totalDur_assignedBin = 0;

for iMouse = 1:nMice
    geno    = T.group{iMouse};
    gIdx    = find(strcmp(genoList, geno));
    fileStr = T.file{iMouse};       % 或改成 Mouse_ID
    
    t = T.IED_time{iMouse};
    d = T.norm_duration{iMouse};
    if isempty(t) || all(isnan(t)) || isempty(d)
        continue;
    end
    
    t   = t(:);
    d   = d(:);
    nEv = min(numel(t), numel(d));
    t   = t(1:nEv);
    d   = d(1:nEv);
    
    [tSorted, order] = sort(t);
    dSorted = d(order);
    
    t0      = tSorted(1);
    t_shift = tSorted - t0;         % 秒
    t_min   = t_shift / 60;         % 分钟
    
    binIdx = discretize(t_min, edges_min);   % 1..nBins
    valid  = ~isnan(binIdx);
    if ~any(valid)
        continue;
    end
    
    binIdx_valid = binIdx(valid);
    d_valid      = dSorted(valid);
    
    totalDur_valid = totalDur_valid + numel(d_valid);
    
    rowsThisMouse = table();
    for b = 1:nBins
        idxB = (binIdx_valid == b);
        if ~any(idxB)
            continue;
        end
        
        d_bin = d_valid(idxB);
        totalDur_assignedBin = totalDur_assignedBin + numel(d_bin);
        
        tabRow = table( ...
            iMouse, ...
            string(fileStr), ...
            string(binLabels{b}), ...
            binCenters(b), ...
            {d_bin}, ...
            'VariableNames', {'MouseRowIdx','MouseFile', ...
                              'TimeBin','TimeBin_center_min','Durations'});
        
        rowsThisMouse = [rowsThisMouse; tabRow];
    end
    
    if ~isempty(rowsThisMouse)
        if isempty(rowsPerGeno_dur{gIdx})
            rowsPerGeno_dur{gIdx} = rowsThisMouse;
        else
            rowsPerGeno_dur{gIdx} = [rowsPerGeno_dur{gIdx}; rowsThisMouse];
        end
    end
end

fprintf('Duration: total valid events (0–50 min) = %d\n', totalDur_valid);
fprintf('Duration: assigned to bins            = %d\n', totalDur_assignedBin);

% pooled-by-timebin（某 genotype × 时间段，合并所有老鼠的 duration）
for g = 1:nGeno
    if isempty(rowsPerGeno_dur{g})
        continue;
    end
    tabG   = rowsPerGeno_dur{g};
    tabPool = table();
    
    for b = 1:nBins
        thisLabel = string(binLabels{b});
        idxB      = (tabG.TimeBin == thisLabel);
        
        if ~any(idxB)
            Dur_all = [];
        else
            DurCells = tabG.Durations(idxB);
            Dur_all  = cat(1, DurCells{:});
        end
        
        rowB = table( ...
            thisLabel, ...
            binCenters(b), ...
            {Dur_all}, ...
            'VariableNames', {'TimeBin','TimeBin_center_min','Durations'});
        
        tabPool = [tabPool; rowB];
    end
    
    tabPoolDur{g} = tabPool;
end

% 导出 Duration：8 个 sheet
durFile = fullfile(baseDir, 'Fig2_Duration_10minBins.xlsx');
fprintf('Saving duration to: %s\n', durFile);

for g = 1:nGeno
    if isempty(rowsPerGeno_dur{g})
        continue;
    end
    genoName = genoList{g};
    writetable(rowsPerGeno_dur{g}, durFile, 'Sheet', genoName);
end

for g = 1:nGeno
    if isempty(tabPoolDur{g})
        continue;
    end
    genoName      = genoList{g};
    sheetNamePool = [genoName '_TimePool'];
    writetable(tabPoolDur{g}, durFile, 'Sheet', sheetNamePool);
end

%% ================== 3. Burden（event-level：duration × 对应 bin 的 frequency） ==================

rowsPerGenoBurden = cell(nGeno,1);
tabPoolBurden     = cell(nGeno,1);

totalEv_valid       = 0;
totalEv_assignedBin = 0;

for iMouse = 1:nMice
    geno    = T.group{iMouse};
    gIdx    = find(strcmp(genoList, geno));
    fileStr = T.file{iMouse};
    
    t = T.IED_time{iMouse};
    d = T.norm_duration{iMouse};
    if isempty(t) || all(isnan(t)) || isempty(d)
        continue;
    end
    
    t   = t(:);
    d   = d(:);
    nEv = min(numel(t), numel(d));
    t   = t(1:nEv);
    d   = d(1:nEv);
    
    [tSorted, order] = sort(t);
    dSorted = d(order);
    
    t0      = tSorted(1);
    t_shift = tSorted - t0;         % 秒
    t_min   = t_shift / 60;         % 分钟
    
    binIdx = discretize(t_min, edges_min);
    valid  = ~isnan(binIdx);
    if ~any(valid)
        continue;
    end
    
    binIdx_valid = binIdx(valid);
    d_valid      = dSorted(valid);
    
    freq_mouse  = freqMat(iMouse, :);          % 一只鼠在 5 个 bin 的频率
    freq_events = freq_mouse(binIdx_valid);    % 每个 event 对应的 freq
    
    burden_valid = d_valid .* freq_events(:);  % event-level burden
    
    totalEv_valid = totalEv_valid + numel(burden_valid);
    
    rowsThisMouse = table();
    for b = 1:nBins
        idxB = (binIdx_valid == b);
        if ~any(idxB)
            continue;
        end
        
        burden_bin = burden_valid(idxB);
        totalEv_assignedBin = totalEv_assignedBin + numel(burden_bin);
        
        tabRow = table( ...
            iMouse, ...
            string(fileStr), ...
            string(binLabels{b}), ...
            binCenters(b), ...
            {burden_bin}, ...
            'VariableNames', {'MouseRowIdx','MouseFile', ...
                              'TimeBin','TimeBin_center_min','Burdens'});
        
        rowsThisMouse = [rowsThisMouse; tabRow];
    end
    
    if ~isempty(rowsThisMouse)
        if isempty(rowsPerGenoBurden{gIdx})
            rowsPerGenoBurden{gIdx} = rowsThisMouse;
        else
            rowsPerGenoBurden{gIdx} = [rowsPerGenoBurden{gIdx}; rowsThisMouse];
        end
    end
end

fprintf('Burden: total valid events (0–50 min) = %d\n', totalEv_valid);
fprintf('Burden: assigned to bins              = %d\n', totalEv_assignedBin);

% pooled-by-timebin burden
for g = 1:nGeno
    if isempty(rowsPerGenoBurden{g})
        continue;
    end
    tabG = rowsPerGenoBurden{g};
    tabPool = table();
    
    for b = 1:nBins
        thisLabel = string(binLabels{b});
        idxB      = (tabG.TimeBin == thisLabel);
        
        if ~any(idxB)
            Bur_all = [];
        else
            BurCells = tabG.Burdens(idxB);
            Bur_all  = cat(1, BurCells{:});
        end
        
        rowB = table( ...
            thisLabel, ...
            binCenters(b), ...
            {Bur_all}, ...
            'VariableNames', {'TimeBin','TimeBin_center_min','Burdens'});
        
        tabPool = [tabPool; rowB];
    end
    
    tabPoolBurden{g} = tabPool;
end

% 导出 Burden：8 个 sheet
burdenFile = fullfile(baseDir, 'Fig2_Burden_10minBins.xlsx');
fprintf('Saving burden to: %s\n', burdenFile);

for g = 1:nGeno
    if isempty(rowsPerGenoBurden{g})
        continue;
    end
    genoName = genoList{g};
    writetable(rowsPerGenoBurden{g}, burdenFile, 'Sheet', genoName);
end

for g = 1:nGeno
    if isempty(tabPoolBurden{g})
        continue;
    end
    genoName      = genoList{g};
    sheetNamePool = [genoName '_TimePool'];
    writetable(tabPoolBurden{g}, burdenFile, 'Sheet', sheetNamePool);
end
%%
%% ==== 把每只老鼠所有 10-min bin 的 burden 合并到一个 struct 里 ====

% 这里假设前面的代码已经算好了：
%   T                  = struct2table(dataByMouse_first8);
%   genoList           = unique(T.group,'stable');
%   rowsPerGenoBurden  = 每个 genotype 一个 table，包含：
%                        MouseRowIdx, MouseFile, TimeBin, TimeBin_center_min, Burdens

nMice = height(T);

% 预分配 nMice×1 的 struct，每一行对应一只老鼠
burdenByMouse_10min = struct( ...
    'genotype', cell(nMice,1), ...   % 第一列：genotype
    'mouseID',  cell(nMice,1), ...   % 第二列：mouseID
    'burden',   cell(nMice,1));      % 第三列：这一只老鼠所有 burden（n×1 double）

for iMouse = 1:nMice
    % -------- 1. 基本信息：genotype 和 mouseID --------
    geno = T.group{iMouse};
    burdenByMouse_10min(iMouse).genotype = geno;
    
    % 优先用 Mouse_ID，如果没有这个列就用 file
    if ismember('Mouse_ID', T.Properties.VariableNames)
        mouseID = T.Mouse_ID{iMouse};
    else
        mouseID = T.file{iMouse};
    end
    burdenByMouse_10min(iMouse).mouseID = mouseID;
    
    % -------- 2. 找到这个 genotype 下面，这一只老鼠的所有 burden --------
    gIdx = find(strcmp(genoList, geno));
    if isempty(gIdx) || isempty(rowsPerGenoBurden{gIdx})
        burdenByMouse_10min(iMouse).burden = [];
        continue;
    end
    
    tabG = rowsPerGenoBurden{gIdx};      % 这个 genotype 的所有 [mouse×timebin] 行
    
    % 找到属于这一只老鼠的行（所有时间段）
    idxThisMouse = (tabG.MouseRowIdx == iMouse);
    if ~any(idxThisMouse)
        burdenByMouse_10min(iMouse).burden = [];
        continue;
    end
    
    % 取出这一只老鼠所有时间段的 Burdens（cell，每个 cell 是一个向量）
    BurCells = tabG.Burdens(idxThisMouse);
    
    % 把所有时间段的 burden 纵向拼在一起，得到 n×1 double
    burdenVec = vertcat(BurCells{:});
    
    burdenByMouse_10min(iMouse).burden = burdenVec;
end

% 保存到同一个 10minsbin 文件夹里
save(fullfile(baseDir, 'burdenByMouse_10min_struct.mat'), 'burdenByMouse_10min');
fprintf('Saved per-mouse burden struct to: %s\n', ...
    fullfile(baseDir, 'burdenByMouse_10min_struct.mat'));




%%

%% ==== 调试：查看某只老鼠（如 HOM1022）每个 event 的 TimeBin / Freq / Duration / Burden ====

% 目标老鼠 ID（看你表里 Mouse_ID 的写法）
targetMouseID = 'D:\2025PTZ\2025-04-30_1022_HOM-female-adult\2025-04-30_11-54-07\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-LFP\Str_Cor_histPeak_diff\Fig2_data\Fig2_triple_points_20251121_173142.xlsx';   % 如果没有 Mouse_ID，可以改成 file 里的一部分字符串

% ---------- 找到这只老鼠在 T 里的行号 ----------
if ismember('Mouse_ID', T.Properties.VariableNames)
    iMouse = find(strcmp(T.Mouse_ID, targetMouseID));
else
    % 如果没有 Mouse_ID，就用 file 名里包含 'HOM1022' 来找
    iMouse = find(contains(T.file, targetMouseID));
end

if isempty(iMouse)
    error('找不到 Mouse = %s 对应的行，请检查 Mouse_ID 或 file 名。', targetMouseID);
end
iMouse = iMouse(1);   % 保险起见，取第一个匹配

fprintf('Target mouse row index = %d\n', iMouse);

% ---------- 取出这只鼠的 IED_time 和 norm_duration ----------
t = T.IED_time{iMouse};         % 秒
d = T.norm_duration{iMouse};    % 秒（或你定义的单位）

if isempty(t) || all(isnan(t)) || isempty(d)
    error('该老鼠没有有效的 IED_time 或 norm_duration 数据。');
end

t   = t(:);
d   = d(:);
nEv = min(numel(t), numel(d));
t   = t(1:nEv);
d   = d(1:nEv);

% 按时间排序，保持和你前面代码一致
[tSorted, order] = sort(t);
dSorted          = d(order);

% 相对第一个 IED 的时间（分钟），同样和前面代码保持一致
t0      = tSorted(1);
t_shift = tSorted - t0;        % 秒
t_min   = t_shift / 60;        % 分钟

% 计算每个 event 落在哪个 10-min bin
binIdx = discretize(t_min, edges_min);   % 1..nBins
valid  = ~isnan(binIdx);

if ~any(valid)
    error('这只老鼠的 IED 都不在 0–50 min 的时间窗内。');
end

binIdx_valid = binIdx(valid);
t_min_valid  = t_min(valid);
t_sec_valid  = tSorted(valid);
d_valid      = dSorted(valid);

% ---------- 找到对应的 Frequency（events/min） ----------
% freqMat(iMouse,:) 是这只鼠在每个 bin 的 events/min
freq_mouse  = freqMat(iMouse, :);            % [1 x nBins]
freq_events = freq_mouse(binIdx_valid);      % 每个 event 对应的频率 [nValid x 1]

% ---------- 重新算一遍 event-level burden（应该和 struct 里的那串值一样） ----------
burden_valid = d_valid .* freq_events(:);    % [nValid x 1]

% ---------- 组装成一个表 ----------
TimeBin_str = string(binLabels(binIdx_valid));   % '00-10','10-20',...

tabEv = table( ...
    (1:numel(burden_valid))', ...      % event 索引（按时间排序后的编号）
    t_sec_valid, ...                   % 绝对时间（相对 recording 开始），秒
    t_min_valid, ...                   % 相对第一个 IED 的时间，分钟
    TimeBin_str(:), ...                % 所属 10-min bin
    freq_events(:), ...                % 该 bin 的 frequency (events/min)
    d_valid, ...                       % 该 event 的 norm_duration
    burden_valid, ...                  % 该 event 的 burden
    'VariableNames', {'EvIdx','t_sec','t_min_rel','TimeBin', ...
                      'Freq_bin','Duration','Burden'});

% 按 burden 从大到小排一下，方便你先看“超大”的几个
tabEv_sorted = sortrows(tabEv, 'Burden', 'descend');

% 显示前 20 个（你给的那几个大值应该都在这里）
disp(tabEv_sorted(1:20,:));


%% 尝试去除outlier

%% === 1. 从 burdenByMouse_10min 合并所有 burden 和 genotype ===
burdenByMouse_all = burdenByMouse_10min;   % 只是起个别名，方便你以后改名

nMice    = numel(burdenByMouse_all);
allBurden = [];        % nEvents x 1
allGeno   = {};        % nEvents x 1，存 'WT' / 'HOM' / ...

for iM = 1:nMice
    g = burdenByMouse_all(iM).genotype;   % 比如 'WT','HOM',...
    b = burdenByMouse_all(iM).burden;     % 这一只鼠所有 event 的 burden (n_i x 1)

    if isempty(b) || all(isnan(b))
        continue;                         % 没有数据就跳过
    end

    % 展开成列向量再拼接
    b = b(:);
    allBurden = [allBurden; b];

    % 为这一只鼠的每个 event 复制 genotype 标签
    allGeno   = [allGeno; repmat({g}, numel(b), 1)];
end

allBurden = allBurden(:);   % 确保是列
allGeno   = allGeno(:);     % cell array 列向量

%% === 2. 对 log10(burden) 做 robust z-score（跳过 <=0 或 NaN 的点） ===

validLog = allBurden > 0 & ~isnan(allBurden);   % log10 只能对正数做

y        = log10(allBurden(validLog));
med      = median(y,'omitnan');
mad_val  = mad(y,1);         % median absolute deviation

z        = nan(size(allBurden));       % 先全 NaN
z(validLog) = (log10(allBurden(validLog)) - med) / mad_val;

% 统一阈值，比如 |z| > 3
isOutlier = abs(z) > 4;

fprintf('总共 %d 个点被标记为 outlier，占 %.2f%%\n', ...
    sum(isOutlier), 100*mean(isOutlier));

%% === 3. 清洗后的 burden / genotype，后面画图或做 LME 用 ===
burden_clean = allBurden(~isOutlier);
geno_clean   = allGeno(~isOutlier);

%%
%% === 合并所有 burden + genotype + mouseID，并做 outlier 检测 ===

burdenByMouse_all = burdenByMouse_10min;   % 起个别名

nMice = numel(burdenByMouse_all);

allBurden      = [];   % 所有 event 的 burden
allGeno        = {};   % 对应 genotype ('WT','HOM',...)
allMouseID     = {};   % 对应 mouseID（路径或编号）
allMouseRowIdx = [];   % 在 struct 中的 mouse 行号 (1..nMice)
allEventIdx    = [];   % 该老鼠内部的第几个 event

for iM = 1:nMice
    % --- 这一只鼠的基础信息 ---
    g = burdenByMouse_all(iM).genotype;
    if isstring(g), g = char(g); end
    
    mID = burdenByMouse_all(iM).mouseID;
    if isstring(mID), mID = char(mID); end
    
    b = burdenByMouse_all(iM).burden;   % n_i x 1 double
    if isempty(b) || all(isnan(b))
        continue;
    end
    b   = b(:);
    nEv = numel(b);
    
    % --- 累加到总向量中 ---
    allBurden      = [allBurden; b];
    allGeno        = [allGeno;        repmat({g},   nEv, 1)];
    allMouseID     = [allMouseID;     repmat({mID}, nEv, 1)];
    allMouseRowIdx = [allMouseRowIdx; repmat(iM,    nEv, 1)];
    allEventIdx    = [allEventIdx;    (1:nEv)'];   % 该老鼠内部的 event 序号
end

allBurden       = allBurden(:);
allGeno         = allGeno(:);
allMouseID      = allMouseID(:);
allMouseRowIdx  = allMouseRowIdx(:);
allEventIdx     = allEventIdx(:);

%% === 用 log10(burden) 做 robust z-score ===

validLog = allBurden > 0 & ~isnan(allBurden);   % log10 只对正数有效

y       = log10(allBurden(validLog));
medVal  = median(y,'omitnan');
mad_val = mad(y,1);      % median absolute deviation

z = nan(size(allBurden));                 % 先全 NaN
z(validLog) = (log10(allBurden(validLog)) - medVal) / mad_val;

% 统一 outlier 阈值（你可以改成 3.5 或别的）
zThr      = 5;
isOutlier = abs(z) > zThr;

fprintf('总共 %d 个点被标记为 outlier，占 %.2f%%\n', ...
    sum(isOutlier), 100 * mean(isOutlier));

%% === 1）生成：每个被排除点对应哪只老鼠、哪个 event 的报告 ===

idxOut = find(isOutlier);

tabOut = table( ...
    allMouseRowIdx(idxOut), ...
    allGeno(idxOut), ...
    allMouseID(idxOut), ...
    allEventIdx(idxOut), ...
    allBurden(idxOut), ...
    z(idxOut), ...
    'VariableNames', {'MouseRowIdx','Genotype','MouseID', ...
                      'EventIdx_withinMouse','Burden','z_log10'});

% ---- 用 findgroups + splitapply 统计每只鼠删了多少点 ----
[G, genoGrp, mouseGrp] = findgroups(tabOut.Genotype, tabOut.MouseID);
nOutliersPerMouse      = splitapply(@numel, tabOut.Burden, G);

summStr = table(genoGrp, mouseGrp, nOutliersPerMouse, ...
    'VariableNames', {'Genotype','MouseID','N_outliers'});

disp('每只老鼠被删点的数量汇总：');
disp(summStr);

%% === 2）生成 LME 可用的 clean 表（每行一个 event） ===

idxKeep = ~isOutlier;

tabLME = table( ...
    allBurden(idxKeep), ...
    allGeno(idxKeep), ...
    allMouseID(idxKeep), ...
    allMouseRowIdx(idxKeep), ...
    'VariableNames', {'res','genotype','mouseID','MouseRowIdx'});

%% === 3）写入 Excel ===

% 如果没有 baseDir，就默认用当前工作目录
if ~exist('baseDir','var') || ~isfolder(baseDir)
    baseDir = pwd;
end

outlierFile = fullfile(baseDir, 'Fig2_Burden_10minBins_OutlierReport.xlsx');
lmeFile     = fullfile(baseDir, 'Fig2_Burden_10minBins_LME_clean.xlsx');

% Sheet1: 所有被删掉的点
writetable(tabOut, outlierFile, 'Sheet', 'Outliers');
% Sheet2: 每只鼠删了多少点
writetable(summStr, outlierFile, 'Sheet', 'SummaryByMouse');

% LME 用的干净表
writetable(tabLME, lmeFile);

fprintf('Outlier 报告已保存到：\n  %s\n', outlierFile);
fprintf('LME 清洗后数据已保存到：\n  %s\n', lmeFile);




%% ========= 从 dataByMouse_first8 里提取 0–10 和 40–50 min 的 event-level burden =========
% 定义时间 bin：0–10, 10–20, 20–30, 30–40, 40–50 min
edges_min = 0:10:50;   % [0 10 20 30 40 50]
binWidth  = 10;        % min

T      = struct2table(dataByMouse_first8);
nMice  = height(T);

% 预分配一个 cell，用来存每只鼠 0–10 和 40–50 min 的 burden
burden_0_10_40_50_cell = cell(nMice,1);

for iM = 1:nMice
    t = T.IED_time{iM};        % 秒
    d = T.norm_duration{iM};   % 对应的 duration
    
    if isempty(t) || all(isnan(t)) || isempty(d)
        burden_0_10_40_50_cell{iM} = [];
        continue;
    end
    
    % 保证列向量 & 长度一致
    t = t(:);
    d = d(:);
    nEv = min(numel(t), numel(d));
    t = t(1:nEv);
    d = d(1:nEv);
    
    % 按时间排序（和之前 Fig2 的代码保持一致）
    [tSorted, order] = sort(t);
    dSorted          = d(order);
    
    % 以第一个 IED 作为 0 点，转成“相对时间（分钟）”
    t0      = tSorted(1);
    t_shift = tSorted - t0;      % 秒
    t_min   = t_shift / 60;      % 分钟
    
    % -------- 先算每个 10-min bin 的 frequency（events/min）--------
    counts     = histcounts(t_min, edges_min);   % [1 x 5]
    freq_mouse = counts / binWidth;             % 这只鼠在 5 个 bin 各自的 events/min
    
    % -------- 再算每个 event 属于哪个 bin，并得到 event-level burden --------
    binIdx = discretize(t_min, edges_min);   % 1..5 或 NaN（不在 0–50 min 内）
    valid  = ~isnan(binIdx);
    if ~any(valid)
        burden_0_10_40_50_cell{iM} = [];
        continue;
    end
    
    binIdx_valid = binIdx(valid);        % 每个有效 event 的 bin 编号
    d_valid      = dSorted(valid);       % 对应 duration
    freq_events  = freq_mouse(binIdx_valid);   % 对应 bin 的 frequency
    
    % event-level burden = duration × 对应 bin 的 frequency
    burden_valid = d_valid .* freq_events(:);   % [nValid x 1]
    
    % -------- 只保留 0–10 和 40–50 min（即 bin 1 和 bin 5）--------
    mask_0_10_40_50 = (binIdx_valid == 1) | (binIdx_valid == 5);
    burden_0_10_40_50 = burden_valid(mask_0_10_40_50);
    
    % 存到 cell 里
    burden_0_10_40_50_cell{iM} = burden_0_10_40_50;
end

% ========= 把结果写回 dataByMouse_first8 的第 9 个 field =========
% 新字段名：burden_0_10_40_50，每个元素是 n×1 double（不平均）
for iM = 1:nMice
    dataByMouse_first8(iM).burden_0_10_40_50 = burden_0_10_40_50_cell{iM};
end

% 小提示
fprintf('已将 0–10 与 40–50 min 的 event-level burden 写入 dataByMouse_first8.burden_0_10_40_50。\n');

%%
%% ========= 从 dataByMouse_first8 里提取 10–20, 20–30, 30–40 min 的 event-level burden =========
% 定义时间 bin：0–10, 10–20, 20–30, 30–40, 40–50 min
edges_min = 0:10:50;   % [0 10 20 30 40 50]
binWidth  = 10;        % 单位：min

T      = struct2table(dataByMouse_first8);
nMice  = height(T);

% 预分配一个 cell，用来存每只鼠 10–20, 20–30, 30–40 min 的 burden
burden_10_40_mid_cell = cell(nMice,1);

for iM = 1:nMice
    t = T.IED_time{iM};        % 秒
    d = T.norm_duration{iM};   % 对应的 duration
    
    if isempty(t) || all(isnan(t)) || isempty(d)
        burden_10_40_mid_cell{iM} = [];
        continue;
    end
    
    % 保证列向量 & 长度一致
    t = t(:);
    d = d(:);
    nEv = min(numel(t), numel(d));
    t = t(1:nEv);
    d = d(1:nEv);
    
    % -------- 按时间排序（和 Fig2 代码一致）--------
    [tSorted, order] = sort(t);
    dSorted          = d(order);
    
    % 以这只鼠的第一个 IED 作为 0 点，转成“相对时间（分钟）”
    t0      = tSorted(1);    % onset = IED_time 的第一个值
    t_shift = tSorted - t0;  % 秒
    t_min   = t_shift / 60;  % 分钟
    
    % -------- 先算每个 10-min bin 的 frequency（events/min）--------
    counts     = histcounts(t_min, edges_min);   % [1 x 5]
    freq_mouse = counts / binWidth;             % 这只鼠在 5 个 bin 各自的 events/min
    
    % -------- 再算每个 event 属于哪个 bin，并得到 event-level burden --------
    binIdx = discretize(t_min, edges_min);   % 1..5 或 NaN（不在 0–50 min 内）
    valid  = ~isnan(binIdx);
    if ~any(valid)
        burden_10_40_mid_cell{iM} = [];
        continue;
    end
    
    binIdx_valid = binIdx(valid);        % 每个有效 event 的 bin 编号
    d_valid      = dSorted(valid);       % 对应 duration
    freq_events  = freq_mouse(binIdx_valid);   % 对应 bin 的 frequency
    
    % event-level burden = duration × 对应 bin 的 frequency
    burden_valid = d_valid .* freq_events(:);   % [nValid x 1]
    
    % -------- 只保留 10–20, 20–30, 30–40 min（即 bin 2,3,4）--------
    mask_10_40_mid = (binIdx_valid == 2) | (binIdx_valid == 3) | (binIdx_valid == 4);
    burden_10_40_mid = burden_valid(mask_10_40_mid);
    
    % 存到 cell 里
    burden_10_40_mid_cell{iM} = burden_10_40_mid;
end

% ========= 把结果写回 dataByMouse_first8 的新 field =========
% 新字段名：burden_10_40_mid，每个元素是 n×1 double（不平均）
for iM = 1:nMice
    dataByMouse_first8(iM).burden_10_40_mid = burden_10_40_mid_cell{iM};
end

fprintf('已将 10–20, 20–30, 30–40 min 的 event-level burden 写入 dataByMouse_first8.burden_10_40_mid。\n');


%%
%% ========= 从 dataByMouse_first8 里提取 5–45 min 的 event-level burden =========
% 时间 bin 仍然按 0–10, 10–20, 20–30, 30–40, 40–50 min 定义，用于算 frequency
edges_min = 0:10:50;   % [0 10 20 30 40 50]
binWidth  = 10;        % 单位：min

T      = struct2table(dataByMouse_first8);
%T      = struct2table(burdenByMouse_10min);
nMice  = height(T);

% 预分配一个 cell，用来存每只鼠 5–45 min 的 burden
burden_5_45_cell = cell(nMice,1);

for iM = 1:nMice
    t = T.IED_time{iM};        % 秒
    d = T.norm_duration{iM};   % 对应的 duration
    
    if isempty(t) || all(isnan(t)) || isempty(d)
        burden_5_45_cell{iM} = [];
        continue;
    end
    
    % 保证列向量 & 长度一致
    t = t(:);
    d = d(:);
    nEv = min(numel(t), numel(d));
    t = t(1:nEv);
    d = d(1:nEv);
    
    % -------- 按时间排序（和 Fig2 代码一致）--------
    [tSorted, order] = sort(t);
    dSorted          = d(order);
    
    % 以这只鼠的第一个 IED 作为 0 点，转成“相对时间（分钟）”
    t0      = tSorted(1);    % onset = IED_time 的第一个值
    t_shift = tSorted - t0;  % 秒
    t_min   = t_shift / 60;  % 分钟
    
    % -------- 先算每个 10-min bin 的 frequency（events/min）--------
    counts     = histcounts(t_min, edges_min);   % [1 x 5]
    freq_mouse = counts / binWidth;             % 这只鼠在 5 个 bin 各自的 events/min
    
    % -------- 再算每个 event 属于哪个 bin，并得到 event-level burden --------
    binIdx = discretize(t_min, edges_min);   % 1..5 或 NaN（不在 0–50 min 内）
    valid  = ~isnan(binIdx);
    if ~any(valid)
        burden_5_45_cell{iM} = [];
        continue;
    end
    
    binIdx_valid = binIdx(valid);        % 每个有效 event 的 bin 编号
    d_valid      = dSorted(valid);       % 对应 duration
    t_min_valid  = t_min(valid);         % 对应相对时间（min）
    freq_events  = freq_mouse(binIdx_valid);   % 对应 bin 的 frequency
    
    % event-level burden = duration × 对应 bin 的 frequency
    burden_valid = d_valid .* freq_events(:);   % [nValid x 1]
    
    % -------- 只保留 5–45 min 之间的事件（含 5, 不含 45）--------
    mask_5_45   = (t_min_valid >= 5) & (t_min_valid < 45);
    burden_5_45 = burden_valid(mask_5_45);
    
    % 存到 cell 里
    burden_5_45_cell{iM} = burden_5_45;
end

% ========= 把结果写回 dataByMouse_first8 的新 field =========
% 新字段名：burden_5_45，每个元素是 n×1 double（不平均）
for iM = 1:nMice
    %dataByMouse_first8(iM).burden_5_45 = burden_5_45_cell{iM};
    burdenByMouse_10min(iM).burden_5_45 = burden_5_45_cell{iM};
end

fprintf('已将 5–45 min 的 event-level burden 写入 dataByMouse_first8.burden_5_45。\n');
%%

%% ========= 只保留 onset 后中间 80% event 的 event-level burden =========
% 定义时间 bin：0–10, 10–20, 20–30, 30–40, 40–50 min
edges_min = 0:10:50;   % [0 10 20 30 40 50]
binWidth  = 10;        % 单位：min

T      = struct2table(dataByMouse_first8);
nMice  = height(T);

% 预分配一个 cell，用来存每只鼠“中间 80% event”的 burden
burden_middle80_cell = cell(nMice,1);

for iM = 1:nMice
    t = T.IED_time{iM};        % 秒，所有 IED 的时间
    d = T.norm_duration{iM};   % 对应的 duration （和 t 一一对应）
    
    if isempty(t) || all(isnan(t)) || isempty(d)
        burden_middle80_cell{iM} = [];
        continue;
    end
    
    % ---- 保证列向量 & 长度一致 ----
    t = t(:);
    d = d(:);
    nEv_raw = min(numel(t), numel(d));
    t = t(1:nEv_raw);
    d = d(1:nEv_raw);
    
    % ---- 按时间排序（和 Fig2 代码保持一致）----
    [tSorted, order] = sort(t);
    dSorted          = d(order);
    
    % 以这只鼠最早的 IED 作为 onset（0 点），转成“相对时间（分钟）”
    t0      = tSorted(1);      % onset = IED_time 的第一个值
    t_shift = tSorted - t0;    % 秒
    t_min   = t_shift / 60;    % 分钟（相对 onset）
    
    % ---- 先算每个 10-min bin 的 frequency（events/min）----
    counts     = histcounts(t_min, edges_min);   % [1 x 5]
    freq_mouse = counts / binWidth;             % 这只鼠在 5 个 bin 各自的 events/min
    
    % ---- 再算每个 event 属于哪个 bin，并得到 event-level burden ----
    binIdx = discretize(t_min, edges_min);   % 1..5 或 NaN（不在 0–50 min 内）
    valid  = ~isnan(binIdx);                 % 只保留 0–50 min 内的 event
    
    if ~any(valid)
        burden_middle80_cell{iM} = [];
        continue;
    end
    
    binIdx_valid = binIdx(valid);        % 每个有效 event 的 bin 编号
    d_valid      = dSorted(valid);       % 对应 duration
    t_min_valid  = t_min(valid);         % 对应相对时间（min）
    freq_events  = freq_mouse(binIdx_valid);   % 对应 bin 的 frequency
    
    % event-level burden = duration × 对应 bin 的 frequency
    burden_valid = d_valid .* freq_events(:);   % [nValid x 1]
    nValid       = numel(burden_valid);
    
    % ---- 只保留“中间 80% 的 event”（按时间排序后的 event 序号）----
    % 也就是丢掉最早 10% 和最晚 10% 的 event
    if nValid < 5
        % 如果有效事件太少，就不做 80% 筛选，全部保留
        burden_middle80 = burden_valid;
    else
        % 以索引为准：1..nValid
        idx_all = (1:nValid).';
        lowIdx  = floor(nValid * 0.10) + 1;   % > 10% 部分的第一个
        highIdx = ceil (nValid * 0.90);       % < 90% 部分的最后一个
        
        if lowIdx > highIdx
            % 极端小样本情况下，干脆全部保留
            burden_middle80 = burden_valid;
        else
            keepMask = (idx_all >= lowIdx) & (idx_all <= highIdx);
            burden_middle80 = burden_valid(keepMask);
        end
    end
    
    % 存到 cell 里
    burden_middle80_cell{iM} = burden_middle80;
end

% ========= 把结果写回 dataByMouse_first8 的新 field =========
% 新字段名：burden_middle80，每只鼠是 n×1 double（不平均）
for iM = 1:nMice
    burdenByMouse_10min(iM).burden_middle80 = burden_middle80_cell{iM};
end

fprintf('已将 onset 后中间 80%% event 的 event-level burden 写入 dataByMouse_first8.burden_middle80。\n');




%% ========= 只保留 onset 后中间 90% event 的 event-level burden =========
% 时间 bin 仍然是 0–10, 10–20, 20–30, 30–40, 40–50 min
edges_min = 0:10:50;   % [0 10 20 30 40 50]
binWidth  = 10;        % 单位：min

T      = struct2table(dataByMouse_first8);
nMice  = height(T);

% 预分配一个 cell，用来存每只鼠“中间 90% event”的 burden
burden_middle90_cell = cell(nMice,1);

for iM = 1:nMice
    t = T.IED_time{iM};        % 秒，所有 IED 的时间
    d = T.norm_duration{iM};   % 对应 duration
    
    if isempty(t) || all(isnan(t)) || isempty(d)
        burden_middle90_cell{iM} = [];
        continue;
    end
    
    % ---- 保证列向量 & 长度一致 ----
    t = t(:);
    d = d(:);
    nEv_raw = min(numel(t), numel(d));
    t = t(1:nEv_raw);
    d = d(1:nEv_raw);
    
    % ---- 按时间排序（和 Fig2 代码一致）----
    [tSorted, order] = sort(t);
    dSorted          = d(order);
    
    % 以这只鼠最早的 IED 作为 onset（0 点），转成“相对时间（分钟）”
    t0      = tSorted(1);      % onset = IED_time 的第一个值
    t_shift = tSorted - t0;    % 秒
    t_min   = t_shift / 60;    % 分钟（相对 onset）
    
    % ---- 先算每个 10-min bin 的 frequency（events/min）----
    counts     = histcounts(t_min, edges_min);   % [1 x 5]
    freq_mouse = counts / binWidth;             % 这只鼠在 5 个 bin 各自的 events/min
    
    % ---- 再算每个 event 属于哪个 bin，并得到 event-level burden ----
    binIdx = discretize(t_min, edges_min);   % 1..5 或 NaN（不在 0–50 min 内）
    valid  = ~isnan(binIdx);                 % 只保留 0–50 min 内的 event
    
    if ~any(valid)
        burden_middle90_cell{iM} = [];
        continue;
    end
    
    binIdx_valid = binIdx(valid);        % 每个有效 event 的 bin 编号
    d_valid      = dSorted(valid);       % 对应 duration
    t_min_valid  = t_min(valid);         % 对应相对时间（min）
    freq_events  = freq_mouse(binIdx_valid);   % 对应 bin 的 frequency
    
    % event-level burden = duration × 对应 bin 的 frequency
    burden_valid = d_valid .* freq_events(:);   % [nValid x 1]
    nValid       = numel(burden_valid);
    
    % ---- 只保留“中间 90% 的 event”（按时间排序后的 event 序号）----
    % 去掉最早 5% 和最晚 5%
    if nValid < 5
        % 有效事件太少就不过滤，全部保留
        burden_middle90 = burden_valid;
    else
        idx_all = (1:nValid).';
        lowIdx  = floor(nValid * 0.05) + 1;   % > 5% 部分的第一个
        highIdx = ceil (nValid * 0.95);       % < 95% 部分的最后一个
        
        if lowIdx > highIdx
            % 极端小样本情况下，干脆全部保留
            burden_middle90 = burden_valid;
        else
            keepMask        = (idx_all >= lowIdx) & (idx_all <= highIdx);
            burden_middle90 = burden_valid(keepMask);
        end
    end
    
    % 存到 cell 里
    burden_middle90_cell{iM} = burden_middle90;
end

% ========= 把结果写回 dataByMouse_first8 的新 field =========
% 新字段名：burden_middle90，每只鼠是 n×1 double（不平均）
for iM = 1:nMice
    burdenByMouse_10min(iM).burden_middle90 = burden_middle90_cell{iM};
end

fprintf('已将 onset 后中间 90%% event 的 event-level burden 写入 dataByMouse_first8.burden_middle90。\n');


%% ========= 按新定义计算每个 event 的 burden =========
% burden(event i) = norm_duration(i) / (IED_time_last - IED_time_first)
% 结果存入 dataByMouse_first8(i).burden_dur_over_totalTime

T     = struct2table(dataByMouse_first8);
nMice = height(T);

burden_new_cell = cell(nMice,1);

for iM = 1:nMice
    t = T.IED_time{iM};        % 秒，IED 的时间戳
    d = T.norm_duration{iM};   % 对应的 duration
    
    if isempty(t) || all(isnan(t)) || isempty(d)
        burden_new_cell{iM} = [];
        continue;
    end
    
    % ---- 列向量 + 长度匹配 ----
    t = t(:);
    d = d(:);
    nEv = min(numel(t), numel(d));
    t = t(1:nEv);
    d = d(1:nEv);
    
    % 只对 t、d 都非 NaN 的 event 计算
    valid = ~isnan(t) & ~isnan(d);
    if ~any(valid)
        burden_new_cell{iM} = nan(nEv,1);
        continue;
    end
    
    t_valid = t(valid);
    d_valid = d(valid);
    
    % ---- 用时间排序来定义“第一个 / 最后一个 IED” ----
    [t_sorted, order] = sort(t_valid);        % 升序
    d_sorted          = d_valid(order);
    
    totalTime = t_sorted(end) - t_sorted(1);  % 秒
    
    if totalTime <= 0
        % 如果总时间异常（只有一个点或时间不递增），就全部设为 NaN
        b_valid_sorted = nan(numel(t_sorted),1);
    else
        % 计算排序后的 burden，再按原顺序还原
        b_valid_sorted = d_sorted / totalTime;          % duration / 总时长
    end
    
    % 把排序后的 burden 映射回 valid 事件的原顺序
    b_valid = nan(numel(t_valid),1);
    b_valid(order) = b_valid_sorted;
    
    % 最终向量对齐到所有 event（包括 NaN 的位置）
    burden_vec = nan(nEv,1);
    burden_vec(valid) = b_valid;
    
    burden_new_cell{iM} = burden_vec;
end

% ========= 写回 dataByMouse_first8 的新 field =========
for iM = 1:nMice
    burdenByMouse_10min(iM).burden_dur_over_totalTime = burden_new_cell{iM};
end

fprintf('已按 duration/总时间 计算每个 event 的 burden，并写入 dataByMouse_first8.burden_dur_over_totalTime。\n');

%%

%% ========= 新定义：burden = duration / totalTime，然后只保留 onset 后中间 80% 的 event =========
% totalTime = IED_time 里最后一个值和第一个值的差值（秒）
% 只考虑 0–50 min 内的 event，在这些 event 中按时间序列保留中间 80%
% 结果存入 dataByMouse_first8(i).burden_middle80_durOverTotalTime

% 时间 bin 仍然用 0–10, 10–20, 20–30, 30–40, 40–50 min 来界定 0–50 min
edges_min = 0:10:50;   % [0 10 20 30 40 50]

T     = struct2table(dataByMouse_first8);
nMice = height(T);

burden_mid80_new_cell = cell(nMice,1);

for iM = 1:nMice
    t = T.IED_time{iM};        % 秒，IED 时间戳
    d = T.norm_duration{iM};   % 对应 duration
    
    if isempty(t) || all(isnan(t)) || isempty(d)
        burden_mid80_new_cell{iM} = [];
        continue;
    end
    
    % ---- 保证列向量 & 长度一致 ----
    t = t(:);
    d = d(:);
    nEv = min(numel(t), numel(d));
    t = t(1:nEv);
    d = d(1:nEv);
    
    % 只对 t、d 都非 NaN 的 event 进行后续计算
    valid_td = ~isnan(t) & ~isnan(d);
    if ~any(valid_td)
        burden_mid80_new_cell{iM} = [];
        continue;
    end
    
    t_valid = t(valid_td);
    d_valid = d(valid_td);
    
    % ---- 按时间排序，确定 onset 和 totalTime ----
    [t_sorted, order] = sort(t_valid);      % 升序
    d_sorted          = d_valid(order);
    
    if numel(t_sorted) < 2
        % 只有一个 event，totalTime = 0，无法定义；直接返回 NaN 或空
        burden_mid80_new_cell{iM} = d_sorted * NaN;
        continue;
    end
    
    % onset = 最早的 IED；totalTime = 最晚 - 最早
    t0        = t_sorted(1);               % onset（秒）
    totalTime = t_sorted(end) - t_sorted(1);  % 秒
    
    if totalTime <= 0
        % 时间异常，给这一鼠全 NaN
        burden_sorted_all = d_sorted * NaN;
    else
        % 新定义：每个 event 的 burden = duration / totalTime
        burden_sorted_all = d_sorted / totalTime;   % [nValid_td x 1]
    end
    
    % ---- 计算相对 onset 的时间（分钟），并限定在 0–50 min 内 ----
    t_shift = t_sorted - t0;       % 秒
    t_min   = t_shift / 60;        % 分钟
    
    % 用 discretize 判定哪些在 0–50 min 内
    binIdx_0_50 = discretize(t_min, edges_min);   % 1..5 或 NaN
    in_0_50     = ~isnan(binIdx_0_50);
    
    if ~any(in_0_50)
        % 没有落在 0–50 min 内的 event
        burden_mid80_new_cell{iM} = [];
        continue;
    end
    
    % 取出 0–50 min 内的事件（保持时间顺序）
    burden_0_50 = burden_sorted_all(in_0_50);
    nValid_0_50 = numel(burden_0_50);
    
    % ---- 只保留 0–50 min 内这些 event 中“中间 80%” ----
    % 丢掉最早 10% 和最晚 10%（按在 0–50 min 内的顺序编号）
    if nValid_0_50 < 5
        % 太少就不过滤，全部保留
        burden_mid80 = burden_0_50;
    else
        idx_all = (1:nValid_0_50).';
        lowIdx  = floor(nValid_0_50 * 0.10) + 1;   % >10% 的第一个
        highIdx = ceil (nValid_0_50 * 0.90);       % <90% 的最后一个
        
        if lowIdx > highIdx
            % 极端情况下，干脆不裁剪
            burden_mid80 = burden_0_50;
        else
            keepMask    = (idx_all >= lowIdx) & (idx_all <= highIdx);
            burden_mid80 = burden_0_50(keepMask);
        end
    end
    
    % 存入 cell
    burden_mid80_new_cell{iM} = burden_mid80;
end

% ========= 写回 dataByMouse_first8 的新 field =========
% 新字段名：burden_middle80_durOverTotalTime，每只鼠是 n×1 double（不平均）
for iM = 1:nMice
    burdenByMouse_10min(iM).burden_middle80_durOverTotalTime = burden_mid80_new_cell{iM};
end

fprintf('已按 duration/总时间 + onset 后中间 80%% 计算 event-level burden，并写入 dataByMouse_first8.burden_middle80_durOverTotalTime。\n');

%% ========= 每只老鼠：对 burden_middle80_durOverTotalTime 求和 =========
% 新字段：burden_middle80_sum_durOverTotalTime（标量 double）

nMice = numel(dataByMouse_first8);

% 预分配一个向量存每只鼠的总 burden
burdenSum_middle80 = nan(nMice,1);

for iM = 1:nMice
    if ~isfield(dataByMouse_first8, 'burden_middle80_durOverTotalTime')
        error('dataByMouse_first8 里尚未存在字段 burden_middle80_durOverTotalTime，请先运行前面的代码。');
    end
    
    b = dataByMouse_first8(iM).burden_middle80_durOverTotalTime;
    
    if isempty(b)
        burdenSum_middle80(iM) = NaN;     % 没有事件就记 NaN
    else
        burdenSum_middle80(iM) = sum(b, 'omitnan');   % 只“加和”，不取平均
    end
end

% ========= 写回 dataByMouse_first8 的新 field =========
for iM = 1:nMice
    dataByMouse_first8(iM).burden_middle80_sum_durOverTotalTime = burdenSum_middle80(iM);
end

fprintf('已将每只老鼠 burden\\_middle80\\_durOverTotalTime 的总和写入 dataByMouse_first8.burden_middle80_sum_durOverTotalTime。\n');
%%
%% ========= 每只老鼠：对 burden_dur_over_totalTime 求和 =========
% 前提：dataByMouse_first8(i).burden_dur_over_totalTime 已经存在，
%       每只是一个 nEvent x 1 double（duration / totalTime）

nMice = numel(dataByMouse_first8);

% 预分配一个向量用于存每只老鼠的总 burden
burden_sum = nan(nMice,1);

% 检查字段是否存在
if ~isfield(dataByMouse_first8, 'burden_dur_over_totalTime')
    error('dataByMouse_first8 中不存在字段 burden_dur_over_totalTime，请先运行相应计算代码。');
end

for iM = 1:nMice
    b = dataByMouse_first8(iM).burden_dur_over_totalTime;
    
    if isempty(b)
        burden_sum(iM) = NaN;          % 没有事件就记 NaN
    else
        % 直接对这一只老鼠的所有 event burden 求和（不求平均）
        burden_sum(iM) = sum(b, 'omitnan');
    end
end

% ========= 把结果写回 dataByMouse_first8 的新字段 =========
% 新字段名：burden_sum_dur_over_totalTime（标量 double）
for iM = 1:nMice
    burdenByMouse_10min(iM).burden_sum_dur_over_totalTime = burden_sum(iM);
end

fprintf('已将每只老鼠 burden\\_dur\\_over\\_totalTime 的总和写入 dataByMouse_first8.burden_sum_dur_over_totalTime。\n');

%%
%% ========= 新算法：burden = duration / 总时间，且只保留 onset 后中间 90% 的 event =========
% totalTime = IED_time 最大值 - 最小值（秒）
% 中间 90%：按时间排序后的 event 序号，丢掉最早 5% 和最晚 5%
% 输出：
%   dataByMouse_first8(i).burden_middle90_durOverTotalTime      (vector，每个 event 的 burden)
%   dataByMouse_first8(i).burden_middle90_sum_durOverTotalTime  (scalar，总和)

T     = struct2table(dataByMouse_first8);
nMice = height(T);

% 预分配
burden_mid90_cell = cell(nMice,1);   % 每只鼠中间 90% 的 event burden（向量）
burden_mid90_sum  = nan(nMice,1);    % 每只鼠这些 burden 的总和（标量）

for iM = 1:nMice
    t = T.IED_time{iM};        % 秒，IED 时间戳
    d = T.norm_duration{iM};   % 对应 duration
    
    if isempty(t) || all(isnan(t)) || isempty(d)
        burden_mid90_cell{iM} = [];
        burden_mid90_sum(iM)  = NaN;
        continue;
    end
    
    % ---- 列向量 & 长度匹配 ----
    t = t(:);
    d = d(:);
    nEv = min(numel(t), numel(d));
    t = t(1:nEv);
    d = d(1:nEv);
    
    % 只对 t、d 都非 NaN 的 event 继续
    valid_td = ~isnan(t) & ~isnan(d);
    if ~any(valid_td)
        burden_mid90_cell{iM} = [];
        burden_mid90_sum(iM)  = NaN;
        continue;
    end
    
    t_valid = t(valid_td);
    d_valid = d(valid_td);
    
    % ---- 按时间排序，定义 onset 和 totalTime ----
    [t_sorted, order] = sort(t_valid);   % 升序
    d_sorted          = d_valid(order);
    
    if numel(t_sorted) < 2
        % 只有一个 event，totalTime 无法定义（或为 0） -> 全部 NaN
        burden_sorted_all = d_sorted * NaN;
    else
        t_first   = t_sorted(1);
        t_last    = t_sorted(end);
        totalTime = t_last - t_first;    % 秒
        
        if totalTime <= 0
            burden_sorted_all = d_sorted * NaN;
        else
            % 新定义：每个 event 的 burden = duration / totalTime
            burden_sorted_all = d_sorted / totalTime;   % [nValid_td x 1]
        end
    end
    
    nValid = numel(burden_sorted_all);
    
    % ---- 只保留中间 90% 的 event（按时间排序后的索引）----
    if nValid < 5
        % 太少就不过滤，全部保留
        burden_mid90 = burden_sorted_all;
    else
        idx_all = (1:nValid).';
        
        % 去掉最早 5% 和最晚 5%
        lowIdx  = floor(nValid * 0.05) + 1;   % > 5% 的第一个
        highIdx = ceil (nValid * 0.95);       % < 95% 的最后一个
        
        if lowIdx > highIdx
            % 极端情况下干脆全部保留
            burden_mid90 = burden_sorted_all;
        else
            keepMask    = (idx_all >= lowIdx) & (idx_all <= highIdx);
            burden_mid90 = burden_sorted_all(keepMask);
        end
    end
    
    % 保存这只老鼠中间 90% 的 burden 向量和总和
    burden_mid90_cell{iM} = burden_mid90;
    if isempty(burden_mid90)
        burden_mid90_sum(iM) = NaN;
    else
        burden_mid90_sum(iM) = sum(burden_mid90, 'omitnan');
    end
end

% ========= 写回 dataByMouse_first8 的新字段 =========

for iM = 1:nMice
    % 每个 event 的 burden（中间 90%，不平均）
    burdenByMouse_10min(iM).burden_middle90_durOverTotalTime = burden_mid90_cell{iM};
    
    % 这些 burden 的总和（标量）
    burdenByMouse_10min(iM).burden_middle90_sum_durOverTotalTime = burden_mid90_sum(iM);
end

fprintf(['已按 duration/总时间 + onset 后中间 90%% 计算 event-level burden，' ...
         '并写入 dataByMouse_first8.burden_middle90_durOverTotalTime 和 ' ...
         'dataByMouse_first8.burden_middle90_sum_durOverTotalTime。\n']);



%% ========= 新算法：burden = duration / 总时间，且只保留 onset 后中间 80% 的 event =========
% totalTime = IED_time 最大值 - 最小值（秒）
% 中间 80%：按时间排序后的 event 序号，丢掉最早 10% 和最晚 10%
% 输出：
%   dataByMouse_first8(i).burden_middle80_durOverTotalTime      (vector，每个 event 的 burden)
%   dataByMouse_first8(i).burden_middle80_sum_durOverTotalTime  (scalar，总和)

T     = struct2table(dataByMouse_first8);
nMice = height(T);

% 预分配
burden_mid80_cell = cell(nMice,1);   % 每只鼠中间 80% 的 event burden（向量）
burden_mid80_sum  = nan(nMice,1);    % 每只鼠这些 burden 的总和（标量）

for iM = 1:nMice
    t = T.IED_time{iM};        % 秒，IED 时间戳
    d = T.norm_duration{iM};   % 对应 duration
    
    if isempty(t) || all(isnan(t)) || isempty(d)
        burden_mid80_cell{iM} = [];
        burden_mid80_sum(iM)  = NaN;
        continue;
    end
    
    % ---- 列向量 & 长度匹配 ----
    t = t(:);
    d = d(:);
    nEv = min(numel(t), numel(d));
    t = t(1:nEv);
    d = d(1:nEv);
    
    % 只对 t、d 都非 NaN 的 event 继续
    valid_td = ~isnan(t) & ~isnan(d);
    if ~any(valid_td)
        burden_mid80_cell{iM} = [];
        burden_mid80_sum(iM)  = NaN;
        continue;
    end
    
    t_valid = t(valid_td);
    d_valid = d(valid_td);
    
    % ---- 按时间排序，定义 onset 和 totalTime ----
    [t_sorted, order] = sort(t_valid);   % 升序
    d_sorted          = d_valid(order);
    
    if numel(t_sorted) < 2
        % 只有一个 event，totalTime 无法定义（或为 0） -> 全部 NaN
        burden_sorted_all = d_sorted * NaN;
    else
        t_first   = t_sorted(1);
        t_last    = t_sorted(end);
        totalTime = t_last - t_first;    % 秒
        
        if totalTime <= 0
            burden_sorted_all = d_sorted * NaN;
        else
            % 新定义：每个 event 的 burden = duration / totalTime
            burden_sorted_all = d_sorted / totalTime;   % [nValid_td x 1]
        end
    end
    
    nValid = numel(burden_sorted_all);
    
    % ---- 只保留中间 80% 的 event（按时间排序后的索引）----
    if nValid < 5
        % 太少就不过滤，全部保留
        burden_mid80 = burden_sorted_all;
    else
        idx_all = (1:nValid).';
        
        % 去掉最早 10% 和最晚 10%
        lowIdx  = floor(nValid * 0.10) + 1;   % > 10% 的第一个
        highIdx = ceil (nValid * 0.90);       % < 90% 的最后一个
        
        if lowIdx > highIdx
            % 极端情况下干脆全部保留
            burden_mid80 = burden_sorted_all;
        else
            keepMask    = (idx_all >= lowIdx) & (idx_all <= highIdx);
            burden_mid80 = burden_sorted_all(keepMask);
        end
    end
    
    % 保存这只老鼠中间 80% 的 burden 向量和总和
    burden_mid80_cell{iM} = burden_mid80;
    if isempty(burden_mid80)
        burden_mid80_sum(iM) = NaN;
    else
        burden_mid80_sum(iM) = sum(burden_mid80, 'omitnan');
    end
end

% ========= 写回 dataByMouse_first8 的新字段 =========

for iM = 1:nMice
    % 每个 event 的 burden（中间 80%，不平均）
    burdenByMouse_10min(iM).burden_middle80_durOverTotalTime = burden_mid80_cell{iM};
    
    % 这些 burden 的总和（标量）
    burdenByMouse_10min(iM).burden_middle80_sum_durOverTotalTime = burden_mid80_sum(iM);
end

fprintf(['已按 duration/总时间 + onset 后中间 80%% 计算 event-level burden，' ...
         '并写入 dataByMouse_first8.burden_middle80_durOverTotalTime 和 ' ...
         'dataByMouse_first8.burden_middle80_sum_durOverTotalTime。\n']);




%% ========= 新算法：burden = duration / 总时间，且只保留 onset 后中间 95% 的 event =========
% totalTime = IED_time 最大值 - 最小值（秒）
% 中间 95%：按时间排序后的 event 序号，丢掉最早 2.5% 和最晚 2.5%
% 输出：
%   dataByMouse_first8(i).burden_middle95_durOverTotalTime      (vector，每个 event 的 burden)
%   dataByMouse_first8(i).burden_middle95_sum_durOverTotalTime  (scalar，总和)

T     = struct2table(dataByMouse_first8);
nMice = height(T);

% 预分配
burden_mid95_cell = cell(nMice,1);   % 每只鼠中间 95% 的 event burden（向量）
burden_mid95_sum  = nan(nMice,1);    % 每只鼠这些 burden 的总和（标量）

for iM = 1:nMice
    t = T.IED_time{iM};        % 秒，IED 时间戳
    d = T.norm_duration{iM};   % 对应 duration
    
    if isempty(t) || all(isnan(t)) || isempty(d)
        burden_mid95_cell{iM} = [];
        burden_mid95_sum(iM)  = NaN;
        continue;
    end
    
    % ---- 列向量 & 长度匹配 ----
    t = t(:);
    d = d(:);
    nEv = min(numel(t), numel(d));
    t = t(1:nEv);
    d = d(1:nEv);
    
    % 只对 t、d 都非 NaN 的 event 继续
    valid_td = ~isnan(t) & ~isnan(d);
    if ~any(valid_td)
        burden_mid95_cell{iM} = [];
        burden_mid95_sum(iM)  = NaN;
        continue;
    end
    
    t_valid = t(valid_td);
    d_valid = d(valid_td);
    
    % ---- 按时间排序，定义 onset 和 totalTime ----
    [t_sorted, order] = sort(t_valid);   % 升序
    d_sorted          = d_valid(order);
    
    if numel(t_sorted) < 2
        % 只有一个 event，totalTime 无法定义（或为 0） -> 全部 NaN
        burden_sorted_all = d_sorted * NaN;
    else
        t_first   = t_sorted(1);
        t_last    = t_sorted(end);
        totalTime = t_last - t_first;    % 秒
        
        if totalTime <= 0
            burden_sorted_all = d_sorted * NaN;
        else
            % 新定义：每个 event 的 burden = duration / totalTime
            burden_sorted_all = d_sorted / totalTime;   % [nValid_td x 1]
        end
    end
    
    nValid = numel(burden_sorted_all);
    
    % ---- 只保留中间 95% 的 event（按时间排序后的索引）----
    if nValid < 5
        % 太少就不过滤，全部保留
        burden_mid95 = burden_sorted_all;
    else
        idx_all = (1:nValid).';
        
        % 去掉最早 2.5% 和最晚 2.5%
        lowIdx  = floor(nValid * 0.025) + 1;   % > 2.5% 的第一个
        highIdx = ceil (nValid * 0.975);       % < 97.5% 的最后一个
        
        if lowIdx > highIdx
            % 极端情况下干脆全部保留
            burden_mid95 = burden_sorted_all;
        else
            keepMask    = (idx_all >= lowIdx) & (idx_all <= highIdx);
            burden_mid95 = burden_sorted_all(keepMask);
        end
    end
    
    % 保存这只老鼠中间 95% 的 burden 向量和总和
    burden_mid95_cell{iM} = burden_mid95;
    if isempty(burden_mid95)
        burden_mid95_sum(iM) = NaN;
    else
        burden_mid95_sum(iM) = sum(burden_mid95, 'omitnan');
    end
end

% ========= 写回 dataByMouse_first8 的新字段 =========

for iM = 1:nMice
    % 每个 event 的 burden（中间 95%，不平均）
    burdenByMouse_10min(iM).burden_middle95_durOverTotalTime = burden_mid95_cell{iM};
    
    % 这些 burden 的总和（标量）
    burdenByMouse_10min(iM).burden_middle95_sum_durOverTotalTime = burden_mid95_sum(iM);
end

fprintf(['已按 duration/总时间 + onset 后中间 95%% 计算 event-level burden，' ...
         '并写入 dataByMouse_first8.burden_middle95_durOverTotalTime 和 ' ...
         'dataByMouse_first8.burden_middle95_sum_durOverTotalTime。\n']);


%% === 对 burdenByMouse_10min 中从 'burden' 开始的所有字段取 log10 ===
% 假设从字段 'burden' 往后的字段都是数值型 (vector / scalar)
% 对每个字段都生成一个对应的 log10 字段，名字为原名后加 '_log10'

fn = fieldnames(burdenByMouse_10min);

% 找到 'burden' 在字段列表中的起始位置
idxStart = find(strcmp(fn,'burden'), 1, 'first');
if isempty(idxStart)
    error('在 burdenByMouse_10min 中没有找到字段 ''burden''。');
end

nMice = numel(burdenByMouse_10min);

for f = idxStart:numel(fn)
    fname    = fn{f};                 % 原字段名，例如 'burden_middle90_durOverTotalTime'
    newName  = [fname '_log10'];      % 新字段名，例如 'burden_middle90_durOverTotalTime_log10'
    
    for iM = 1:nMice
        val = burdenByMouse_10min(iM).(fname);   % 取出原来的数值
        
        if isempty(val)
            burdenByMouse_10min(iM).(newName) = [];
        else
            % 对整个向量 / 标量取 log10
            burdenByMouse_10min(iM).(newName) = log10(val);
        end
    end
end

fprintf('已对 burdenByMouse_10min 中从 ''burden'' 开始的所有字段取 log10，并依次添加 *_log10 字段。\n');


%% ===== 新算法：burden = duration / totalTime（每只鼠的总时间）=====
% totalTime = 该鼠 IED_time 最大值 - 最小值（秒）
% 把每个 event 的 burden = duration / totalTime
% 然后像原来一样按 0–10,10–20,...,40–50 min 的时间段存到 Excel

rowsPerGenoBurden_durOverTotalTime = cell(nGeno,1);
tabPoolBurden_durOverTotalTime     = cell(nGeno,1);

totalEv_valid_durOverTotalTime       = 0;
totalEv_assignedBin_durOverTotalTime = 0;

for iMouse = 1:nMice
    geno    = T.group{iMouse};
    gIdx    = find(strcmp(genoList, geno));
    fileStr = T.file{iMouse};
    
    t = T.IED_time{iMouse};
    d = T.norm_duration{iMouse};
    if isempty(t) || all(isnan(t)) || isempty(d)
        continue;
    end
    
    % ---- 列向量 & 长度匹配 ----
    t   = t(:);
    d   = d(:);
    nEv = min(numel(t), numel(d));
    t   = t(1:nEv);
    d   = d(1:nEv);
    
    % ---- 按时间排序，计算 totalTime ----
    [tSorted, order] = sort(t);
    dSorted          = d(order);
    
    if numel(tSorted) < 2
        % 只有一个 IED，无 meaningful totalTime
        continue;
    end
    
    t_first       = tSorted(1);
    t_last        = tSorted(end);
    totalTime_sec = t_last - t_first;          % 秒
    
    if totalTime_sec <= 0
        continue;
    end
    
    % ---- 转成相对时间（分钟），按 10 min bin 分箱 ----
    t_shift = tSorted - t_first;   % 秒
    t_min   = t_shift / 60;        % 分钟
    
    binIdx = discretize(t_min, edges_min);     % 0–10,10–20,...,40–50
    valid  = ~isnan(binIdx) & ~isnan(dSorted);
    if ~any(valid)
        continue;
    end
    
    binIdx_valid = binIdx(valid);
    d_valid      = dSorted(valid);            % 这些 event 的 duration（秒）
    
    % ===== 关键：新的 event-level burden 定义 =====
    % 每个 event 的 burden = duration / totalTime_sec（无量纲）
    burden_valid = d_valid / totalTime_sec;   % same size as d_valid
    
    totalEv_valid_durOverTotalTime = totalEv_valid_durOverTotalTime + numel(burden_valid);
    
    % ---- 按时间 bin 逐行写入表格（和原来结构一致）----
    rowsThisMouse = table();
    for b = 1:nBins
        idxB = (binIdx_valid == b);
        if ~any(idxB)
            continue;
        end
        
        burden_bin = burden_valid(idxB);   % 该鼠在该时间 bin 内的所有 event 的 burden
        totalEv_assignedBin_durOverTotalTime = ...
            totalEv_assignedBin_durOverTotalTime + numel(burden_bin);
        
        tabRow = table( ...
            iMouse, ...
            string(fileStr), ...
            string(binLabels{b}), ...
            binCenters(b), ...
            {burden_bin}, ...
            'VariableNames', {'MouseRowIdx','MouseFile', ...
                              'TimeBin','TimeBin_center_min','Burdens'});
        
        rowsThisMouse = [rowsThisMouse; tabRow];
    end
    
    if ~isempty(rowsThisMouse)
        if isempty(rowsPerGenoBurden_durOverTotalTime{gIdx})
            rowsPerGenoBurden_durOverTotalTime{gIdx} = rowsThisMouse;
        else
            rowsPerGenoBurden_durOverTotalTime{gIdx} = ...
                [rowsPerGenoBurden_durOverTotalTime{gIdx}; rowsThisMouse];
        end
    end
end

fprintf('Burden (dur/totalTime): total valid events (0–50 min) = %d\n', ...
    totalEv_valid_durOverTotalTime);
fprintf('Burden (dur/totalTime): assigned to bins              = %d\n', ...
    totalEv_assignedBin_durOverTotalTime);

% ===== pooled-by-timebin burden（同 genotype 内按时间段 pool）=====
for g = 1:nGeno
    if isempty(rowsPerGenoBurden_durOverTotalTime{g})
        continue;
    end
    tabG    = rowsPerGenoBurden_durOverTotalTime{g};
    tabPool = table();
    
    for b = 1:nBins
        thisLabel = string(binLabels{b});
        idxB      = (tabG.TimeBin == thisLabel);
        
        if ~any(idxB)
            Bur_all = [];
        else
            BurCells = tabG.Burdens(idxB);
            Bur_all  = cat(1, BurCells{:});   % 把所有鼠在该 bin 的 burden 拼在一起
        end
        
        rowB = table( ...
            thisLabel, ...
            binCenters(b), ...
            {Bur_all}, ...
            'VariableNames', {'TimeBin','TimeBin_center_min','Burdens'});
        
        tabPool = [tabPool; rowB];
    end
    
    tabPoolBurden_durOverTotalTime{g} = tabPool;
end

% ===== 导出到同一路径下的新 Excel 文件 =====
burdenFile_durOverTotalTime = fullfile(baseDir, 'Fig2_Burden_durOverTotalTime_10minBins.xlsx');
fprintf('Saving duration/totalTime burden to: %s\n', burdenFile_durOverTotalTime);

% 每个 genotype 一张 sheet（按鼠、按时间段，event-level cell）
for g = 1:nGeno
    if isempty(rowsPerGenoBurden_durOverTotalTime{g})
        continue;
    end
    genoName = genoList{g};
    writetable(rowsPerGenoBurden_durOverTotalTime{g}, ...
        burdenFile_durOverTotalTime, 'Sheet', genoName);
end

% 每个 genotype 再加一张 pooled sheet（时间段 pooled）
for g = 1:nGeno
    if isempty(tabPoolBurden_durOverTotalTime{g})
        continue;
    end
    genoName      = genoList{g};
    sheetNamePool = [genoName '_TimePool'];
    writetable(tabPoolBurden_durOverTotalTime{g}, ...
        burdenFile_durOverTotalTime, 'Sheet', sheetNamePool);
end




%% ========= 新算法：burden = AP_dur / 总时间，并生成 total / middle90 / middle80 =========
% 对每只老鼠：
%   totalTime = max(IED_time) - min(IED_time)   （秒）
%   burden_i  = AP_dur(i) / totalTime
%
% 输出到 dataByMouse_first8 中三个新字段：
%   1) burden_AP_over_totalTime_allEvents   : 所有 event 的 burden（与 IED_time 对齐，NaN 表示无效）
%   2) burden_AP_over_totalTime_middle90   : 中间 90% event 的 burden（按时间排序后的子集）
%   3) burden_AP_over_totalTime_middle80   : 中间 80% event 的 burden（按时间排序后的子集）

nMice = numel(dataByMouse_first8);

for iM = 1:nMice
    
    % ---- 取出 IED_time 和 AP_dur ----
    if ~isfield(dataByMouse_first8, 'IED_time') || ~isfield(dataByMouse_first8, 'AP_dur')
        error('dataByMouse_first8 必须包含 IED_time 和 AP_dur 字段。');
    end
    
    t  = dataByMouse_first8(iM).IED_time;   % 秒
    dA = dataByMouse_first8(iM).AP_dur;     % AP_dur，每个 event 的 duration
    
    if isempty(t) || all(isnan(t)) || isempty(dA)
        % 没有有效数据的情况
        dataByMouse_first8(iM).burden_AP_over_totalTime_allEvents  = [];
        dataByMouse_first8(iM).burden_AP_over_totalTime_middle90   = [];
        dataByMouse_first8(iM).burden_AP_over_totalTime_middle80   = [];
        continue;
    end
    
    % ---- 列向量 & 长度匹配 ----
    t  = t(:);
    dA = dA(:);
    nEv = min(numel(t), numel(dA));
    t  = t(1:nEv);
    dA = dA(1:nEv);
    
    % 只对 t 和 dA 都非 NaN 的事件计算 burden
    valid_td = ~isnan(t) & ~isnan(dA);
    
    if ~any(valid_td)
        dataByMouse_first8(iM).burden_AP_over_totalTime_allEvents  = nan(nEv,1);
        dataByMouse_first8(iM).burden_AP_over_totalTime_middle90   = [];
        dataByMouse_first8(iM).burden_AP_over_totalTime_middle80   = [];
        continue;
    end
    
    t_valid  = t(valid_td);
    dA_valid = dA(valid_td);
    
    % ---- 按时间排序，定义 totalTime ----
    [t_sorted, order] = sort(t_valid);   % 升序
    dA_sorted         = dA_valid(order);
    
    if numel(t_sorted) < 2
        % 只有一个事件，totalTime 无法定义（或为 0），全部记 NaN
        burden_sorted_all = dA_sorted * NaN;
    else
        t_first   = t_sorted(1);
        t_last    = t_sorted(end);
        totalTime = t_last - t_first;    % 秒
        
        if totalTime <= 0
            burden_sorted_all = dA_sorted * NaN;
        else
            % 新定义：每个 event 的 burden = AP_dur / totalTime
            burden_sorted_all = dA_sorted / totalTime;   % [nValid_td x 1]
        end
    end
    
    nValid = numel(burden_sorted_all);
    
    % ========= 1) 所有 event 的 burden（对齐到原始事件顺序） =========
    % 先把排序后的 burden 映射回 valid 事件的原顺序
    b_valid = nan(numel(t_valid),1);
    b_valid(order) = burden_sorted_all;         % 还原到 valid_td 的原事件顺序
    
    % 再映射回所有事件索引（包含无效事件的 NaN 占位）
    burden_allEvents = nan(nEv,1);
    burden_allEvents(valid_td) = b_valid;
    
    % ========= 2) 只保留中间 90% 的事件（按时间排序） =========
    % 丢掉最早 5% 和最晚 5% 的事件
    if nValid < 5
        % 有效事件太少时不过滤
        burden_mid90 = burden_sorted_all;
    else
        idx_all = (1:nValid).';
        lowIdx  = floor(nValid * 0.05) + 1;   % > 5% 的第一个
        highIdx = ceil (nValid * 0.95);       % < 95% 的最后一个
        
        if lowIdx > highIdx
            burden_mid90 = burden_sorted_all;
        else
            keepMask   = (idx_all >= lowIdx) & (idx_all <= highIdx);
            burden_mid90 = burden_sorted_all(keepMask);
        end
    end
    
    % ========= 3) 只保留中间 80% 的事件（按时间排序） =========
    % 丢掉最早 10% 和最晚 10% 的事件
    if nValid < 5
        burden_mid80 = burden_sorted_all;
    else
        idx_all = (1:nValid).';
        lowIdx  = floor(nValid * 0.10) + 1;   % > 10% 的第一个
        highIdx = ceil (nValid * 0.90);       % < 90% 的最后一个
        
        if lowIdx > highIdx
            burden_mid80 = burden_sorted_all;
        else
            keepMask   = (idx_all >= lowIdx) & (idx_all <= highIdx);
            burden_mid80 = burden_sorted_all(keepMask);
        end
    end
    
    % ========= 写回到 struct =========
    dataByMouse_first8(iM).burden_AP_over_totalTime_allEvents = burden_allEvents;  % nEv x 1
    dataByMouse_first8(iM).burden_AP_over_totalTime_middle90  = burden_mid90;      % k90 x 1
    dataByMouse_first8(iM).burden_AP_over_totalTime_middle80  = burden_mid80;      % k80 x 1
end

fprintf(['已按 AP\\_dur / 总时间 计算每个 event 的 burden，' ...
         '并写入 dataByMouse_first8.burden_AP_over_totalTime_allEvents, ' ...
         'burden_AP_over_totalTime_middle90, burden_AP_over_totalTime_middle80。\n']);


%%
%% ===== 新算法：burden = duration / totalTime（每只鼠的总时间）=====
% totalTime = 该鼠 IED_time 最大值 - 最小值（秒）
% 把每个 event 的 burden = duration / totalTime
% 然后像原来一样按 0–10,10–20,...,40–50 min 的时间段存到 Excel

rowsPerGenoBurden_durOverTotalTime = cell(nGeno,1);
tabPoolBurden_durOverTotalTime     = cell(nGeno,1);

totalEv_valid_durOverTotalTime       = 0;
totalEv_assignedBin_durOverTotalTime = 0;

for iMouse = 1:nMice
    geno    = T.group{iMouse};
    gIdx    = find(strcmp(genoList, geno));
    fileStr = T.file{iMouse};
    
    t = T.IED_time{iMouse};
    d = T.norm_duration{iMouse};
    if isempty(t) || all(isnan(t)) || isempty(d)
        continue;
    end
    
    % ---- 列向量 & 长度匹配 ----
    t   = t(:);
    d   = d(:);
    nEv = min(numel(t), numel(d));
    t   = t(1:nEv);
    d   = d(1:nEv);
    
    % ---- 按时间排序，计算 totalTime ----
    [tSorted, order] = sort(t);
    dSorted          = d(order);
    
    if numel(tSorted) < 2
        % 只有一个 IED，无 meaningful totalTime
        continue;
    end
    
    t_first       = tSorted(1);
    t_last        = tSorted(end);
    totalTime_sec = t_last - t_first;          % 秒
    
    if totalTime_sec <= 0
        continue;
    end
    
    % ---- 转成相对时间（分钟），按 10 min bin 分箱 ----
    t_shift = tSorted - t_first;   % 秒
    t_min   = t_shift / 60;        % 分钟
    
    binIdx = discretize(t_min, edges_min);     % 0–10,10–20,...,40–50
    valid  = ~isnan(binIdx) & ~isnan(dSorted);
    if ~any(valid)
        continue;
    end
    
    binIdx_valid = binIdx(valid);
    d_valid      = dSorted(valid);            % 这些 event 的 duration（秒）
    
    % ===== 关键：新的 event-level burden 定义 =====
    % 每个 event 的 burden = duration / totalTime_sec（无量纲）
    burden_valid = d_valid / totalTime_sec;   % same size as d_valid
    
    totalEv_valid_durOverTotalTime = totalEv_valid_durOverTotalTime + numel(burden_valid);
    
    % ---- 按时间 bin 逐行写入表格（和原来结构一致）----
    rowsThisMouse = table();
    for b = 1:nBins
        idxB = (binIdx_valid == b);
        if ~any(idxB)
            continue;
        end
        
        burden_bin = burden_valid(idxB);   % 该鼠在该时间 bin 内的所有 event 的 burden
        totalEv_assignedBin_durOverTotalTime = ...
            totalEv_assignedBin_durOverTotalTime + numel(burden_bin);
        
        tabRow = table( ...
            iMouse, ...
            string(fileStr), ...
            string(binLabels{b}), ...
            binCenters(b), ...
            {burden_bin}, ...
            'VariableNames', {'MouseRowIdx','MouseFile', ...
                              'TimeBin','TimeBin_center_min','Burdens'});
        
        rowsThisMouse = [rowsThisMouse; tabRow];
    end
    
    if ~isempty(rowsThisMouse)
        if isempty(rowsPerGenoBurden_durOverTotalTime{gIdx})
            rowsPerGenoBurden_durOverTotalTime{gIdx} = rowsThisMouse;
        else
            rowsPerGenoBurden_durOverTotalTime{gIdx} = ...
                [rowsPerGenoBurden_durOverTotalTime{gIdx}; rowsThisMouse];
        end
    end
end

fprintf('Burden (dur/totalTime): total valid events (0–50 min) = %d\n', ...
    totalEv_valid_durOverTotalTime);
fprintf('Burden (dur/totalTime): assigned to bins              = %d\n', ...
    totalEv_assignedBin_durOverTotalTime);

% ===== pooled-by-timebin burden（同 genotype 内按时间段 pool）=====
for g = 1:nGeno
    if isempty(rowsPerGenoBurden_durOverTotalTime{g})
        continue;
    end
    tabG    = rowsPerGenoBurden_durOverTotalTime{g};
    tabPool = table();
    
    for b = 1:nBins
        thisLabel = string(binLabels{b});
        idxB      = (tabG.TimeBin == thisLabel);
        
        if ~any(idxB)
            Bur_all = [];
        else
            BurCells = tabG.Burdens(idxB);
            Bur_all  = cat(1, BurCells{:});   % 把所有鼠在该 bin 的 burden 拼在一起
        end
        
        rowB = table( ...
            thisLabel, ...
            binCenters(b), ...
            {Bur_all}, ...
            'VariableNames', {'TimeBin','TimeBin_center_min','Burdens'});
        
        tabPool = [tabPool; rowB];
    end
    
    tabPoolBurden_durOverTotalTime{g} = tabPool;
end

% ===== 导出到同一路径下的新 Excel 文件 =====
burdenFile_durOverTotalTime = fullfile(baseDir, 'Fig2_Burden_durOverTotalTime_10minBins.xlsx');
fprintf('Saving duration/totalTime burden to: %s\n', burdenFile_durOverTotalTime);

% 每个 genotype 一张 sheet（按鼠、按时间段，event-level cell）
for g = 1:nGeno
    if isempty(rowsPerGenoBurden_durOverTotalTime{g})
        continue;
    end
    genoName = genoList{g};
    writetable(rowsPerGenoBurden_durOverTotalTime{g}, ...
        burdenFile_durOverTotalTime, 'Sheet', genoName);
end

% 每个 genotype 再加一张 pooled sheet（时间段 pooled）
for g = 1:nGeno
    if isempty(tabPoolBurden_durOverTotalTime{g})
        continue;
    end
    genoName      = genoList{g};
    sheetNamePool = [genoName '_TimePool'];
    writetable(tabPoolBurden_durOverTotalTime{g}, ...
        burdenFile_durOverTotalTime, 'Sheet', sheetNamePool);
end


%% ========= 对 AP_dur/totalTime 版本的 burden 做 per-mouse 求和 =========
% 前提：dataByMouse_first8(i) 已经有三个 vector 字段：
%   - burden_AP_over_totalTime_allEvents
%   - burden_AP_over_totalTime_middle90
%   - burden_AP_over_totalTime_middle80
%
% 现在：对每只老鼠，分别对这三个向量求和（omitnan），
%       写入三个新的 scalar 字段：
%   - burden_AP_over_totalTime_allEvents_sum
%   - burden_AP_over_totalTime_middle90_sum
%   - burden_AP_over_totalTime_middle80_sum

nMice = numel(burdenByMouse_10min);

% 简单检查一下字段是否存在
needFields = { ...
    'burden_AP_over_totalTime_allEvents', ...
    'burden_AP_over_totalTime_middle90', ...
    'burden_AP_over_totalTime_middle80'};

for f = 1:numel(needFields)
    if ~isfield(dataByMouse_first8, needFields{f})
        error('dataByMouse_first8 中缺少字段：%s，请先运行前一步的 burden 计算代码。', needFields{f});
    end
end

% 预分配
sum_allEvents = nan(nMice,1);
sum_mid90     = nan(nMice,1);
sum_mid80     = nan(nMice,1);

for iM = 1:nMice
    
    % -------- 1) allEvents 的总和 --------
    b_all = dataByMouse_first8(iM).burden_AP_over_totalTime_allEvents;
    if isempty(b_all)
        sum_allEvents(iM) = NaN;
    else
        sum_allEvents(iM) = sum(b_all, 'omitnan');
    end
    
    % -------- 2) middle90 的总和 --------
    b_mid90 = dataByMouse_first8(iM).burden_AP_over_totalTime_middle90;
    if isempty(b_mid90)
        sum_mid90(iM) = NaN;
    else
        sum_mid90(iM) = sum(b_mid90, 'omitnan');
    end
    
    % -------- 3) middle80 的总和 --------
    b_mid80 = dataByMouse_first8(iM).burden_AP_over_totalTime_middle80;
    if isempty(b_mid80)
        sum_mid80(iM) = NaN;
    else
        sum_mid80(iM) = sum(b_mid80, 'omitnan');
    end
end

% ========= 写回到 dataByMouse_first8 的新字段 =========

for iM = 1:nMice
    burdenByMouse_10min(iM).burden_AP_over_totalTime_allEvents_sum = sum_allEvents(iM);
    burdenByMouse_10min(iM).burden_AP_over_totalTime_middle90_sum  = sum_mid90(iM);
    burdenByMouse_10min(iM).burden_AP_over_totalTime_middle80_sum  = sum_mid80(iM);
end

fprintf(['已将每只老鼠 AP\\_dur/totalTime burden 的总和写入：\n' ...
    '  dataByMouse_first8.burden_AP_over_totalTime_allEvents_sum\n' ...
    '  dataByMouse_first8.burden_AP_over_totalTime_middle90_sum\n' ...
    '  dataByMouse_first8.burden_AP_over_totalTime_middle80_sum\n']);


