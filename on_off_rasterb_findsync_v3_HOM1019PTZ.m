%% =========================================================
%  Depth-based raster + LFP + CWT (type1 & type2 only)
%
%  整合：
%   - code1: spike / depth / unitType + depth-based raster + summary
%   - code2: loading_lfp + baseline_correction + detect_swd + CWT / LFP 画图思路
%
%  运行前请确认：loading_lfp / baseline_correction / plot_swd_spectrogram_0_60
%                 / plot_filtered_lfp / detect_swd 这些函数在路径上。
% =========================================================
% === 自定义红白蓝渐变 colormap ===
red    = [227 73 50]/255;    % +1
white  = [1 1 1];            % 0
blue   = [65 104 174]/255;   % -1
custom_cmap = [linspace(blue(1), white(1), 128)', ...
                linspace(blue(2), white(2), 128)', ...
                linspace(blue(3), white(3), 128)';
                linspace(white(1), red(1), 128)', ...
                linspace(white(2), red(2), 128)', ...
                linspace(white(3), red(3), 128)'];
%% =========================================================
% === 0) 路径设置：AP & LFP ===
% =========================================================
% AP (spike) 路径
base_path = 'D:\2025PTZ\2025-05-01_1019_HOM-female-adult\2025-05-01_13-03-35\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP';

% LFP 路径：从 AP 路径自动替换成 ProbeA-LFP
lfp_path = strrep(base_path, 'ProbeA-AP', 'ProbeA-LFP');

%% =========================================================
% === 0.1) 加载 LFP d，并做 SWD detection（code2 部分） ===
% =========================================================
if exist(lfp_path,'dir')
    fprintf('Changing directory to LFP folder:\n  %s\n', lfp_path);
    cwd = pwd;
    cd(lfp_path);

    % —— 用户自己的脚本：应加载 d 和 original_sampling_rate 等 ——
    % 例如：d (channels × time), original_sampling_rate = 2500
    loading_lfp;
     d_shift = [d(:, 11747:end), zeros(size(d,1), 11747)];
%
    d = d_shift;
    cd(cwd);
else
    warning('LFP folder not found: %s', lfp_path);
end

% 确保 original_sampling_rate 存在，没有就用 2500 Hz
if ~exist('original_sampling_rate','var') || isempty(original_sampling_rate)
    original_sampling_rate = 2500;
end

% 参照 code2：baseline_correction + spectrogram + filtered LFP + detect_swd
if exist('d','var') && ~isempty(d)
    fprintf('Running baseline_correction / plot_swd_spectrogram_0_60 / plot_filtered_lfp / detect_swd ...\n');

    corrected_baseline = baseline_correction(mean(d), original_sampling_rate);
    plot_swd_spectrogram_0_60(corrected_baseline, original_sampling_rate);
    A = plot_filtered_lfp(mean(d), original_sampling_rate);

    % 下面三行与 code2 一致；detect_swd 内部应更新 swd_events（2500 Hz）
    detect_swd(corrected_baseline);
    detect_swd(A);
    detect_swd(mean(d));

    % 此时应在 workspace 中得到：
    %   d               : LFP
    %   original_sampling_rate
    %   swd_events      : cell，LFP 采样点索引
else
    warning('Variable d not loaded by loading_lfp; CWT/LFP panels will show "not found".');
end
%%

swd_events = swd_events_QualityM; 

swd_events_2500Hz = swd_events;

swd_events_30000Hz = cell(size(swd_events));% Replace this with your actual variable
original_sampling_rate = 2500; % Original sampling rate in Hz
new_sampling_rate = 30000; % New sampling rate in Hz
scaling_factor = new_sampling_rate / original_sampling_rate;
% Populate each component of the cell array
for i = 1:length(swd_events)
    swd_events_30000Hz{i} = swd_events{i} * scaling_factor; % Scale time points
end

% Display the converted SWD events
disp('SWD events converted to 30000Hz:');
disp(swd_events_30000Hz);


% Assuming swd_events_2500Hz is a 1x13 cell array containing the time points
% recorded at 2500 Hz

% Define the original sampling rate

% Convert time points from samples to seconds
swd_events_seconds = cell(size(swd_events)); % Initialize new cell array

% Convert each time point in the cell array to seconds
for i = 1:length(swd_events)
    swd_events_seconds{i} = swd_events{i} / original_sampling_rate; % Convert to seconds
end

% Display the converted SWD events in seconds
disp('SWD events in seconds:');
disp(swd_events_seconds);




%% =========================================================
% === 1) Load spike times & templates (.npy) ===
% =========================================================
spike_times_path      = fullfile(base_path, 'spike_times.npy');
spike_templates_path  = fullfile(base_path, 'spike_templates.npy');
excel_path            = fullfile(base_path, 'clustertwo.xlsx');
unitType_path         = fullfile(base_path, 'unitTypes.mat');

try
    fprintf('Loading spike_times.npy ...\n');
    spike_time_wt = readNPY(spike_times_path);
    spike_time_full_wt = double(spike_time_wt) / 30000;  % convert to seconds
    fprintf('  -> %d spike timestamps loaded.\n', numel(spike_time_full_wt));
catch ME
    error('Failed to load spike_times.npy: %s', ME.message);
end

try
    fprintf('Loading spike_templates.npy ...\n');
    spike_templates_wt = readNPY(spike_templates_path);
    % Kilosort 是 0-based，MATLAB 用 1-based，这里 +1
    spike_templates_full_wt = double(spike_templates_wt) + 1;
    fprintf('  -> %d template IDs loaded.\n', numel(spike_templates_full_wt));
catch ME
    error('Failed to load spike_templates.npy: %s', ME.message);
end

%% =========================================================
% 2) Load cluster info (clustertwo.xlsx)
% =========================================================
try
    fprintf('\nLoading cluster info from: %s\n', excel_path);
    T = readtable(excel_path);  % preferred if has headers

    if width(T) >= 2
        col2 = T{:,2};
        if ~isnumeric(col2)
            col2 = str2double(string(col2)); % convert text to numbers
        end
    else
        warning('Excel file has fewer than 2 columns.');
        col2 = [];
    end
catch ME
    warning('Failed to read Excel as table: %s\nRetrying as matrix...', ME.message);
    try
        X = readmatrix(excel_path);
        if size(X,2) < 2
            error('Excel file has fewer than 2 columns.');
        end
        col2 = X(:,2);
        T = array2table(X);
    catch ME2
        error('Failed to load Excel file: %s', ME2.message);
    end
end

%% =========================================================
% 3) Load unitType.mat
% =========================================================
try
    fprintf('\nLoading unitType from: %s\n', unitType_path);
    tmp = load(unitType_path);
    if isfield(tmp, 'unitType')
        unitType = tmp.unitType;
    else
        fn = fieldnames(tmp);
        unitType = tmp.(fn{1});
        warning('unitType variable not explicitly found. Using "%s" instead.', fn{1});
    end
    unitType = unitType(:); % make column
    fprintf('unitType length = %d\n', numel(unitType));
catch ME
    error('Failed to load unitType.mat: %s', ME.message);
end

%% =========================================================
% 4) Build resultArray_wt from spike_times & templates
% =========================================================
spike_channel_wt = [spike_time_full_wt, spike_templates_full_wt];

[~, sortIdx_wt] = sort(spike_channel_wt(:, 2));
sortedArray2D_wt = spike_channel_wt(sortIdx_wt, :);

uniqueValues_wt = unique(sortedArray2D_wt(:, 2));   % 每一个就是一个 template/neuron
groupedRows_wt = accumarray(sortedArray2D_wt(:, 2), sortedArray2D_wt(:, 1), [], @(x) {x'});

maxGroupSize_wt = max(cellfun(@length, groupedRows_wt));
resultArray_wt = nan(length(uniqueValues_wt), maxGroupSize_wt + 1);
for i = 1:length(uniqueValues_wt)
    resultArray_wt(i, 1) = uniqueValues_wt(i);                  % 第1列放 template ID
    resultArray_wt(i, 2:length(groupedRows_wt{i}) + 1) = groupedRows_wt{i};  % 后面是 spike times
end
disp('✅ Spike data (times & templates) loaded and grouped successfully.');

%% =========================================================
% 5) Sanity checks
% =========================================================
if ~exist('T', 'var') || isempty(T)
    error('Variable T (from clustertwo.xlsx) not loaded.');
end
if ~exist('unitType', 'var') || isempty(unitType)
    error('Variable unitType not loaded.');
end
if ~exist('resultArray_wt', 'var') || isempty(resultArray_wt)
    error('Variable resultArray_wt not created.');
end

disp('✅ Successfully loaded all 4 files: spike_times, spike_templates, clustertwo.xlsx, and unitType.mat.');

%% =========================================================
% 6) unitType 统计 + depth 列提取
% =========================================================
num_type1 = sum(unitType == 1);
num_type2 = sum(unitType == 2);
fprintf('Number of neurons with unitType == 1: %d\n', num_type1);
fprintf('Number of neurons with unitType == 2: %d\n', num_type2);

if istable(T)
    raw_depth = T{:,4};      % 可能是 double，也可能是 cell
else
    raw_depth = T(:,4);
end

Ndepth = numel(raw_depth);
depth_col = nan(Ndepth,1);

if iscell(raw_depth)
    for i = 1:Ndepth
        val = raw_depth{i};
        if isempty(val)
            depth_col(i) = NaN;
        elseif isnumeric(val)
            if isscalar(val)
                depth_col(i) = double(val);
            else
                depth_col(i) = double(val(1));
            end
        elseif ischar(val) || isstring(val)
            tmp = str2double(val);
            if ~isnan(tmp)
                depth_col(i) = tmp;
            else
                depth_col(i) = NaN;
            end
        else
            depth_col(i) = NaN;
        end
    end
else
    depth_col = double(raw_depth);
end

idx_type1 = (unitType == 1);
idx_type2 = (unitType == 2);

depths_type1 = depth_col(idx_type1);
depths_type2 = depth_col(idx_type2);

depths_type1_clean = depths_type1(~isnan(depths_type1));
depths_type2_clean = depths_type2(~isnan(depths_type2));

isCortex_type1   = depths_type1_clean > 2625;
isStriatum_type1 = depths_type1_clean <= 2625;

isCortex_type2   = depths_type2_clean > 2625;
isStriatum_type2 = depths_type2_clean <= 2625;

fprintf('\nType 1 neurons (after removing NaN): %d\n', numel(depths_type1_clean));
fprintf('  Cortex  (>2625): %d\n', sum(isCortex_type1));
fprintf('  Striatum(<=2625): %d\n', sum(isStriatum_type1));

fprintf('\nType 2 neurons (after removing NaN): %d\n', numel(depths_type2_clean));
fprintf('  Cortex  (>2625): %d\n', sum(isCortex_type2));
fprintf('  Striatum(<=2625): %d\n', sum(isStriatum_type2));

%% =========================================================
% 7) 只保留 unitType==1 or 2 的 neuron (resultArray_type12)
% =========================================================
template_id_per_row = resultArray_wt(:,1);    % [nNeuron, 1]

valid_rows = template_id_per_row <= numel(unitType);
template_id_per_row = template_id_per_row(valid_rows);
resultArray_wt = resultArray_wt(valid_rows, :);

nNeuron = size(resultArray_wt, 1);

keep_type12 = (unitType(template_id_per_row) == 1) | (unitType(template_id_per_row) == 2);
template_id_per_row = template_id_per_row(keep_type12);
resultArray_type12  = resultArray_wt(keep_type12, :);

fprintf('\n✅ After filtering: %d neurons remain (unitType==1 or 2).\n', size(resultArray_type12,1));

%% =========================================================
% 8) 统一参数：时间窗口、STR/CTX depth 排序
% =========================================================
win = [2905 2995];% WT1347PTZ
%win = [260 350]; %  HOM1019baseline      可视时间窗口 [s]
x_start = win(1);
x_end   = win(2);
bin_width = 0.05;            % summary histogram bin 宽度 (100 ms)
depth_threshold = 2625;

depth_for_type12 = nan(size(template_id_per_row));
for i = 1:numel(template_id_per_row)
    tid = template_id_per_row(i);
    if tid <= numel(depth_col)
        depth_for_type12(i) = depth_col(tid);
    else
        depth_for_type12(i) = NaN;
    end
end

is_ctx = depth_for_type12 > depth_threshold;
is_str = ~is_ctx | isnan(depth_for_type12);   % NaN 归到 STR 一侧 (上面)

[~,ord_str] = sort(depth_for_type12(is_str), 'ascend', 'MissingPlacement','last');
[~,ord_ctx] = sort(depth_for_type12(is_ctx), 'ascend', 'MissingPlacement','last');
idx_str = find(is_str); idx_str = idx_str(ord_str);
idx_ctx = find(is_ctx); idx_ctx = idx_ctx(ord_ctx);

display_order = [idx_str; idx_ctx];
split_y = numel(idx_str);

%% =========================================================
% 9) Figure 1: STR/CTX split raster + summary，上方 CWT + LFP+SWD
% =========================================================
figure('Name','Raster — type1/2 with STR/CTX split + LFP & CWT & summary',...
       'Position',[100 100 900 900]);
tl = tiledlayout(4,1,'Padding','compact','TileSpacing','compact');

%% (1) CWT panel
nexttile(tl,1); hold on;
if exist('d','var') && ~isempty(d)
    fsLFP = original_sampling_rate;
    scale_factor = 0.195;
    x_lfp = mean(d(: , :), 1) * scale_factor;
    x_lfp = x_lfp(:);

    bp = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',1,'HalfPowerFrequency2',60, ...
        'SampleRate',fsLFP);
    x_filt = filtfilt(bp, x_lfp);
    x_filt = detrend(x_filt, 'linear');

    ix0 = max(1, floor(x_start*fsLFP)+1);
    ix1 = min(numel(x_filt), ceil(x_end*fsLFP));
    sig_win = x_filt(ix0:ix1);
    t_win   = ((ix0:ix1)-1)/fsLFP;

    fb = cwtfilterbank('SignalLength',numel(sig_win), 'SamplingFrequency',fsLFP, ...
        'Wavelet','morse','VoicesPerOctave',24,'TimeBandwidth',60, ...
        'FrequencyLimits',[0.5 60]);
    [cfs, f] = wt(fb, sig_win);

    P = abs(cfs).^2;
    Zlog = log10(P + eps);
    mu = mean(Zlog, 2);
    sd = std(Zlog, 0, 2) + eps;
    Zz = (Zlog - mu) ./ sd;

    clim = [-1, 2];
    imagesc(t_win, f, Zz); axis xy
    xlim([x_start, x_end]); ylim([0.5 60])
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('CWT (Morse, 0–60 Hz) — per-freq z-score of log(power)');
    %colormap(jet);
     caxis(clim);
    colormap(custom_cmap);
    cb = colorbar; ylabel(cb, 'z (per-freq log power)');
else
    text(0.5,0.5,'LFP (d) not found in workspace','HorizontalAlignment','center');
    axis off;
end
hold off;

%% (2) LFP + SWD detection panel
nexttile(tl,2); hold on;
if exist('d','var') && ~isempty(d)
    fsLFP = original_sampling_rate;
    dmean = mean(d,1);
    nLFP  = numel(dmean);
    tLFP  = (0:nLFP-1)/fsLFP;

    plot(tLFP, dmean, 'Color', [0.6 0.6 0.6]); hold on;
    if exist('swd_events','var') && ~isempty(swd_events)
        for iEvt = 1:numel(swd_events)
            evt = swd_events{iEvt};
            if isempty(evt), continue; end
            evt_t = evt / fsLFP;
            plot(evt_t, dmean(evt), 'r', 'LineWidth', 1.5);
        end
    end

    xlim([x_start, x_end]);
    if ~all(isnan(dmean))
        ylim([min(dmean) max(dmean)] * 1.1);
    end
    xlabel('Time (s)');
    ylabel('Amplitude (a.u. or µV)');
    title('LFP mean(d) with SWD detection overlay');
else
    text(0.5,0.5,'LFP (d) not found in workspace','HorizontalAlignment','center');
    axis off;
end
hold off;

%% (3) Raster: STR 上, CTX 下
nexttile(tl,3); hold on;
for k = 1:numel(display_order)
    row_idx = display_order(k);
    template_id = template_id_per_row(row_idx);
    spikes = resultArray_type12(row_idx, 2:end);
    spikes = spikes(spikes > 0 & spikes >= win(1) & spikes <= win(2));
    if isempty(spikes), continue; end

    if unitType(template_id) == 1
        c = 'r';
    else
        c = 'b';
    end
    scatter(spikes, k*ones(size(spikes)), 8, c, '.');
end
if split_y > 0 && split_y < numel(display_order)
    plot([win(1) win(2)], [split_y+0.5 split_y+0.5], 'k--', 'LineWidth', 1);
end
xlim(win);
ylim([0.5, numel(display_order)+0.5]);
xlabel('Time (s)');
ylabel('Neuron (STR top, CTX bottom)');
title('Raster (unitType 1=red, 2=blue) — STR / CTX split');
grid on;
hold off;

%% (4) Summary histogram
nexttile(tl,4); hold on;
all_spikes = [];
for k = 1:size(resultArray_type12,1)
    spikes = resultArray_type12(k, 2:end);
    spikes = spikes(spikes > 0 & spikes >= win(1) & spikes <= win(2));
    all_spikes = [all_spikes, spikes]; %#ok<AGROW>
end
edges = win(1):bin_width:win(2);
counts = histcounts(all_spikes, edges);
bar(edges(1:end-1), counts, 1);
xlim(win);
xlabel('Time (s)');
ylabel('Spike count (all type1/2)');
title('Summary histogram (type1 + type2)');
grid on;
hold off;

%% =========================================================
% 10) Figure 2: 绝对 depth raster + summary，上方 CWT + LFP+SWD
% =========================================================
figure('Name','Raster — type1/2 in absolute depth + LFP & CWT & summary',...
       'Position',[150 150 900 900]);
tl2 = tiledlayout(4,1,'Padding','compact','TileSpacing','compact');

%% (1) CWT panel
nexttile(tl2,1); hold on;
if exist('d','var') && ~isempty(d)
    fsLFP = original_sampling_rate;
    scale_factor = 0.195;
    x_lfp = mean(d(: , :), 1) * scale_factor;
    x_lfp = x_lfp(:);

    bp = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',1,'HalfPowerFrequency2',60, ...
        'SampleRate',fsLFP);
    x_filt = filtfilt(bp, x_lfp);
    x_filt = detrend(x_filt, 'linear');

    ix0 = max(1, floor(x_start*fsLFP)+1);
    ix1 = min(numel(x_filt), ceil(x_end*fsLFP));
    sig_win = x_filt(ix0:ix1);
    t_win   = ((ix0:ix1)-1)/fsLFP;

    fb = cwtfilterbank('SignalLength',numel(sig_win), 'SamplingFrequency',fsLFP, ...
        'Wavelet','morse','VoicesPerOctave',24,'TimeBandwidth',60, ...
        'FrequencyLimits',[0.5 60]);
    [cfs, f] = wt(fb, sig_win);
    P = abs(cfs).^2;
    Zlog = log10(P + eps);
    mu = mean(Zlog, 2);
    sd = std(Zlog, 0, 2) + eps;
    Zz = (Zlog - mu) ./ sd;

    clim = [-1, 2];
    imagesc(t_win, f, Zz); axis xy
    xlim([x_start, x_end]); ylim([0.5 60])
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('CWT (Morse, 0–60 Hz) — per-freq z-score of log(power)');
%     colormap(jet);
    colormap(custom_cmap);
    caxis(clim);
    cb = colorbar; ylabel(cb, 'z (per-freq log power)');
else
    text(0.5,0.5,'LFP (d) not found in workspace','HorizontalAlignment','center');
    axis off;
end
hold off;

%% (2) LFP + SWD detection panel
nexttile(tl2,2); hold on;
if exist('d','var') && ~isempty(d)
    fsLFP = original_sampling_rate;
    dmean = mean(d,1);
    nLFP  = numel(dmean);
    tLFP  = (0:nLFP-1)/fsLFP;

    plot(tLFP, dmean, 'Color', [0.6 0.6 0.6]); hold on;
    if exist('swd_events','var') && ~isempty(swd_events)
        for iEvt = 1:numel(swd_events)
            evt = swd_events{iEvt};
            if isempty(evt), continue; end
            evt_t = evt / fsLFP;
            plot(evt_t, dmean(evt), 'r', 'LineWidth', 1.5);
        end
    end

    xlim([x_start, x_end]);
    if ~all(isnan(dmean))
        ylim([min(dmean) max(dmean)] * 1.1);
    end
    xlabel('Time (s)');
    ylabel('Amplitude (a.u. or µV)');
    title('LFP mean(d) with SWD detection overlay');
else
    text(0.5,0.5,'LFP (d) not found in workspace','HorizontalAlignment','center');
    axis off;
end
hold off;

%% (3) Raster: 绝对 depth
nexttile(tl2,3); hold on;
for i = 1:size(resultArray_type12,1)
    template_id = template_id_per_row(i);
    spikes = resultArray_type12(i, 2:end);
    spikes = spikes(spikes > 0 & spikes >= win(1) & spikes <= win(2));
    if isempty(spikes), continue; end

    if template_id <= numel(depth_col)
        this_depth = depth_col(template_id);
    else
        this_depth = NaN;
    end
    if isnan(this_depth), continue; end

    if unitType(template_id) == 1
        c = 'r';
    else
        c = 'b';
    end
    scatter(spikes, this_depth*ones(size(spikes)), 8, c, '.');
end
xlim(win);
ylim([0 3840]);           % 根据探针深度修改
set(gca,'YDir','normal');
xlabel('Time (s)');
ylabel('Depth (\mum or channel depth)');
title('Raster (unitType 1=red, 2=blue) — absolute depth');
grid on;
hold off;

%% (4) Summary histogram
nexttile(tl2,4); hold on;
all_spikes = [];
for k = 1:size(resultArray_type12,1)
    spikes = resultArray_type12(k, 2:end);
    spikes = spikes(spikes > 0 & spikes >= win(1) & spikes <= win(2));
    all_spikes = [all_spikes, spikes]; %#ok<AGROW>
end
edges = win(1):bin_width:win(2);
counts = histcounts(all_spikes, edges);
bar(edges(1:end-1), counts, 1);
xlim(win);
xlabel('Time (s)');
ylabel('Spike count (all type1/2)');
title('Summary histogram (type1 + type2)');
grid on;
hold off;
%%
%% =========================================================
% 11) DROP-IN: 保存 Figure1 的 raster 数据 + 按 unitID 重新绘图
% =========================================================

% ------- 11.1 抽取 Figure1 的 raster 数据 -------
nRows = numel(display_order);
raster_points_depthOrder = [];           % Nx2: [time, rowIndex]
raster_unit_ids          = nan(nRows,1); % 每一行对应的 template ID
raster_unit_depths       = nan(nRows,1); % 每一行对应的 depth (um)

for k = 1:nRows
    row_idx     = display_order(k);           % 在 resultArray_type12 中的行号
    template_id = template_id_per_row(row_idx);
    spikes      = resultArray_type12(row_idx, 2:end);
    spikes      = spikes(spikes > 0 & spikes >= win(1) & spikes <= win(2));

    raster_unit_ids(k) = template_id;
    if ~isnan(template_id) && template_id <= numel(depth_col)
        raster_unit_depths(k) = depth_col(template_id);
    else
        raster_unit_depths(k) = NaN;
    end

    if ~isempty(spikes)
        raster_points_depthOrder = [raster_points_depthOrder; ...
            [spikes(:), k*ones(numel(spikes),1)]]; %#ok<AGROW>
    end
end

fprintf('\n✅ 已从 Figure1 的 STR/CTX raster 中抽取 %d 个点（%d 行）。\n', ...
    size(raster_points_depthOrder,1), nRows);

% ------- 11.2 保存路径设置 -------
save_dir = 'C:\Users\lab-admin\OneDrive - purdue.edu\1.final_paper\final_figures\DataAnalysis\Fig1_ephys_WT_HOM\HOM_1019_PTZ';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
    fprintf('📁 已创建保存目录: %s\n', save_dir);
end

baseFile = fullfile(save_dir, 'rasterFigure1_depthOrder.mat');
if exist(baseFile, 'file')
    % 若已存在，自动加编号后缀
    kSave = 1;
    while true
        candidate = fullfile(save_dir, sprintf('rasterFigure1_depthOrder_%d.mat', kSave));
        if ~exist(candidate, 'file')
            outFile = candidate;
            break;
        end
        kSave = kSave + 1;
    end
else
    outFile = baseFile;
end

save(outFile, ...
    'raster_points_depthOrder', ...  % Nx2: [time, rowIndex]
    'raster_unit_ids', ...           % nRowsx1: 每行对应的 templateID
    'raster_unit_depths', ...        % nRowsx1: 每行对应的 depth
    'win', 'x_start', 'x_end', ...   % 时间窗口信息
    'display_order', 'template_id_per_row', ...
    'depth_col', 'unitType', ...
    '-v7.3');

fprintf('💾 已保存 raster 相关变量到文件:\n  %s\n', outFile);

% ------- 11.3 按 unitID 排列重画一个 raster -------
[~, sortIdxByID] = sort(raster_unit_ids, 'ascend', 'MissingPlacement','last');

newRowForOld = nan(nRows,1);
for newRow = 1:nRows
    oldRow = sortIdxByID(newRow);
    newRowForOld(oldRow) = newRow;
end

raster_points_byUnitID = raster_points_depthOrder;
for i = 1:size(raster_points_byUnitID,1)
    oldRow = raster_points_byUnitID(i,2);
    raster_points_byUnitID(i,2) = newRowForOld(oldRow);
end

figure('Name','Raster — type1/2 sorted by unitID',...
       'Position',[200 200 800 600]);
hold on;

for oldRow = 1:nRows
    uid = raster_unit_ids(oldRow);
    if isnan(uid), continue; end
    mask_row   = (raster_points_depthOrder(:,2) == oldRow);
    spikes_row = raster_points_depthOrder(mask_row, 1);
    if isempty(spikes_row), continue; end

    newRow = newRowForOld(oldRow);
    if uid <= numel(unitType)
        if unitType(uid) == 1
            c = 'r';
        elseif unitType(uid) == 2
            c = 'b';
        else
            c = [0 0 0];
        end
    else
        c = [0 0 0];
    end

    scatter(spikes_row, newRow*ones(size(spikes_row)), 8, c, '.');
end

xlim(win);
ylim([0.5, nRows+0.5]);
xlabel('Time (s)');
ylabel('Neuron (sorted by unitID)');
title('Raster (unitType 1=red, 2=blue) — sorted by unit ID');
grid on;
hold off;

fprintf('✅ 已生成按 unitID 排列的 raster figure。\n');
