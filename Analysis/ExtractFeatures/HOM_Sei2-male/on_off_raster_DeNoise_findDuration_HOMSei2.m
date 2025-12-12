%% =========================================================
% === Load spike data, cluster info, and unitType ===
% ==================   =======================================
base_path = 'H:\seizure\2024-01-02_WT_HOM-male-adult\2024-01-02_13-43-59\Record Node 101\experiment2\recording2\continuous\Neuropix-PXI-100.ProbeA-AP';

% Define file paths 
spike_times_path      = fullfile(base_path, 'spike_times.npy');
spike_templates_path  = fullfile(base_path, 'spike_templates.npy');
excel_path            = fullfile(base_path, 'clustertwo.xlsx');
unitType_path         = fullfile(base_path, 'unitType.mat');

% =========================================================
% 1) Load spike times & templates (.npy)
% =========================================================
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
    spike_templates_full_wt = double(spike_templates_wt) + 1;  % make 1-based for MATLAB
    fprintf('  -> %d template IDs loaded.\n', numel(spike_templates_full_wt));
catch ME
    error('Failed to load spike_templates.npy: %s', ME.message);
end

% Combine into one 2D array (template ID + spike time)

spike_channel_wt = [spike_time_full_wt, spike_templates_full_wt];

% Sort by template ID
[~, sortIdx_wt] = sort(spike_channel_wt(:, 2));
sortedArray2D_wt = spike_channel_wt(sortIdx_wt, :);

% Group by template ID
uniqueValues_wt = unique(sortedArray2D_wt(:, 2));
groupedRows_wt = accumarray(sortedArray2D_wt(:, 2), sortedArray2D_wt(:, 1), [], @(x) {x'});

% Convert to resultArray_wt (neuron × spike times)
maxGroupSize_wt = max(cellfun(@length, groupedRows_wt));
resultArray_wt = nan(length(uniqueValues_wt), maxGroupSize_wt + 1);
for i = 1:length(uniqueValues_wt)
    resultArray_wt(i, 1) = uniqueValues_wt(i);
    resultArray_wt(i, 2:length(groupedRows_wt{i}) + 1) = groupedRows_wt{i};
end
disp('✅ Spike data (times & templates) loaded and grouped successfully.');

% =========================================================
% 2) Load cluster info (clustertwo.xlsx)
% =========================================================
try
    fprintf('\nLoading cluster info from: %s\n', excel_path);
    T = readtable(excel_path);  % preferred if has headers

    % Optional: get numeric column2 if needed (region or depth info)
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

% =========================================================
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
    fprintf('unitType length = %d\n', numel(unitType));
catch ME
    error('Failed to load unitType.mat: %s', ME.message);
end

% =========================================================
% 4) Sanity checks
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




%%
% 假设：
% unitType: 601x1 double
% T: 含 depth 的 table 或矩阵，depth 在第4列

% === 1. 基本计数 ===
num_type1 = sum(unitType == 1);
num_type2 = sum(unitType == 2);
fprintf('Number of neurons with unitType == 1: %d\n', num_type1);
fprintf('Number of neurons with unitType == 2: %d\n', num_type2);

% === 2. 取出第4列，准备转成纯 double 向量 ===
if istable(T)
    raw_depth = T{:,4};      % 可能是 double，也可能是 cell
else
    raw_depth = T(:,4);
end

% 最终我们要得到：depth_col = N×1 double
N = numel(raw_depth);
depth_col = nan(N,1);        % 先全部设成 NaN

if iscell(raw_depth)
    for i = 1:N
        val = raw_depth{i};

        if isempty(val)
            depth_col(i) = NaN;           % 空的就 NaN

        elseif isnumeric(val)
            % 如果是数值，看是不是标量
            if isscalar(val)
                depth_col(i) = double(val);
            else
                % 如果是数组，取第一个
                depth_col(i) = double(val(1));
            end

        elseif ischar(val) || isstring(val)
            % 字符串就转成数
            tmp = str2double(val);
            if ~isnan(tmp)
                depth_col(i) = tmp;
            else
                depth_col(i) = NaN;
            end

        else
            % 其他奇怪类型，直接 NaN
            depth_col(i) = NaN;
        end
    end
else
    % 本来就是 double/matrix
    depth_col = double(raw_depth);
end

% === 3. 按 unitType 抓对应的 depth ===
idx_type1 = (unitType == 1);
idx_type2 = (unitType == 2);

depths_type1 = depth_col(idx_type1);
depths_type2 = depth_col(idx_type2);

% 去掉 NaN
depths_type1_clean = depths_type1(~isnan(depths_type1));
depths_type2_clean = depths_type2(~isnan(depths_type2));

% === 4. 区域划分（阈值 2625）===
isCortex_type1   = depths_type1_clean > 2625;
isStriatum_type1 = depths_type1_clean <= 2625;

isCortex_type2   = depths_type2_clean > 2625;
isStriatum_type2 = depths_type2_clean <= 2625;

% === 5. 输出 ===
fprintf('\nType 1 neurons (after removing NaN): %d\n', numel(depths_type1_clean));
fprintf('  Cortex  (>2625): %d\n', sum(isCortex_type1));
fprintf('  Striatum(<=2625): %d\n', sum(isStriatum_type1));

fprintf('\nType 2 neurons (after removing NaN): %d\n', numel(depths_type2_clean));
fprintf('  Cortex  (>2625): %d\n', sum(isCortex_type2));
fprintf('  Striatum(<=2625): %d\n', sum(isStriatum_type2));

%% cusimized for PTZ IID recording : detection on mean(d) instead of frequncy specific swd 

loading_lfp
%
% d_shift = [d(:, 11747:end), zeros(size(d,1), 11747)];
% %
% d = d_shift;
corrected_baseline = baseline_correction(mean(d),2500);
plot_swd_spectrogram_0_60(corrected_baseline,2500);
A = plot_filtered_lfp(mean(d), 2500);
detect_swd(corrected_baseline);
detect_swd(A); 
detect_swd(mean(d));

% Given variables
swd_events_2500Hz = swd_events; % Original SWD events at 2500 Hz
original_sampling_rate = 2500; % Original sampling rate in Hz
new_sampling_rate = 30000; % New sampling rate in Hz
scaling_factor = new_sampling_rate / original_sampling_rate;




%%
raw_lfp = corrected_baseline;

fs = 2500;
t = (0:length(raw_lfp)-1)/fs;

% % --- 方法1: IIR Notch ---
% f0 = 60; Q = 30;
% wo = f0/(fs/2); bw = wo/Q;
% [b,a] = iirnotch(wo,bw);
% lfp_clean1 = filtfilt(b,a,raw_lfp);

% --- 方法3: 多频点 Notch ---
f0s = [58 62, 118 122, 178 182 ];
lfp_clean3 = raw_lfp;
for f0 = f0s
    wo = f0/(fs/2);
    bw = wo/30;
    [b,a] = iirnotch(wo,bw);
    lfp_clean1 = filtfilt(b,a,lfp_clean3);
end

% --- 方法2: Butterworth bandstop ---
[b,a] = butter(4,[58 62]/(fs/2),'stop');
lfp_clean2 = filtfilt(b,a,raw_lfp);

% --- 方法3: 多频点 Notch ---
f0s = [60 120 180];
lfp_clean3 = raw_lfp;
for f0 = f0s
    wo = f0/(fs/2);
    bw = wo/30;
    [b,a] = iirnotch(wo,bw);
    lfp_clean3 = filtfilt(b,a,lfp_clean3);
end

% --- 绘图 ---
figure;
plot(t, raw_lfp, 'Color', [0.7 0.7 0.7], 'DisplayName', 'Raw LFP'); hold on;
plot(t, lfp_clean1, 'r',   'DisplayName', 'IIR Notch (60Hz)');
plot(t, lfp_clean2, 'b',   'DisplayName', 'Butterworth 58-62Hz Stop');
plot(t, lfp_clean3, 'g',   'DisplayName', 'Multi-notch 60/120/180Hz');
xlabel('Time (s)');
ylabel('Amplitude');
title('Comparison of 60Hz Noise Removal Methods');
legend('show');
grid on;

%%
detect_swd(lfp_clean3)
%%
swd_events = swd_events;
swd_events_2500Hz = swd_events; % Original SWD events at 2500 Hz
original_sampling_rate = 2500; % Original sampling rate in Hz
new_sampling_rate = 30000; % New sampling rate in Hz
scaling_factor = new_sampling_rate / original_sampling_rate;
%%
% Convert time points
swd_events_30000Hz = cell(size(swd_events));% Replace this with your actual variable

% Pop_ulate each component of the cell array
for i = 1:length(swd_events)
    swd_events_30000Hz{i} = swd_events{i} * scaling_factor; % Scale time points
end

% Display the converted SWD events
disp('SWD events converted to 30000Hz:');
%disp(swd_events_30000Hz);


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
%disp(swd_events_seconds);




%%
% Initialize the logical array
%logic_array = false(1, 90793140); % Logical array with all values initialized to 0 (false)
size_ap = 12*size(corrected_baseline,2);
logic_array = false(1,size_ap);


% Populate the logical array based on swd_events_30000Hz
for i = 1:length(swd_events_30000Hz)
    % Get the current component
    current_event = swd_events_30000Hz{i};
    
    % Find the min and max of the current component
    if ~isempty(current_event)
        min_index = min(current_event);
        max_index = max(current_event);
        
        % Ensure min_index and max_index are within bounds of the logical array
        if min_index >= 1 && max_index <= length(logic_array)
            % Set values between min and max (inclusive) to 1 (true)
            logic_array(min_index:max_index) = true;
        end
    end
end

% Display a confirmation message
disp('Logical array populated successfully.');




%%
figure()
hold on 
plot(logic_array)
ylim([0,2])
hold off



%% load raster array 


timepoint_array = resultArray_wt(:,2:end);
neuron_id = resultArray_wt(:,1);


%% GUI
mkdir('SWD_curation_autosave');

[swd_events_filtered, keep_idx] = curate_swd_events(lfp_clean3, 2500, swd_events, 10, timepoint_array, neuron_id, struct('max_neurons',600));


%%
 %save('swd_events_FixMissing.mat','swd_events_filtered','keep_idx');   % 如变量较大可加 '-v7.3'

 %% GUI for #2 
%% === Prepare labels (NaN/0/1/2) for pick_unsure_onsets ===
% % 需要：lfp_clean3, fs, swd_events 已经在工作区
% % 目标：得到变量 labels 与 swd_events 一一对应
% 
% % 1) 直接用工作区变量
% if exist('swd_event_labels','var')
%     labels = swd_event_labels;
% else
%     % 2) 从 autosave 目录读取最新会话
%     autosave_dir = 'SWD_curation_autosave';
%     labels = [];
%     try
%         d = dir(fullfile(autosave_dir,'SWD_curation_*.mat'));
%         if ~isempty(d)
%             [~,ix] = max([d.datenum]);       % 最新文件
%             Sauto  = load(fullfile(autosave_dir, d(ix).name));
%             if isfield(Sauto,'labels')
%                 labels = Sauto.labels;        % NaN/0/1/2
%             elseif isfield(Sauto,'keep')
%                 % 没有三态则退化到两态（无法区分“不确定”）
%                 labels = double(Sauto.keep);  % 1=keep, 0=del
%             end
%         end
%     catch ME
%         warning('读取 autosave 失败：%s', ME.message);
%     end
% end
% 
% % 3) 若还没拿到 labels，则再退化一次
% if isempty(labels)
%     if exist('keep_idx','var')
%         labels = double(keep_idx);    % 1/0
%     else
%         error(['找不到 labels：请先运行三态 GUI（curate_swd_events）做标注，' ...
%                '或确保工作区有 swd_event_labels / keep_idx，或 autosave 目录存在会话文件。']);
%     end
% end
% 
% % 4) 基本一致性检查与修剪/填充
% M = numel(swd_events);
% if numel(labels) ~= M
%     warning('labels 长度(%d) 与 swd_events 数量(%d) 不一致，自动调整到匹配大小。', numel(labels), M);
%     labels = labels(:);
%     if numel(labels) > M
%         labels = labels(1:M);
%     else
%         labels(end+1:M,1) = NaN;  % 不足部分设为未标注
%     end
% end
% 
% % 5) 调用“不确定事件原点拾取”GUI
% %    注意：只有 label==2 的事件会显示；如果你的 labels 没有 2，则不会显示任何事件
% [unsure_ids, onset_samples, onset_times] = pick_unsure_onsets( ...
%     lfp_clean3, fs, swd_events, labels, ...
%     'RowsPerPage', 10, 'PreSec', 1, 'PostSec', 2, ...
%     'AutosaveDir', 'SWD_onset_autosave');
% 
% % 结果会自动保存到 SWD_onset_autosave/ 下，并抛到 base：
% %   unsure_ids, unsure_onset_samples, unsure_onset_times

%%
% 假设已有：
% lfp_clean3  = 你的 LFP 列向量
% fs          = 2500;
% swd_events  = 1xM cell 的事件样本索引
% swd_event_labels = 长度 M 的 NaN/0/1/2 标签（来自第一 GUI）
% timepoint_array  = (Nneurons x K) 秒级 spike 表
% neuron_id        = Nneurons x 1

[unsure_ids, onset_samples, onset_times] = pick_unsure_onsets_single( ...
    lfp_clean3, fs, swd_events, swd_event_labels, timepoint_array, neuron_id, 10, ...
    struct('max_neurons',600,'bin_w',0.05,'ylim_lfp',[],'cwt_freq_lim',[0.5 60]));





%% ==== Merge onset_times (from pick_unsure_onsets_single) into t_start_s (from curate_swd_events) ====
% 依赖的工作区变量（两个GUI脚本已自动赋值到 base）：
%   swd_event_quality     — Mx4 [event_id, t_start_s, t_end_s, label]     (from curate_swd_events)
%   swd_keep_idx          — Mx1 logical                                   (from curate_swd_events)
%   swd_session_file_csv  — char, 原CSV路径                               (from curate_swd_events)
%   unsure_ids            — Kx1 original event indices (label==2)         (from pick_unsure_onsets_single)
%   unsure_onset_times    — Kx1 onset times (sec), aligned with unsure_ids (from pick_unsure_onsets_single)
%   merged_labels         — (可选) Mx1 最新标签（含relabel后）              (from pick_unsure_onsets_single)
%   merged_keep_idx       — (可选) Mx1 最新keep布尔值                       (from pick_unsure_onsets_single)

% ---- 1) 取出必须数据，并做基本校验
must = {'swd_event_quality','swd_session_file_csv'};
for k = 1:numel(must)
    if ~evalin('base', sprintf('exist(''%s'',''var'')', must{k}))
        error('Missing required base variable: %s. Please run curate_swd_events first.', must{k});
    end
end
EQ   = evalin('base','swd_event_quality');         % [event_id, t_start_s, t_end_s, label]
csv0 = evalin('base','swd_session_file_csv');      % 原CSV路径

% keep/labels 均尽力从“最新”来源拿；没有则回退到旧的
if evalin('base','exist(''merged_labels'',''var'')')
    labels = evalin('base','merged_labels');
else
    labels = EQ(:,4);  % 原 label 列
end
if evalin('base','exist(''merged_keep_idx'',''var'')')
    keep_idx = evalin('base','merged_keep_idx');
elseif evalin('base','exist(''swd_keep_idx'',''var'')')
    keep_idx = evalin('base','swd_keep_idx');
else
    keep_idx = labels ~= 0;  % 兜底：0=delete，其余保留
end

% onset 覆盖源（若不存在，则保持原值）
hasUnsure = evalin('base','exist(''unsure_ids'',''var'') && exist(''unsure_onset_times'',''var'')');
if hasUnsure
    unsure_ids         = evalin('base','unsure_ids');          % 原事件ID（1..M）
    unsure_onset_times = evalin('base','unsure_onset_times');  % 秒
else
    unsure_ids = [];
    unsure_onset_times = [];
end

% ---- 2) 构造按 event_id 对齐的 onset 替换向量
M = size(EQ,1);
event_id  = EQ(:,1);
t_start_s = EQ(:,2);
t_end_s   = EQ(:,3);

t_start_new = t_start_s;  % 默认保留旧值

if ~isempty(unsure_ids)
    onset_by_event = nan(M,1);
    % 注意：unsure_ids 是原事件索引（1..M），和 unsure_onset_times 成对齐顺序
    valid_mask = ~isnan(unsure_onset_times);
    if any(valid_mask)
        onset_by_event(unsure_ids(valid_mask)) = unsure_onset_times(valid_mask);
        repl_mask = ~isnan(onset_by_event);
        % 覆盖 t_start_s
        t_start_new(repl_mask) = onset_by_event(repl_mask);
        % 保护性约束：不让 start 超过 end
        bad = t_start_new > t_end_s;
        if any(bad)
            warning('Some new onsets exceed t_end_s. Clamping to slightly before t_end_s.');
            t_start_new(bad) = t_end_s(bad) - eps;  % 轻微回退，避免相等/越界
        end
    end
end

% ---- 3) 组装更新后的表
T_merged = table( ...
    event_id, ...
    t_start_new, ...
    t_end_s, ...
    labels(:), ...
    logical(keep_idx(:)), ...
    'VariableNames', {'event_id','t_start_s','t_end_s','label','keep'});

% ---- 4) 导出新的 CSV 到同目录，文件名加后缀 _onsetMerged
[fp,fn,~] = fileparts(csv0);
csv1 = fullfile(fp, [fn '_onsetMerged.csv']);
writetable(T_merged, csv1);

% ---- 5) 同步回 base 工作区，便于下游继续用
assignin('base','swd_event_quality', [T_merged.event_id, T_merged.t_start_s, T_merged.t_end_s, T_merged.label]); % 仍保持 Mx4 结构
assignin('base','swd_keep_idx', T_merged.keep);
assignin('base','swd_session_file_csv_merged', csv1);
fprintf('Merged onset times applied. New CSV written to:\n  %s\n', csv1);



% ---- 6) 另存为 .MAT 版本（与 CSV 同名后缀）----
mat1 = fullfile(fp, [fn '_onsetMerged.mat']);

% 同时保存表格与分列变量，便于不同风格的加载
event_id_mat  = T_merged.event_id;
t_start_s_mat = T_merged.t_start_s;
t_end_s_mat   = T_merged.t_end_s;
label_mat     = T_merged.label;
keep_mat      = T_merged.keep;

% 1) 以结构体字段形式保存（兼容性最好）
S_merged = struct();
S_merged.event_id   = event_id_mat;
S_merged.t_start_s  = t_start_s_mat;
S_merged.t_end_s    = t_end_s_mat;
S_merged.label      = label_mat;
S_merged.keep       = keep_mat;
S_merged.source_csv = csv1;

save(mat1, '-struct', 'S_merged');

% 2) 也把 table 原样保存（方便直接 read）
try
    save(mat1, 'T_merged', '-append');   % MATLAB R2018a+ 支持 table 保存
catch ME
    warning('Append table T_merged failed (%s). Struct was saved anyway.', ME.message);
end

% 回写路径到 base
assignin('base','swd_session_file_mat_merged', mat1);
fprintf('Also wrote MAT copy to:\n  %s\n', mat1);


%% ==== Use onsetMerged start times to update the FIRST sample of label==2 events in swd_events_filtered ====
% ===== Build swd_events_QualityM from T_merged (keep==1) and save =====
% Requirements (任一存在即可)：
%   1) 变量 T_merged（table，至少含：event_id, t_start_s, t_end_s, keep）
%   或
%   2) 路径 swd_session_file_mat_merged（指向 *_onsetMerged.mat，内含 T_merged）
%
%   还需要采样率变量：
%     优先使用 fs；若无则使用 original_sampling_rate；都没有则报错
%
% 可选（用于边界夹取）：
%   lfp_data（列向量）或 N_samples（总样本数）
%
% 输出：
%   swd_events_QualityM — 1xN cell（仅保留 keep==1 的事件；每个为连续样本索引）
%   同时保存在 SWD_curation_autosave/swd_events_QualityM_yyyymmdd_HHMMSS.mat

% ------- 采样率 -------
if exist('fs','var') && ~isempty(fs)
    fs_use = fs;
elseif exist('original_sampling_rate','var') && ~isempty(original_sampling_rate)
    fs_use = original_sampling_rate;
else
    error('Need sampling rate: define fs or original_sampling_rate first.');
end

% ------- 获取 T_merged -------
if ~exist('T_merged','var') || isempty(T_merged)
    if exist('swd_session_file_mat_merged','var') && ~isempty(swd_session_file_mat_merged)
        S_on = load(swd_session_file_mat_merged);
        if isfield(S_on,'T_merged')
            T_merged = S_on.T_merged;
        else
            error('File "%s" does not contain T_merged.', swd_session_file_mat_merged);
        end
    else
        error('T_merged not found. Provide T_merged or swd_session_file_mat_merged.');
    end
end

% ------- 基本字段检查 -------
needVars = {'t_start_s','t_end_s','keep'};
for ii = 1:numel(needVars)
    if ~ismember(needVars{ii}, T_merged.Properties.VariableNames)
        error('T_merged missing required column: %s', needVars{ii});
    end
end

% ------- 可选：确定样本总数用于边界夹取 -------
if exist('lfp_data','var') && ~isempty(lfp_data)
    N_samples = numel(lfp_data);
elseif ~exist('N_samples','var')
    N_samples = [];  % 未知则不做上界夹取
end

% ------- 只取 keep==1 的事件 -------
mask_keep = logical(T_merged.keep);
t0 = T_merged.t_start_s(mask_keep);
t1 = T_merged.t_end_s(mask_keep);

% 去掉 NaN / 非法窗口（终点必须 >= 起点）
valid = ~isnan(t0) & ~isnan(t1) & (t1 >= t0);
t0 = t0(valid);
t1 = t1(valid);

% ------- 秒 -> 样本索引（1-based），并夹取到有效范围 -------
start_idx = round(t0 * fs_use) + 1;
end_idx   = round(t1 * fs_use) + 1;

start_idx(start_idx < 1) = 1;
if ~isempty(N_samples)
    end_idx(end_idx   > N_samples) = N_samples;
end

% 确保 start<=end（数值精度问题导致相等亦允许，得到单点事件）
bad = start_idx > end_idx;
start_idx(bad) = [];
end_idx(bad)   = [];

% ------- 构造 1xN cell（与 swd_events 结构一致）-------
N = numel(start_idx);
swd_events_QualityM = cell(1, N);
for i = 1:N
    swd_events_QualityM{i} = start_idx(i):end_idx(i);  % 连续样本索引
end

% ------- 推送到 base 工作区 -------
assignin('base','swd_events_QualityM', swd_events_QualityM);

% ------- 保存到 SWD_curation_autosave，防覆盖 -------
out_dir = 'SWD_curation_autosave';
if ~exist(out_dir,'dir'), mkdir(out_dir); end
timestamp = datestr(now,'yyyymmdd_HHMMSS');
out_path  = fullfile(out_dir, ['swd_events_QualityM_' timestamp '.mat']);
save(out_path, 'swd_events_QualityM');

fprintf('[OK] swd_events_QualityM: %d events saved to %s\n', N, out_path);




%% ==== Build swd_events_QualityM from T_merged (keep==1 only) ====
% Needs in base (at least one path to T_merged):
%   Option A: variable T_merged in workspace
%   Option B: swd_session_file_mat_merged -> MAT file containing T_merged
% Also needs sampling rate: fs (or original_sampling_rate)

% ---- fetch fs ----
if evalin('base','exist(''fs'',''var'')')
    fs = evalin('base','fs');
elseif evalin('base','exist(''original_sampling_rate'',''var'')')
    fs = evalin('base','original_sampling_rate');
else
    error('Need "fs" or "original_sampling_rate" in base.');
end

% ---- fetch T_merged ----
have_T = evalin('base','exist(''T_merged'',''var'')');
if ~have_T
    if evalin('base','exist(''swd_session_file_mat_merged'',''var'')')
        matMerged = evalin('base','swd_session_file_mat_merged');
    elseif evalin('base','exist(''matMerged'',''var'')')
        matMerged = evalin('base','matMerged');
    else
        error('Need T_merged or swd_session_file_mat_merged/matMerged.');
    end
    Sload = load(matMerged);
    if isfield(Sload,'T_merged')
        T_merged = Sload.T_merged;
    else
        % 兼容旧字段
        T_merged = table(Sload.event_id(:), Sload.t_start_s(:), Sload.t_end_s(:), logical(Sload.keep(:)), ...
            'VariableNames', {'event_id','t_start_s','t_end_s','keep'});
    end
else
    T_merged = evalin('base','T_merged');
end

% ---- sanity check columns ----
reqVars = {'event_id','t_start_s','t_end_s','keep'};
if ~all(ismember(reqVars, T_merged.Properties.VariableNames))
    error('T_merged must include columns: %s', strjoin(reqVars, ', '));
end

% ---- estimate N (for clamping to [1,N]) ----
N = [];
if evalin('base','exist(''lfp_data'',''var'')')
    N = numel(evalin('base','lfp_data'));
elseif evalin('base','exist(''d'',''var'')')
    dsz = evalin('base','size(d)');
    N = dsz(2);
end
doClamp = ~isempty(N);

% ---- keep==1 only ----
Tkeep = T_merged(logical(T_merged.keep), :);
K = height(Tkeep);

swd_events_QualityM = cell(K,1);
event_id_QualityM   = Tkeep.event_id(:);

n_clamped = 0;
for i = 1:K
    t0 = Tkeep.t_start_s(i);
    t1 = Tkeep.t_end_s(i);

    % 转样本索引（与现有代码一致：start=round(t*fs)+1）
    s0 = round(t0*fs) + 1;
    e1 = round(t1*fs) + 1;

    % 确保 s0 <= e1
    if e1 < s0
        tmp = s0; s0 = e1; e1 = tmp;
    end

    % 边界夹取
    if doClamp
        s0c = max(1, min(N, s0));
        e1c = max(1, min(N, e1));
        if s0c ~= s0 || e1c ~= e1
            n_clamped = n_clamped + 1;
        end
        s0 = s0c; e1 = e1c;
    end

    % 保证连续（包含端点）
    swd_events_QualityM{i} = s0:e1;
end

% ---- push to base ----
assignin('base','swd_events_QualityM', swd_events_QualityM);
assignin('base','event_id_QualityM',   event_id_QualityM);

fprintf('Built swd_events_QualityM: %d kept events from T_merged (keep==1).\\n', K);
if n_clamped>0 && doClamp
    fprintf('  Note: %d event windows were clamped to [1, N=%d].\\n', n_clamped, N);
end

%%

swd_events = swd_events_filtered; 

swd_events_2500Hz = swd_events;

swd_events_30000Hz = cell(size(swd_events));% Replace this with your actual variable

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




%% repopulte logic array 
%logic_array = false(1, 90793140); % Logical array with all values initialized to 0 (false)
size_ap = 12*size(corrected_baseline,2);
logic_array = false(1,size_ap);


% Populate the logical array based on swd_events_30000Hz
for i = 1:length(swd_events_30000Hz)
    % Get the current component
    current_event = swd_events_30000Hz{i};
    
    % Find the min and max of the current component
    if ~isempty(current_event)
        min_index = min(current_event);
        max_index = max(current_event);
        
        % Ensure min_index and max_index are within bounds of the logical array
        if min_index >= 1 && max_index <= length(logic_array)
            % Set values between min and max (inclusive) to 1 (true)
            logic_array(min_index:max_index) = true;
        end
    end
end

% Display a confirmation message
disp('Logical array populated successfully.');

figure()
hold on 
plot(logic_array)
ylim([0,2])
hold off





%%

% Parameters
sampling_rate = 30000;              % Sampling rate in Hz (AP)
min_burst_duration = 0.01;          % Minimum burst duration in seconds
min_burst_samples = min_burst_duration * sampling_rate;
bin_size = 0.1;                     % Bin size in seconds (0.1 s)
bin_samples = bin_size * sampling_rate;

% ---- Identify burst events ----
burst_diff = diff([0, logic_array, 0]); % Add 0 at boundaries to detect edges
burst_starts = find(burst_diff == 1);
burst_ends   = find(burst_diff == -1) - 1;
burst_durations = burst_ends - burst_starts + 1;

% ---- Filter bursts lasting at least min_burst_duration ----
valid_bursts  = find(burst_durations >= min_burst_samples);
burst_starts  = burst_starts(valid_bursts);
burst_ends    = burst_ends(valid_bursts);

% ---- Preallocate arrays for storing spike rates ----
num_neurons = size(resultArray_wt, 1);
burst_spike_rate     = zeros(num_neurons, 1);
non_burst_spike_rate = zeros(num_neurons, 1);

% ---- Total burst / non-burst time ----
total_burst_time = sum((burst_ends - burst_starts + 1)) / sampling_rate;
total_non_burst_time = length(logic_array) / sampling_rate - total_burst_time;

% ---- Analyze spikes during burst and non-burst periods ----
for neuron = 1:num_neurons
    spike_times = resultArray_wt(neuron, :);
    spike_times = spike_times(spike_times > 0);                  % Remove zero padding if any
    spike_samples = round(spike_times * sampling_rate);          % Convert to sample indices

    % Count spikes during burst events
    burst_spike_count = 0;
    for b = 1:length(burst_starts)
        burst_range = burst_starts(b):burst_ends(b);
        burst_spike_count = burst_spike_count + sum(ismember(spike_samples, burst_range));
    end

    % Count spikes during non-burst events
    all_burst_ranges = cell2mat(arrayfun(@(s, e) s:e, burst_starts, burst_ends, 'UniformOutput', false));
    non_burst_spike_count = sum(~ismember(spike_samples, all_burst_ranges));

    % spike rates
    burst_spike_rate(neuron)     = burst_spike_count     / max(total_burst_time, 1e-6);
    non_burst_spike_rate(neuron) = non_burst_spike_count / max(total_non_burst_time, 1e-6);
end

% ---- Compute burst-to-non-burst spike rate ratio ----
burst_to_non_burst_rate_ratio = burst_spike_rate ./ (non_burst_spike_rate + 1e-6);

% ---- Select neurons with higher burst spike rate ----
selected_neurons = find(burst_to_non_burst_rate_ratio > 0.5);
% (<0.5 meaning off, >2 meaning on; 这里你可以自己调)

% =========================================================
% Visualization layout
% =========================================================
x_start = 0;                              % Starting time in seconds
x_end   = max(resultArray_wt(:));         % Ending time based on the spike data

figure;
subplot_positions = [0.1, 0.85, 0.8, 0.1;   % 1st subplot
                     0.1, 0.7,  0.8, 0.1;   % 2nd
                     0.1, 0.55, 0.8, 0.1;   % 3rd
                     0.1, 0.4,  0.8, 0.1;   % 4th
                     0.1, 0.25, 0.8, 0.13;  % 5th
                     0.1, 0.1,  0.8, 0.07]; % 6th
ax = zeros(6,1);

% =========================================================
% 1) CWT subplot
% =========================================================
ax(1) = subplot('Position', subplot_positions(1,:)); hold on

% sampling rate of LFP
fs = 2500;
if exist('original_sampling_rate','var') && ~isempty(original_sampling_rate)
    fs = original_sampling_rate;
end

% mean LFP (convert to µV)
scale_factor = 0.195;
x = mean(d(: , :), 1) * scale_factor;
x = x(:);

% bandpass + detrend
bp = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',1,'HalfPowerFrequency2',60, ...
    'SampleRate',fs);
x = filtfilt(bp, x);
x = detrend(x, 'linear');

% window
ix0 = max(1, floor(x_start*fs)+1);
ix1 = min(numel(x), ceil(x_end*fs));
sig_win = x(ix0:ix1);
t_win   = ((ix0:ix1)-1)/fs;

% cwt
fb = cwtfilterbank('SignalLength',numel(sig_win), 'SamplingFrequency',fs, ...
    'Wavelet','morse','VoicesPerOctave',24,'TimeBandwidth',60, ...
    'FrequencyLimits',[0.5 60]);
[cfs, f] = wt(fb, sig_win);

P    = abs(cfs).^2;
Zlog = log10(P + eps);
mu   = mean(Zlog, 2);
sd   = std(Zlog, 0, 2) + eps;
Zz   = (Zlog - mu) ./ sd;
clim = [-2, 4];

imagesc(t_win, f, Zz); axis xy
xlim([x_start, x_end]); ylim([0.5 60])
xlabel('Time (s)'); ylabel('Frequency (Hz)')
title('CWT (Morse, 0–60 Hz) — per-freq z-score of log(power)')
colormap(jet); caxis(clim)
cb = colorbar; ylabel(cb, 'z (per-freq log power)')
hold off

% =========================================================
% 2) Filtered LFP subplot
% =========================================================
ax(2)=subplot('Position', subplot_positions(2, :)); hold on
low_cutoff = 5;
high_cutoff = 60;
[b, a] = butter(4, [low_cutoff high_cutoff] / (2500 / 2), 'bandpass');
filtered_5_60_lfp = filtfilt(b, a, mean(d));
filtered_lfp_microV = filtered_5_60_lfp * scale_factor;
time_vector = (1:length(filtered_5_60_lfp)) / 2500;
plot(time_vector, filtered_lfp_microV, 'b');
xlabel('Time (s)');
ylabel('Amplitude (µV)');
title('Filtered LFP Signal (5-60 Hz) in µV');
ylim([-2500 2500]);
xlim([x_start, x_end])
hold off

% =========================================================
% 3) SWD detection subplot
% =========================================================
ax(3)=subplot('Position', subplot_positions(3, :));
time_in_seconds = (1:length(mean(d))) / fs;
plot(time_in_seconds, lfp_clean3, 'Color', [0.6 0.6 0.6]); hold on;
dmean = mean(d);
for i = 1:length(swd_events)
    event = swd_events{i};
    event_time = event / fs;
    plot(event_time, dmean(event), 'r', 'LineWidth', 2);
end
xlabel('Time (seconds)');
ylabel('Amplitude');
title('SWD Detection in LFP Signal');
xlim([x_start, x_end])
ylim([-20000 20000]);
hold off;

% =========================================================
% 4) Burst detection subplot
% =========================================================
ax(4)=subplot('Position', subplot_positions(4, :));
hold on;
for b = 1:length(burst_starts)
    start_time = burst_starts(b) / sampling_rate;
    end_time   = burst_ends(b)   / sampling_rate;
    fill([start_time, end_time, end_time, start_time], ...
         [0, 0, 1, 1], 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
end
title('Burst Detection (>= min dur)');
ylabel('Burst');
ylim([0, 1]);
xlim([x_start, x_end]); 
hold off;

% =========================================================
% 5) Raster plot — split by depth; color by unitType
% =========================================================

% ---- 5.1 安全取 depth 列 ----
if exist('T','var')
    if istable(T)
        raw_depth = T{:,4};
    else
        raw_depth = T(:,4);
    end
    Ndepth = numel(raw_depth);
    depth_col = nan(Ndepth,1);
    if iscell(raw_depth)
        for ii = 1:Ndepth
            v = raw_depth{ii};
            if isempty(v)
                depth_col(ii) = NaN;
            elseif isnumeric(v)
                if isscalar(v)
                    depth_col(ii) = double(v);
                else
                    depth_col(ii) = double(v(1));
                end
            elseif ischar(v) || isstring(v)
                tmp = str2double(v);
                if ~isnan(tmp)
                    depth_col(ii) = tmp;
                end
            end
        end
    else
        depth_col = double(raw_depth);
    end
else
    depth_col = nan(size(resultArray_wt,1),1);
end

% ---- 5.2 确保 unitType 跟 neuron 数一致 ----
if numel(unitType) > size(resultArray_wt,1)
    unitType = unitType(1:size(resultArray_wt,1));
elseif numel(unitType) < size(resultArray_wt,1)
    unitType(numel(unitType)+1:size(resultArray_wt,1),1) = NaN;
end

% ---- 5.3 先按你的 burst 结果 + unitType==1/2 过滤 ----
valid_by_unitType = find(unitType == 1 | unitType == 2);
selected_neurons  = intersect(selected_neurons, valid_by_unitType);

% ---- 5.4 再按 depth 分成 striatum / cortex ----
depth_sel = depth_col(selected_neurons);
unit_sel  = unitType(selected_neurons);

isStr = depth_sel <= 2625;     % striatum
isCtx = depth_sel >  2625;     % cortex

[~, idx_str_sort] = sort(depth_sel(isStr), 'ascend');
[~, idx_ctx_sort] = sort(depth_sel(isCtx), 'ascend');

str_neurons = selected_neurons(isStr);
str_neurons = str_neurons(idx_str_sort);

ctx_neurons = selected_neurons(isCtx);
ctx_neurons = ctx_neurons(idx_ctx_sort);

% 按顺序拼起来（上：str，下：ctx）
display_order     = [str_neurons; ctx_neurons];
display_unitType  = unitType(display_order);
% display_depth    = depth_col(display_order); % 如要标注可以用

% ---- 5.5 画 raster ----
ax(5)=subplot('Position', subplot_positions(5, :));
hold on;
for yy = 1:numel(display_order)
    neuron = display_order(yy);
    spike_times = resultArray_wt(neuron, :);
    spike_times = spike_times(spike_times > 0);

    if display_unitType(yy) == 1
        thisColor = 'r';
    elseif display_unitType(yy) == 2
        thisColor = 'b';
    else
        thisColor = [0.3 0.3 0.3];
    end

    scatter(spike_times, yy*ones(size(spike_times)), 10, thisColor, '.');
end

% 画 str / ctx 分界线
n_str = numel(str_neurons);
if n_str > 0 && n_str < numel(display_order)
    plot([x_start x_end], [n_str+0.5 n_str+0.5], 'k--', 'LineWidth', 1);
end

xlabel('Time (s)');
ylabel('Neuron (sorted by depth)');
title('Raster: Striatum (top) vs Cortex (bottom)');
xlim([x_start, x_end]);
ylim([0, numel(display_order)+1]);

% 注意：这里不要放 legend，放在最后统一放
hold off;

% =========================================================
% 6) Binned spike rates — same selection as raster
% =========================================================
ax(6)=subplot('Position', subplot_positions(6, :));
bin_width = 0.1;
time_bins = x_start:bin_width:x_end;

selected_spike_times = [];
for i = 1:numel(display_order)
    neuron = display_order(i);
    spike_times = resultArray_wt(neuron, :);
    spike_times = spike_times(spike_times > 0);
    selected_spike_times = [selected_spike_times, spike_times];
end

binned_spikes = histcounts(selected_spike_times, time_bins) / bin_width;
bar(time_bins(1:end-1), binned_spikes, 'k');
xlabel('Time (s)');
ylabel('Spike Rate (spikes/s)');
title('Binned Spike Rates (unitType 1 & 2)');
xlim([x_start, x_end]);

% =========================================================
% Link axes
% =========================================================
linkaxes(ax, 'x');
set(gcf, 'Position', [100, 100, 800, 900]);

% =========================================================
% Global legend (so it won't squeeze subplot(5))
% =========================================================
% 我们用一个“假”对象来做 legend
axes(ax(5)); hold on;
h1 = plot(nan, nan, 'r.', 'MarkerSize', 12);
h2 = plot(nan, nan, 'b.', 'MarkerSize', 12);
h3 = plot(nan, nan, 'k--', 'LineWidth', 1);
hold off;

hL = legend([h1 h2 h3], {'unitType 1','unitType 2','Str/Ctx boundary'}, ...
    'Orientation','horizontal', 'NumColumns', 3);
% 手动调一下位置（根据你窗口大小可以再微调）
set(hL, 'Position',[0.25 0.015 0.5 0.03], 'Box','off');
%%
%  Drop-in: SWD LFP + Raster + Binned Spike Rate (3-panel)
%  依赖变量：
%    lfp_clean3          — 2500 Hz LFP（已去 60/120/180 Hz）
%    swd_events          — 1xN cell, 2500 Hz 样本索引 (通常是 swd_events_QualityM)
%    resultArray_wt      — [nNeurons x (1+K)]，第1列为 neuron/template ID，其余为 spike time (s)
%    unitType            — 每个 template 的类型 (0/1/2)
%    T                   — clustertwo.xlsx 对应的 table/matrix，第 4 列为 depth (µm)
%    x_start, x_end      — 显示时间范围 (秒)，若不存在则自动用 LFP 全长
% =========================================================

% ---------- 1) 基本参数 & x 轴范围 ----------
fsLFP = 2500;
if exist('original_sampling_rate','var') && ~isempty(original_sampling_rate)
    fsLFP = original_sampling_rate;
end

% LFP 信号
if exist('lfp_clean3','var') && ~isempty(lfp_clean3)
    lfp_for_plot = lfp_clean3(:);
else
    % 兜底：用 mean(d)
    lfp_for_plot = mean(d,1);
    lfp_for_plot = lfp_for_plot(:);
end

t_lfp = (0:numel(lfp_for_plot)-1)/fsLFP;

% x 轴范围（若前面已有 x_start/x_end 就用它们，否则用 LFP 全长）
if ~exist('x_start','var') || isempty(x_start)
    x_start = 0;
end
if ~exist('x_end','var') || isempty(x_end)
    x_end = t_lfp(end);
end

% ---------- 2) 准备 depth & display_order（与前面逻辑一致） ----------
% depth_col
if exist('T','var')
    if istable(T)
        raw_depth2 = T{:,4};
    else
        raw_depth2 = T(:,4);
    end
    Ndepth2 = numel(raw_depth2);
    depth_col2 = nan(Ndepth2,1);
    if iscell(raw_depth2)
        for ii = 1:Ndepth2
            v = raw_depth2{ii};
            if isempty(v)
                depth_col2(ii) = NaN;
            elseif isnumeric(v)
                if isscalar(v)
                    depth_col2(ii) = double(v);
                else
                    depth_col2(ii) = double(v(1));
                end
            elseif ischar(v) || isstring(v)
                tmp = str2double(v);
                if ~isnan(tmp)
                    depth_col2(ii) = tmp;
                end
            end
        end
    else
        depth_col2 = double(raw_depth2);
    end
else
    depth_col2 = nan(size(resultArray_wt,1),1);
end

% unitType 长度对齐
nUnits = size(resultArray_wt,1);
if numel(unitType) > nUnits
    unitType2 = unitType(1:nUnits);
elseif numel(unitType) < nUnits
    unitType2 = unitType(:);
    unitType2(numel(unitType2)+1:nUnits,1) = NaN;
else
    unitType2 = unitType(:);
end

% 只要 type1/2
valid_by_type = find(unitType2==1 | unitType2==2);

% 按 depth 分成 Str/ Ctx 并排序
depth_sel2 = depth_col2(valid_by_type);
isStr2 = depth_sel2 <= 2625;    % striatum
isCtx2 = depth_sel2 >  2625;    % cortex

str_units = valid_by_type(isStr2);
ctx_units = valid_by_type(isCtx2);

[~, ix_str2] = sort(depth_col2(str_units), 'ascend');
[~, ix_ctx2] = sort(depth_col2(ctx_units), 'ascend');

str_units = str_units(ix_str2);
ctx_units = ctx_units(ix_ctx2);

display_order2    = [str_units; ctx_units];
display_unitType2 = unitType2(display_order2);

n_str2 = numel(str_units);

% ---------- 3) 统计 binned spike rate ----------
bin_width3 = 0.01;                               % bin 宽度 = 0.01 s
time_bins3 = x_start:bin_width3:x_end;           % bin 边界
sel_spike_times3 = [];

% 收集所有显示出来的 unit（type1/2）的 spike time
for ii = 1:numel(display_order2)
    u = display_order2(ii);
    st = resultArray_wt(u,2:end);
    st = st(st > 0 & st >= x_start & st <= x_end);
    sel_spike_times3 = [sel_spike_times3, st]; %#ok<AGROW>
end

% 把 spike time 做 histogram → 每个 bin 内 spike 数
if isempty(sel_spike_times3)
    % 如果当前窗口内一个 spike 都没有，直接给全 0
    binned_spikes3 = zeros(1, numel(time_bins3)-1);
else
    spike_counts3   = histcounts(sel_spike_times3, time_bins3);
    % 单位：spikes/s（除以 bin 宽度）
    binned_spikes3  = spike_counts3 / bin_width3;
end

% bin 中心，用来画平滑曲线和找局部峰
time_bins3_center = time_bins3(1:end-1) + bin_width3/2;

% ---- 对 raster summation 做平滑拟合（同 GUI 里 smooth 逻辑）----
smooth_win3 = 3;  % 平滑窗口（单位：bin 数），=1 表示不平滑
if smooth_win3 > 1
    binned_spikes3_fit = smooth(binned_spikes3(:), smooth_win3);
else
    binned_spikes3_fit = binned_spikes3(:);
end

% 让拟合曲线的最大值与 histogram 的最大值一致
max_raw = max(binned_spikes3);
max_fit = max(binned_spikes3_fit);
if max_fit > 0
    scale_factor = max_raw / max_fit;
    binned_spikes3_fit = binned_spikes3_fit * scale_factor;
end

% bin 中心
time_bins3_center = time_bins3(1:end-1) + bin_width3/2;


% ---------- 3b) 读取 Excel 中的 Cortex / Striatum 峰值 ----------
% 目标 Excel 路径
peak_dir  = 'H:\seizure\2024-01-02_WT_HOM-male-adult\2024-01-02_13-43-59\Record Node 101\experiment2\recording2\continuous\Neuropix-PXI-100.ProbeA-LFP\Str_Cor_histPeak_diff';
peak_file = fullfile(peak_dir, 'qualityMatrix_20251106_152149_withFinalPeaks.xlsx');

% 初始化
ctx_peak_t_final = [];
ctx_peak_y_final = [];
str_peak_t_final = [];
str_peak_y_final = [];
Peak_Cortex      = [];
Peak_Striatum    = [];

if exist(peak_file,'file')
    Tpeak = readtable(peak_file);

    % C 列: cortex peak y
    colC = Tpeak{:,3};
    nC   = numel(colC);
    ctx_peak_y_final = nan(nC,1);
    for i = 1:nC
        v = colC(i);
        if iscell(v), v = v{1}; end
        if isnumeric(v)
            if isscalar(v)
                ctx_peak_y_final(i) = double(v);
            elseif ~isempty(v)
                ctx_peak_y_final(i) = double(v(1));
            end
        elseif ischar(v) || isstring(v)
            s = char(v);
            token = regexp(s, '[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', 'match', 'once');
            if ~isempty(token)
                ctx_peak_y_final(i) = str2double(token);
            end
        end
    end

    % H 列: striatum peak y
    colH = Tpeak{:,8};
    nH   = numel(colH);
    str_peak_y_final = nan(nH,1);
    for i = 1:nH
        v = colH(i);
        if iscell(v), v = v{1}; end
        if isnumeric(v)
            if isscalar(v)
                str_peak_y_final(i) = double(v);
            elseif ~isempty(v)
                str_peak_y_final(i) = double(v(1));
            end
        elseif ischar(v) || isstring(v)
            s = char(v);
            token = regexp(s, '[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', 'match', 'once');
            if ~isempty(token)
                str_peak_y_final(i) = str2double(token);
            end
        end
    end

    % L 列: striatum peak t (x)
    colL = Tpeak{:,12};
    nL   = numel(colL);
    str_peak_t_final = nan(nL,1);
    for i = 1:nL
        v = colL(i);
        if iscell(v), v = v{1}; end
        if isnumeric(v)
            if isscalar(v)
                str_peak_t_final(i) = double(v);
            elseif ~isempty(v)
                str_peak_t_final(i) = double(v(1));
            end
        elseif ischar(v) || isstring(v)
            s = char(v);
            token = regexp(s, '[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', 'match', 'once');
            if ~isempty(token)
                str_peak_t_final(i) = str2double(token);
            end
        end
    end

    % M 列: cortex peak t (x)
    colM = Tpeak{:,13};
    nM   = numel(colM);
    ctx_peak_t_final = nan(nM,1);
    for i = 1:nM
        v = colM(i);
        if iscell(v), v = v{1}; end
        if isnumeric(v)
            if isscalar(v)
                ctx_peak_t_final(i) = double(v);
            elseif ~isempty(v)
                ctx_peak_t_final(i) = double(v(1));
            end
        elseif ischar(v) || isstring(v)
            s = char(v);
            token = regexp(s, '[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', 'match', 'once');
            if ~isempty(token)
                ctx_peak_t_final(i) = str2double(token);
            end
        end
    end

    % 组装成 [t, y]
    Peak_Cortex   = [ctx_peak_t_final(:), ctx_peak_y_final(:)];
    Peak_Striatum = [str_peak_t_final(:), str_peak_y_final(:)];
else
    warning('Peak file not found: %s', peak_file);
end

% ---------- 3c) 以 CTX/STR 锚点，在拟合曲线上找 ±0.1 s 内局部峰值 ----------
anchor_win = 0.1;   % 秒

green_ctx_t = nan(size(ctx_peak_t_final));
green_ctx_y = nan(size(ctx_peak_t_final));

for i = 1:numel(ctx_peak_t_final)
    t0 = ctx_peak_t_final(i);
    if isnan(t0), continue; end
    idx = find(time_bins3_center >= t0 - anchor_win & time_bins3_center <= t0 + anchor_win);
    if isempty(idx), continue; end
    [ymax, j] = max(binned_spikes3_fit(idx));
    k = idx(j);
    green_ctx_t(i) = time_bins3_center(k);
    green_ctx_y(i) = ymax;
end

green_str_t = nan(size(str_peak_t_final));
green_str_y = nan(size(str_peak_t_final));

for i = 1:numel(str_peak_t_final)
    t0 = str_peak_t_final(i);
    if isnan(t0), continue; end
    idx = find(time_bins3_center >= t0 - anchor_win & time_bins3_center <= t0 + anchor_win);
    if isempty(idx), continue; end
    [ymax, j] = max(binned_spikes3_fit(idx));
    k = idx(j);
    green_str_t(i) = time_bins3_center(k);
    green_str_y(i) = ymax;
end

% ---------- 4) 三联图 ----------
figure('Color','w');
tl3 = tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

% ---- Panel 1: SWD detection in LFP ----
axA = nexttile(tl3,1); hold(axA,'on');
plot(axA, t_lfp, lfp_for_plot, 'Color',[0.6 0.6 0.6]);

ylA = ylim(axA);  % 保存原始 y 轴范围

% 用红色显示 swd_events（红 patch + LFP 红色高亮）
if exist('swd_events','var') && ~isempty(swd_events)
    for iEv = 1:numel(swd_events)
        ev = swd_events{iEv};
        if isempty(ev), continue; end

        % 事件的开始/结束时间（秒）
        t0 = (ev(1)-1)/fsLFP;
        t1 = (ev(end)-1)/fsLFP;

        % 与当前显示窗口相交才画
        if t1 < x_start || t0 > x_end
            continue;
        end

        % 截断在 x_start/x_end 范围内
        t0_clip = max(t0, x_start);
        t1_clip = min(t1, x_end);

        % 1) 红色半透明 patch 标记 SWD 区间
        patch(axA, [t0_clip t1_clip t1_clip t0_clip], ...
                    [ylA(1) ylA(1) ylA(2) ylA(2)], ...
                    [1 0 0], 'FaceAlpha',0.12, 'EdgeColor','none');

        % 2) 在 SWD 区间内，用红色重新画一遍 LFP 段
        idx0 = max(1, floor(t0_clip*fsLFP)+1);
        idx1 = min(numel(lfp_for_plot), ceil(t1_clip*fsLFP));
        if idx1 > idx0
            tt_seg  = t_lfp(idx0:idx1);
            lfp_seg = lfp_for_plot(idx0:idx1);
            plot(axA, tt_seg, lfp_seg, 'r', 'LineWidth', 1.0);
        end
    end

    % 恢复原始 y 轴范围
    ylim(axA, ylA);
end

xlim(axA, [x_start x_end]);
xlabel(axA,'Time (s)');
ylabel(axA,'LFP (arb.)');
title(axA,'SWD Detection in LFP (red = SWD events)');
grid(axA,'on');
hold(axA,'off');

% ---- Panel 2: Raster (unitType 1/2, depth sorted; Str on top, Ctx bottom) ----
axB = nexttile(tl3,2); hold(axB,'on');
for yy = 1:numel(display_order2)
    u = display_order2(yy);
    st = resultArray_wt(u,2:end);
    st = st(st>0 & st>=x_start & st<=x_end);
    if isempty(st), continue; end

    if display_unitType2(yy)==1
        thisColor = 'r';
    elseif display_unitType2(yy)==2
        thisColor = 'b';
    else
        thisColor = [0.3 0.3 0.3];
    end
    scatter(axB, st, yy*ones(size(st)), 6, thisColor, '.');
end

% Str / Ctx 分界线
if n_str2 > 0 && n_str2 < numel(display_order2)
    plot(axB, [x_start x_end], [n_str2+0.5 n_str2+0.5], 'k--', 'LineWidth',1);
end

xlim(axB, [x_start x_end]);
ylim(axB, [0 numel(display_order2)+1]);
ylabel(axB,'Neuron (depth sorted)');
title(axB,'Raster (unitType 1 = red, type 2 = blue)');
set(axB,'YDir','normal');
grid(axB,'on');
hold(axB,'off');

% ---- Panel 3: Binned spike rate ----
axC = nexttile(tl3,3); hold(axC,'on');
bar(axC, time_bins3(1:end-1), binned_spikes3, 1, ...
    'FaceColor',[0.2 0.2 0.2], 'EdgeColor','none');

% 叠加平滑拟合曲线
plot(axC, time_bins3_center, binned_spikes3_fit, '-', ...
    'LineWidth',1.5, 'Color',[1.0 0.4 0.0]);

% 标记 Cortex / Striatum 原始峰值（只画在当前 x 范围内）
if ~isempty(Peak_Cortex)
    maskC = ctx_peak_t_final >= x_start & ctx_peak_t_final <= x_end & ...
            ~isnan(ctx_peak_t_final) & ~isnan(ctx_peak_y_final);
    plot(axC, ctx_peak_t_final(maskC), ctx_peak_y_final(maskC), 'o', ...
        'MarkerSize',6, ...
        'MarkerFaceColor',[0.85 0 0], ...  % 深红
        'MarkerEdgeColor','k');
end

if ~isempty(Peak_Striatum)
    maskS = str_peak_t_final >= x_start & str_peak_t_final <= x_end & ...
            ~isnan(str_peak_t_final) & ~isnan(str_peak_y_final);
    plot(axC, str_peak_t_final(maskS), str_peak_y_final(maskS), '^', ...
        'MarkerSize',6, ...
        'MarkerFaceColor',[0 0 0.85], ...  % 深蓝
        'MarkerEdgeColor','k');
end

% 标记“以 CTX/STR 锚点在拟合曲线上找到的局部峰”（绿色）
maskGC = ~isnan(green_ctx_t) & green_ctx_t >= x_start & green_ctx_t <= x_end;
if any(maskGC)
    plot(axC, green_ctx_t(maskGC), green_ctx_y(maskGC), 'o', ...
        'MarkerSize',6, ...
        'MarkerFaceColor',[0 0.8 0], ...   % 绿色
        'MarkerEdgeColor','k');
end

maskGS = ~isnan(green_str_t) & green_str_t >= x_start & green_str_t <= x_end;
if any(maskGS)
    plot(axC, green_str_t(maskGS), green_str_y(maskGS), 'o', ...
        'MarkerSize',6, ...
        'MarkerFaceColor',[0 0.8 0], ...   % 同样绿色
        'MarkerEdgeColor','k');
end

xlim(axC, [x_start x_end]);
xlabel(axC,'Time (s)');
ylabel(axC,'Spike rate (spikes/s)');
title(axC, sprintf('Binned spike rate (bin = %.2f s, unitType 1/2 only)', bin_width3));
grid(axC,'on');
hold(axC,'off');

% ---- Link x-axis ----
linkaxes([axA axB axC],'x');

%%  %%%%%%%%%%%%%

% ===== FWHM-based duration around each green peak (CTX + STR) =====
% 使用拟合曲线 binned_spikes3_fit 与 time_bins3_center，
% 以每个绿色峰为锚点计算 FWHM，并标记在 axC 上。

hold(axC,'on');

% 把 CTX 和 STR 的绿色峰拼在一起统一处理
all_peak_t = [green_ctx_t(:); green_str_t(:)];
all_peak_y = [green_ctx_y(:); green_str_y(:)];

nFit = numel(binned_spikes3_fit);

for ip = 1:numel(all_peak_t)
    tp = all_peak_t(ip);
    yp = all_peak_y(ip);

    % 跳过 NaN 或不在当前显示窗口的点
    if isnan(tp) || isnan(yp), continue; end
    if tp < x_start || tp > x_end, continue; end

    % 找到最接近锚点时间的拟合曲线索引
    [~, idx0] = min(abs(time_bins3_center - tp));
    if idx0 < 1 || idx0 > nFit, continue; end

    y0 = binned_spikes3_fit(idx0);
    if y0 <= 0, continue; end

    half = y0 / 2;   % FWHM 的 half-height

    % ----- 向左找 half 交点 -----
    iL = idx0;
    while iL > 1 && binned_spikes3_fit(iL) >= half
        iL = iL - 1;
    end
    if iL == 1 && binned_spikes3_fit(iL) >= half
        % 左边一直没跌破 half，就用最左边的点
        tL = time_bins3_center(1);
    else
        % 在 iL 和 iL+1 之间做线性插值
        x1 = time_bins3_center(iL);
        y1 = binned_spikes3_fit(iL);
        x2 = time_bins3_center(iL+1);
        y2 = binned_spikes3_fit(iL+1);
        tL = x1 + (half - y1) * (x2 - x1) / (y2 - y1);
    end

    % ----- 向右找 half 交点 -----
    iR = idx0;
    while iR < nFit && binned_spikes3_fit(iR) >= half
        iR = iR + 1;
    end
    if iR == nFit && binned_spikes3_fit(iR) >= half
        % 右边一直没跌破 half，就用最右边的点
        tR = time_bins3_center(end);
    else
        % 在 iR-1 和 iR 之间做线性插值
        x1 = time_bins3_center(iR-1);
        y1 = binned_spikes3_fit(iR-1);
        x2 = time_bins3_center(iR);
        y2 = binned_spikes3_fit(iR);
        tR = x1 + (half - y1) * (x2 - x1) / (y2 - y1);
    end

    duration = tR - tL;   % FWHM duration (秒)

    % ----- 在图上标记 FWHM -----
    % 半高的绿色横线
    plot(axC, [tL tR], [half half], '-', ...
        'Color',[0 0.6 0], 'LineWidth',1.2);

    % 左右两条竖虚线（可选）
    plot(axC, [tL tL], [0 half], ':', 'Color',[0 0.6 0]);
    plot(axC, [tR tR], [0 half], ':', 'Color',[0 0.6 0]);

    % 在横线中点标注 duration
    text(axC, (tL + tR)/2, half, sprintf('%.3f s', duration), ...
        'Color',[0 0.5 0], ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom', ...
        'FontSize',8);
end

hold(axC,'off');



%% ===== DROP-IN: 导出 Fig2 的红/蓝/绿三点 + FWHM duration =====
% 依赖变量：
%   ctx_peak_t_final, ctx_peak_y_final   % 红点（Cortex）
%   str_peak_t_final, str_peak_y_final   % 蓝点（Striatum）
%   green_ctx_t, green_ctx_y             % 绿点（基于 CTX 的拟合峰）
%   time_bins3_center, binned_spikes3_fit
%   x_start, x_end

% ---------- 1) 重新用 FWHM 逻辑，以每个绿色 CTX 点为锚点计算 duration ----------
t_center = time_bins3_center(:);
fit_y    = binned_spikes3_fit(:);
nFit     = numel(fit_y);

nEvt       = numel(green_ctx_t);
FWHM_tL    = nan(nEvt,1);
FWHM_tR    = nan(nEvt,1);
FWHM_dur   = nan(nEvt,1);
FWHM_halfY = nan(nEvt,1);   % 半高对应的 y，可选

for i = 1:nEvt
    tp = green_ctx_t(i);
    yp = green_ctx_y(i);

    if isnan(tp) || isnan(yp)
        continue;
    end
    if tp < x_start || tp > x_end
        continue;
    end

    % 找到最接近锚点时间的拟合曲线索引
    [~, idx0] = min(abs(t_center - tp));
    if idx0 < 1 || idx0 > nFit
        continue;
    end

    y0 = fit_y(idx0);
    if y0 <= 0
        continue;
    end
    half = y0 / 2;
    FWHM_halfY(i) = half;

    % ----- 向左找 half 交点 -----
    iL = idx0;
    while iL > 1 && fit_y(iL) >= half
        iL = iL - 1;
    end
    if iL == 1 && fit_y(iL) >= half
        tL = t_center(1);
    else
        x1 = t_center(iL);   y1 = fit_y(iL);
        x2 = t_center(iL+1); y2 = fit_y(iL+1);
        tL = x1 + (half - y1) * (x2 - x1) / (y2 - y1);
    end

    % ----- 向右找 half 交点 -----
    iR = idx0;
    while iR < nFit && fit_y(iR) >= half
        iR = iR + 1;
    end
    if iR == nFit && fit_y(iR) >= half
        tR = t_center(end);
    else
        x1 = t_center(iR-1); y1 = fit_y(iR-1);
        x2 = t_center(iR);   y2 = fit_y(iR);
        tR = x1 + (half - y1) * (x2 - x1) / (y2 - y1);
    end

    FWHM_tL(i)  = tL;
    FWHM_tR(i)  = tR;
    FWHM_dur(i) = tR - tL;
end

% ---------- 2) 只保留“不被掩盖”的三点组（红/蓝/绿都在当前 x 范围 & duration 有效） ----------
event_idx = (1:numel(ctx_peak_t_final)).';

mask_visible = ...
    ~isnan(ctx_peak_t_final) & ctx_peak_t_final >= x_start & ctx_peak_t_final <= x_end & ...
    ~isnan(ctx_peak_y_final) & ...
    ~isnan(str_peak_t_final) & str_peak_t_final >= x_start & str_peak_t_final <= x_end & ...
    ~isnan(str_peak_y_final) & ...
    ~isnan(green_ctx_t)      & green_ctx_t      >= x_start & green_ctx_t      <= x_end & ...
    ~isnan(green_ctx_y)      & ...
    ~isnan(FWHM_dur);

idx_keep = find(mask_visible);

event_idx_keep = event_idx(idx_keep);

red_t   = ctx_peak_t_final(idx_keep);   % 红：Cortex 原始 peak
red_y   = ctx_peak_y_final(idx_keep);

blue_t  = str_peak_t_final(idx_keep);   % 蓝：Striatum 原始 peak
blue_y  = str_peak_y_final(idx_keep);

green_t = green_ctx_t(idx_keep);        % 绿：拟合后的峰（锚点）
green_y = green_ctx_y(idx_keep);

fwhm_tL_keep  = FWHM_tL(idx_keep);
fwhm_tR_keep  = FWHM_tR(idx_keep);
fwhm_dur_keep = FWHM_dur(idx_keep);
halfY_keep    = FWHM_halfY(idx_keep);

% ---------- 3) 打包为 table & struct ----------
Fig2_table = table( ...
    event_idx_keep, ...
    red_t,   red_y, ...
    blue_t,  blue_y, ...
    green_t, green_y, ...
    fwhm_tL_keep, fwhm_tR_keep, fwhm_dur_keep, halfY_keep, ...
    'VariableNames', { ...
        'event_idx', ...
        'ctx_t_red',  'ctx_y_red',  ...   % 红点（Cortex）
        'str_t_blue', 'str_y_blue', ...   % 蓝点（Striatum）
        'fit_t_green','fit_y_green', ...  % 绿点（拟合峰）
        'FWHM_tL', 'FWHM_tR', 'FWHM_duration', 'FWHM_halfHeight'});

Fig2_data = struct();
Fig2_data.event_idx        = event_idx_keep;
Fig2_data.ctx_t_red        = red_t;
Fig2_data.ctx_y_red        = red_y;
Fig2_data.str_t_blue       = blue_t;
Fig2_data.str_y_blue       = blue_y;
Fig2_data.fit_t_green      = green_t;
Fig2_data.fit_y_green      = green_y;
Fig2_data.FWHM_tL          = fwhm_tL_keep;
Fig2_data.FWHM_tR          = fwhm_tR_keep;
Fig2_data.FWHM_duration    = fwhm_dur_keep;
Fig2_data.FWHM_halfHeight  = halfY_keep;

% ---------- 4) 保存到指定路径下的 Fig2_data 文件夹 ----------
base_dir = 'H:\seizure\2024-01-02_WT_HOM-male-adult\2024-01-02_13-43-59\Record Node 101\experiment2\recording2\continuous\Neuropix-PXI-100.ProbeA-LFP\Str_Cor_histPeak_diff';
out_dir  = fullfile(base_dir, 'Fig2_data');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

timestamp = datestr(now,'yyyymmdd_HHMMSS');
mat_path  = fullfile(out_dir, ['Fig2_triple_points_' timestamp '.mat']);
xls_path  = fullfile(out_dir, ['Fig2_triple_points_' timestamp '.xlsx']);

save(mat_path, 'Fig2_data', 'Fig2_table');
writetable(Fig2_table, xls_path);

fprintf('[Fig2] Saved %d visible triples to:\n  %s\n  %s\n', ...
    numel(idx_keep), mat_path, xls_path);

%% PSTH
%% ====== Load SWD onset times from Excel ======
onset_base_dir = 'H:\seizure\2024-01-02_WT_HOM-male-adult\2024-01-02_13-43-59\Record Node 101\experiment2\recording2\continuous\Neuropix-PXI-100.ProbeA-LFP\Str_Cor_histPeak_diff';
onset_file     = 'swd_real_onset_FixMiss_20251106_150753.xlsx';
onset_path     = fullfile(onset_base_dir, onset_file);

if ~exist(onset_path,'file')
    error('Onset Excel not found: %s', onset_path);
end

onset_sec_all = readmatrix(onset_path);   % 每行一个 onset (s)
onset_sec_all = onset_sec_all(:);
onset_sec_all = onset_sec_all(~isnan(onset_sec_all));

nEvents = numel(onset_sec_all);
fprintf('Loaded %d SWD onsets from %s\n', nEvents, onset_path);

%% ====== PSTH 通用参数 ======
win_pre   = -1.0;    % onset 前 1 s
win_post  =  2.0;    % onset 后 2 s
bin_width = 0.01;    % 10 ms bin

edges_rel   = win_pre:bin_width:win_post;
centers_rel = edges_rel(1:end-1) + bin_width/2;

%% ------------------------------------------------------------------------
%% 1) CTX + STR 合并 PSTH
%% ------------------------------------------------------------------------

% ---------- 1.1 选择 neuron 行（CTX + STR 合并，只要 type1/2） ----------
if exist('selected_neurons_CTX','var') || exist('selected_neurons_STR','var')
    sel_rows_all = [];
    if exist('selected_neurons_CTX','var') && ~isempty(selected_neurons_CTX)
        sel_rows_all = [sel_rows_all; selected_neurons_CTX(:)];
    end
    if exist('selected_neurons_STR','var') && ~isempty(selected_neurons_STR)
        sel_rows_all = [sel_rows_all; selected_neurons_STR(:)];
    end
    sel_rows_all = unique(sel_rows_all);
else
    sel_rows_all = (1:size(resultArray_wt,1)).';
end

% 只保留 type1/2
if exist('unitType','var') && ~isempty(unitType)
    ut = unitType(:);
    mask_type12 = false(size(sel_rows_all));
    for i = 1:numel(sel_rows_all)
        r = sel_rows_all(i);
        if r <= numel(ut)
            mask_type12(i) = (ut(r)==1 || ut(r)==2);
        end
    end
    sel_rows_all = sel_rows_all(mask_type12);
end

if isempty(sel_rows_all)
    error('No neurons selected for combined CTX+STR PSTH.');
end

nNeurons_all = numel(sel_rows_all);
fprintf('Using %d neurons (CTX + STR combined) for PSTH.\n', nNeurons_all);

% ---------- 1.2 合并 spike time ----------
spk_all = [];
for i = 1:numel(sel_rows_all)
    r  = sel_rows_all(i);
    st = resultArray_wt(r,2:end);
    st = st(st>0);          % 去掉 0
    spk_all = [spk_all; st(:)]; %#ok<AGROW>   % 纵向拼接成列
end

if isempty(spk_all)
    error('No spikes found in selected neurons (combined).');
end

% ---------- 1.3 累积所有事件的相对时间 ----------
all_rel = [];

for k = 1:nEvents
    t0 = onset_sec_all(k);
    t_start = t0 + win_pre;
    t_end   = t0 + win_post;

    inwin = spk_all >= t_start & spk_all <= t_end;
    if ~any(inwin)
        continue;
    end

    rel = spk_all(inwin) - t0;   % 相对时间
    all_rel = [all_rel; rel(:)]; %#ok<AGROW>  % 一定用竖着拼
end

if isempty(all_rel)
    warning('No spikes fall into the [%g,%g] window around SWD onsets (combined).', win_pre, win_post);
    all_rel = 0;  % 防止 histcounts 报错
end

% ---------- 1.4 计算 PSTH 并 plot ----------
counts_all = histcounts(all_rel, edges_rel);          % 总 spikes 数
rate_pop_all = counts_all / (bin_width * nEvents);    % population spikes/s (每个事件平均)
rate_per_neuron_all = rate_pop_all / nNeurons_all;    % spikes/s/neuron

figure('Color','w');
bar(centers_rel, rate_per_neuron_all, 1.0, ...
    'FaceColor',[0.2 0.2 0.2], 'EdgeColor','none');

xlabel('Time relative to SWD onset (s)');
ylabel('Firing rate (spikes/s/neuron)');
title(sprintf('CTX + STR combined PSTH (nEvents = %d, nNeurons = %d)', ...
    nEvents, nNeurons_all));
xlim([win_pre win_post]);
grid on;


%% ------------------------------------------------------------------------
%% 2) Cortex 单独 PSTH
%% ------------------------------------------------------------------------

if exist('selected_neurons_CTX','var') && ~isempty(selected_neurons_CTX)

    % ---------- 2.1 选择 CTX neurons（只要 type1/2） ----------
    sel_rows_ctx = selected_neurons_CTX(:);

    if exist('unitType','var') && ~isempty(unitType)
        ut = unitType(:);
        mask_type12_ctx = false(size(sel_rows_ctx));
        for i = 1:numel(sel_rows_ctx)
            r = sel_rows_ctx(i);
            if r <= numel(ut)
                mask_type12_ctx(i) = (ut(r)==1 || ut(r)==2);
            end
        end
        sel_rows_ctx = sel_rows_ctx(mask_type12_ctx);
    end

    if isempty(sel_rows_ctx)
        warning('No CTX neurons (type1/2) selected for PSTH.');
    else
        nNeurons_ctx = numel(sel_rows_ctx);
        fprintf('Using %d CTX neurons for PSTH.\n', nNeurons_ctx);

        % ---------- 2.2 合并 CTX spike time ----------
        spk_ctx = [];
        for i = 1:numel(sel_rows_ctx)
            r  = sel_rows_ctx(i);
            st = resultArray_wt(r,2:end);
            st = st(st>0);
            spk_ctx = [spk_ctx; st(:)]; %#ok<AGROW>
        end

        if isempty(spk_ctx)
            warning('No CTX spikes found in selected neurons.');
        else
            % ---------- 2.3 累积所有事件的相对时间 ----------
            all_rel_ctx = [];
            for k = 1:nEvents
                t0 = onset_sec_all(k);
                t_start = t0 + win_pre;
                t_end   = t0 + win_post;

                inwin = spk_ctx >= t_start & spk_ctx <= t_end;
                if ~any(inwin)
                    continue;
                end
                rel = spk_ctx(inwin) - t0;
                all_rel_ctx = [all_rel_ctx; rel(:)]; %#ok<AGROW>
            end

            if isempty(all_rel_ctx)
                warning('No CTX spikes in window [%g,%g] around SWD onsets.', win_pre, win_post);
                all_rel_ctx = 0;
            end

            % ---------- 2.4 计算 PSTH 并 plot ----------
            counts_ctx = histcounts(all_rel_ctx, edges_rel);
            rate_pop_ctx = counts_ctx / (bin_width * nEvents);
            rate_per_neuron_ctx = rate_pop_ctx / nNeurons_ctx;

            figure('Color','w');
            bar(centers_rel, rate_per_neuron_ctx, 1.0, ...
                'FaceColor',[0.85 0.4 0.0], 'EdgeColor','none');  % 橙色一点

            xlabel('Time relative to SWD onset (s)');
            ylabel('Firing rate (spikes/s/neuron)');
            title(sprintf('Cortex PSTH (nEvents = %d, nNeurons = %d)', ...
                nEvents, nNeurons_ctx));
            xlim([win_pre win_post]);
            grid on;
        end
    end
else
    warning('selected_neurons_CTX not found or empty; skip CTX-only PSTH.');
end


%%

%% ====== 以 swd_real_onset_FixMiss 的 onset 为基准，画 CTX+STR 合并 PSTH + scatter ======

% ---------- 1) 读入 SWD onset 时间 ----------
onset_base_dir = 'H:\seizure\2024-01-02_WT_HOM-male-adult\2024-01-02_13-43-59\Record Node 101\experiment2\recording2\continuous\Neuropix-PXI-100.ProbeA-LFP\Str_Cor_histPeak_diff';
onset_file     = 'swd_real_onset_FixMiss_20251106_150753.xlsx';
onset_path     = fullfile(onset_base_dir, onset_file);

if ~exist(onset_path,'file')
    error('Onset Excel not found: %s', onset_path);
end

onset_sec_all = readmatrix(onset_path);   % 每行一个 onset (s)
onset_sec_all = onset_sec_all(:);
onset_sec_all = onset_sec_all(~isnan(onset_sec_all));

nEvents = numel(onset_sec_all);
fprintf('Loaded %d SWD onsets from %s\n', nEvents, onset_path);

% ---------- 2) 选择要包含的 neuron（CTX + STR 合并） ----------
if exist('selected_neurons_CTX','var') || exist('selected_neurons_STR','var')
    sel_rows = [];
    if exist('selected_neurons_CTX','var') && ~isempty(selected_neurons_CTX)
        sel_rows = [sel_rows; selected_neurons_CTX(:)];
    end
    if exist('selected_neurons_STR','var') && ~isempty(selected_neurons_STR)
        sel_rows = [sel_rows; selected_neurons_STR(:)];
    end
    sel_rows = unique(sel_rows);
else
    sel_rows = (1:size(resultArray_wt,1)).';
end

% 可选：只要 type1/2 的 neuron
if exist('unitType','var') && ~isempty(unitType)
    ut = unitType(:);
    mask_type12 = false(size(sel_rows));
    for i = 1:numel(sel_rows)
        r = sel_rows(i);
        if r <= numel(ut)
            mask_type12(i) = (ut(r)==1 || ut(r)==2);
        end
    end
    sel_rows = sel_rows(mask_type12);
end

if isempty(sel_rows)
    error('No neurons selected for PSTH (sel_rows is empty).');
end

nNeurons = numel(sel_rows);
fprintf('Using %d neurons (CTX + STR combined) for PSTH.\n', nNeurons);

% ---------- 3) 合并这些 neuron 的 spike times（秒） ----------
spk_all = [];
for i = 1:numel(sel_rows)
    r = sel_rows(i);
    st = resultArray_wt(r,2:end);
    st = st(st>0);          % 去掉 0 填充
    spk_all = [spk_all, st]; %#ok<AGROW>
end
spk_all = spk_all(:);

if isempty(spk_all)
    error('No spikes found in selected neurons.');
end

% ---------- 4) 构建 PSTH 参数 ----------
win_pre   = -1.0;    % onset 前 1 秒
win_post  =  2.0;    % onset 后 2 秒
bin_width = 0.01;    % 10 ms bin

edges_rel   = win_pre:bin_width:win_post;
centers_rel = edges_rel(1:end-1) + bin_width/2;

% ---------- 5) 累积所有事件的相对时间（用于 PSTH + scatter） ----------
all_rel = [];           % 所有 event 的相对时间（1 列）
event_idx_vec = [];     % 每个 spike 属于第几个 event（用于 raster scatter）

for k = 1:nEvents
    t0 = onset_sec_all(k);
    t_start = t0 + win_pre;
    t_end   = t0 + win_post;

    inwin = spk_all >= t_start & spk_all <= t_end;
    if ~any(inwin)
        continue;
    end

    rel = spk_all(inwin) - t0;   % 相对时间（列向量）
    all_rel       = [all_rel; rel(:)];               %#ok<AGROW>
    event_idx_vec = [event_idx_vec; k*ones(numel(rel),1)]; %#ok<AGROW>
end

if isempty(all_rel)
    warning('No spikes fall into the [%g,%g] window around SWD onsets.', win_pre, win_post);
    all_rel = 0;          % 防止 histcounts 报错
    event_idx_vec = 0;
end

% ---------- 6) 计算 PSTH：spikes/s/neuron ----------
counts   = histcounts(all_rel, edges_rel);          % 所有事件 + 所有 neuron 的总 spike 数
rate_pop = counts / (bin_width * nEvents);          % population spikes/s (平均每个事件)
rate_per_neuron = rate_pop / nNeurons;              % spikes/s/neuron

% ---------- 7) Plot 1: 标准 PSTH（bar） ----------
figure('Color','w');
bar(centers_rel, rate_per_neuron, 1.0, ...
    'FaceColor',[0.2 0.2 0.2], 'EdgeColor','none');
xlabel('Time relative to SWD onset (s)');
ylabel('Firing rate (spikes/s/neuron)');
title(sprintf('CTX + STR combined PSTH (nEvents = %d, nNeurons = %d)', nEvents, nNeurons));
xlim([win_pre win_post]);
grid on;

% ---------- 8) Plot 2: PSTH 的 scatter 版本（bin center vs rate） ----------
figure('Color','w');
scatter(centers_rel, rate_per_neuron, 15, 'filled');   % 每个 bin 中心一个点
xlabel('Time relative to SWD onset (s)');
ylabel('Firing rate (spikes/s/neuron)');
title('PSTH as scatter (bin centers)');
xlim([win_pre win_post]);
grid on;

% ---------- 9) Plot 3: event × spike 的 scatter（raster 风格） ----------
% x 轴：time relative to onset
% y 轴：event index（1..nEvents）
figure('Color','w');
scatter(all_rel, event_idx_vec, 5, 'k', '.');   % 每个 spike 一个点
xlabel('Time relative to SWD onset (s)');
ylabel('SWD event index');
title('All spikes relative to SWD onset (raster-style scatter)');
xlim([win_pre win_post]);
ylim([0.5, nEvents+0.5]);
set(gca,'YDir','normal');
grid on;

%% ====== 每个 event 单独的 PSTH（spikes/s/neuron），并在一张图上 overlay ======

M = numel(edges_rel) - 1;              % bin 数
rate_each_evt = nan(nEvents, M);       % 每行一个 event 的 PSTH

for k = 1:nEvents
    t0 = onset_sec_all(k);             % 第 k 个 SWD 的 onset (s)
    t_start = t0 + win_pre;
    t_end   = t0 + win_post;

    % 取这个 event 窗口内的 spike，并转成相对时间
    inwin = spk_all >= t_start & spk_all <= t_end;
    if ~any(inwin)
        % 如果这个 event 周围完全没 spike，就留 NaN（或 0）即可
        rate_each_evt(k,:) = 0;
        continue;
    end
    rel = spk_all(inwin) - t0;         % 相对 onset 的时间

    % 每个 event 自己的 PSTH：spikes/s/neuron
    cnt = histcounts(rel, edges_rel);                      % spikes / bin
    rate_evt = cnt / (bin_width * nNeurons);               % spikes/s/neuron

    rate_each_evt(k,:) = rate_evt;
end

% ---------- 画 overlay 图 ----------
figure('Color','w'); hold on;

% 先画每个 event 的 PSTH（浅灰色）
for k = 1:nEvents
    plot(centers_rel, rate_each_evt(k,:), '-', ...
        'Color',[0.8 0.8 0.8]);   % 可以改成更浅一点防止太亮
end

% 再画平均 PSTH（黑色粗线）
rate_mean = nanmean(rate_each_evt, 1);
plot(centers_rel, rate_mean, 'k-', 'LineWidth', 2);

xlabel('Time relative to SWD onset (s)');
ylabel('Firing rate (spikes/s/neuron)');
title(sprintf('PSTH per event overlaid (nEvents = %d, nNeurons = %d)', nEvents, nNeurons));
xlim([win_pre win_post]);
grid on;
hold off;
%% 导出 PSTH
%% ====== 导出每个 event 的 PSTH trace 到 Fig2_data（防止覆盖）======

% 目标文件夹
base_dir = 'H:\seizure\2024-01-02_WT_HOM-male-adult\2024-01-02_13-43-59\Record Node 101\experiment2\recording2\continuous\Neuropix-PXI-100.ProbeA-LFP\Str_Cor_histPeak_diff';
out_dir  = fullfile(base_dir, 'Fig2_data');
if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

% ---------- 如果还没有 rate_each_evt，就按当前参数重算一遍 ----------
% 需要已有变量：onset_sec_all, spk_all, win_pre, win_post, bin_width,
%               edges_rel, centers_rel, nNeurons, nEvents
if ~exist('rate_each_evt','var') || isempty(rate_each_evt)
    fprintf('rate_each_evt not found in workspace. Recomputing per-event PSTH...\n');
    M = numel(edges_rel) - 1;
    rate_each_evt = nan(nEvents, M);

    for k = 1:nEvents
        t0 = onset_sec_all(k);             % 第 k 个 SWD onset (s)
        t_start = t0 + win_pre;
        t_end   = t0 + win_post;

        inwin = spk_all >= t_start & spk_all <= t_end;
        if ~any(inwin)
            rate_each_evt(k,:) = 0;
            continue;
        end
        rel = spk_all(inwin) - t0;         % 相对 onset 时间

        cnt = histcounts(rel, edges_rel);            % spikes / bin
        rate_evt = cnt / (bin_width * nNeurons);     % spikes/s/neuron

        rate_each_evt(k,:) = rate_evt;
    end
else
    % 若已存在，检查尺寸
    M = numel(edges_rel) - 1;
    if size(rate_each_evt,2) ~= M
        error('Existing rate_each_evt has %d bins, but edges_rel implies %d bins.', ...
              size(rate_each_evt,2), M);
    end
end

% ---------- 打包成 struct，便于以后加载 ----------
PSTH_per_event = struct();
PSTH_per_event.rate_each_evt = rate_each_evt;   % [nEvents x nBins]
PSTH_per_event.centers_rel   = centers_rel(:);  % [nBins x 1]
PSTH_per_event.edges_rel     = edges_rel(:);    % [nBins+1 x 1]
PSTH_per_event.win_pre       = win_pre;
PSTH_per_event.win_post      = win_post;
PSTH_per_event.bin_width     = bin_width;
PSTH_per_event.onset_sec_all = onset_sec_all(:);
PSTH_per_event.nEvents       = nEvents;
PSTH_per_event.nNeurons      = nNeurons;

% ---------- 生成不覆盖的文件名 ----------
timestamp = datestr(now,'yyyymmdd_HHMMSS');
mat_path  = fullfile(out_dir, ['PSTH_per_event_' timestamp '.mat']);
xls_path  = fullfile(out_dir, ['PSTH_per_event_' timestamp '.xlsx']);

% ---------- 保存当前 PSTH 图到同一个文件夹（Fig2_data） ----------
figPSTH = gcf;  % 当前的 PSTH overlay 图

fig_png_path = fullfile(out_dir, ['PSTH_per_event_' timestamp '.png']);
fig_fig_path = fullfile(out_dir, ['PSTH_per_event_' timestamp '.fig']);

% 保存为高分辨率 PNG 和可再次编辑的 FIG
exportgraphics(figPSTH, fig_png_path, 'Resolution', 300);
savefig(figPSTH, fig_fig_path);


% ---------- 保存 .mat ----------
save(mat_path, 'PSTH_per_event', 'rate_each_evt', ...
               'centers_rel', 'edges_rel', ...
               'win_pre', 'win_post', 'bin_width', ...
               'onset_sec_all', 'nEvents', 'nNeurons', '-v7.3');

% ---------- 组装成 table 并保存成 Excel ----------
% 行：time bin；列：每个 event 的 PSTH(trace)
nBins = numel(centers_rel);
K     = size(rate_each_evt,1);

T_psth = table();
T_psth.time_center_s = centers_rel(:);   % 第一列：时间轴

for k = 1:K
    varName = sprintf('event_%03d', k);  % 列名：event_001, event_002, ...
    T_psth.(varName) = rate_each_evt(k,:).';   % 每列一个 event 的 PSTH
end

writetable(T_psth, xls_path);

fprintf('[PSTH export] Saved %d events × %d bins to:\n  %s\n  %s\n', ...
        K, nBins, mat_path, xls_path);



%%
function [swd_events_filtered, keep_idx, event_quality] = curate_swd_events( ...
    lfp_data, fs, swd_events, win_sec, timepoint_array, neuron_id, opts)
% curate_swd_events — Interactive GUI for SWD event curation (CWT + tri-state labeling)
%
% Additions:
%   • Go (G): jump to event by number or approximate time (s / mm:ss / hh:mm:ss[.fff])
%   • Hotkeys q/w/e label 0/1/2 and auto-advance to next event
%   • Figure-level key handler only (no per-control KeyPressFcn) to avoid double triggers
%   • Lightweight debounce in key handler
%
% Outputs:
%   swd_events_filtered : cell array of kept events (by keep_idx)
%   keep_idx            : logical Mx1 (true = keep)
%   event_quality       : Mx4 [event_id, t_start_s, t_end_s, label]
%
% Requires: Signal Processing Toolbox (cwt, butter/filtfilt, etc.)

    if nargin < 4 || isempty(win_sec), win_sec = 10; end
    if nargin < 5, timepoint_array = []; end
    if nargin < 6, neuron_id = []; end
    if nargin < 7 || isempty(opts), opts = struct(); end

    % -------- Options --------
    opt.max_neurons     = ifdef(opts,'max_neurons', 200);
    opt.marker          = ifdef(opts,'marker','.');
    opt.markersize      = ifdef(opts,'markersize',6);
    opt.show_raster     = ifdef(opts,'show_raster', true);
    opt.bin_w           = ifdef(opts,'bin_w', 0.05);
    opt.ylim_lfp        = ifdef(opts,'ylim_lfp', []);
    opt.cwt_freq_lim    = ifdef(opts,'cwt_freq_lim', [0.5 60]);
    opt.cwt_robust_prct = ifdef(opts,'cwt_robust_prct', [5 95]);
    opt.autosave_dir    = ifdef(opts,'autosave_dir','SWD_curation_autosave');

    % -------- Input reshape / guards --------
    lfp_data = lfp_data(:);
    N = numel(lfp_data);
    M = numel(swd_events);
    if M == 0
        warning('swd_events is empty; returning empty results.');
        swd_events_filtered = {};
        keep_idx = false(0,1);
        event_quality = zeros(0,4); % [event_id, t_start_s, t_end_s, label]
        return;
    end

    starts = zeros(M,1); ends = zeros(M,1); lens = zeros(M,1);
    for i = 1:M
        ev = swd_events{i};
        if isempty(ev)
            starts(i) = NaN; ends(i) = NaN; lens(i) = NaN;
        else
            starts(i) = max(1, ev(1));
            ends(i)   = min(N, ev(end));
            lens(i)   = ends(i) - starts(i) + 1;
        end
    end

    % ---- Raster preprocessing ----
    haveRaster = ~isempty(timepoint_array);
    if haveRaster
        nNeurons = size(timepoint_array,1);
        spikeCell = cell(nNeurons,1);
        for r = 1:nNeurons
            v = timepoint_array(r,:);
            v = v(~isnan(v) & v>0);
            spikeCell{r} = v(:);    % seconds
        end
        if isempty(neuron_id), neuron_id = (1:nNeurons).'; end
        plot_rows = min(opt.max_neurons, numel(spikeCell));
        row_idx   = (1:plot_rows).';
    else
        nNeurons = 0; row_idx = []; spikeCell = {};
    end

    % ====== State ======
    S.idx         = find(~isnan(starts),1,'first'); if isempty(S.idx), S.idx = 1; end
    S.view_global = false;
    S.show_raster = opt.show_raster;
    S.win_half    = max(0.5, win_sec/2);
    S.fs          = fs;
    S.lfp         = lfp_data;
    S.starts      = starts; S.ends = ends; S.lens = lens;
    S.M = M; S.N = N;
    S.swd_events  = swd_events;

    % Tri-state labels: NaN=unlabeled, 0=delete, 1=keep, 2=unsure
    S.labels      = nan(M,1);
    % keep compatibility (unlabeled/unsure/keep => true; delete => false)
    S.keep        = true(M,1);

    % LFP y-axis range
    if isempty(opt.ylim_lfp)
        ypad = 0.05 * range(lfp_data); if ypad == 0, ypad = 1; end
        S.ylim_fix = [min(lfp_data)-ypad, max(lfp_data)+ypad];
    else
        S.ylim_fix = opt.ylim_lfp;
    end

    % Raster / bins
    S.haveRaster  = haveRaster; S.spikeCell = spikeCell; S.row_idx = row_idx; S.neuron_id = neuron_id;
    S.bin_w       = opt.bin_w;

    % CWT options
    S.cwt_freq_lim    = opt.cwt_freq_lim;
    S.cwt_robust_prct = opt.cwt_robust_prct;
    S.cwt_show        = true;

    % Playback
    S.play        = false;
    S.play_hz     = 1.0;
    S.play_min_hz = 0.1;
    S.play_max_hz = 20.0;
    S.timer       = [];
    S.min_dt      = 0.05;

    % Session & autosave
    S.session_id  = char(datetime('now','Format','yyyyMMdd_HHmmss'));
    S.autosave_dir = opt.autosave_dir;
    if ~exist(S.autosave_dir,'dir'), mkdir(S.autosave_dir); end
    S.save_base   = fullfile(S.autosave_dir, ['SWD_curation_', S.session_id]);
    S.file_mat    = [S.save_base, '.mat'];
    S.file_csv    = [S.save_base, '.csv'];

    % ====== Figure / GUI ======
    S.fig = figure('Name','SWD curation','Color','w', ...
        'NumberTitle','off','Units','normalized','Position',[0.08 0.08 0.84 0.78], ...
        'WindowKeyPressFcn', @(~,e) keyHandler(e), ...
        'CloseRequestFcn',   @(~,~) onClose(true));

    % Axes
    posCWT = [0.07 0.72 0.90 0.20];
    posLFP = [0.07 0.50 0.90 0.19];
    posRAS = [0.07 0.33 0.90 0.12];
    posSUM = [0.07 0.24 0.90 0.07];

    S.ax0 = axes('Parent',S.fig,'Position',posCWT);  % CWT
    S.ax1 = axes('Parent',S.fig,'Position',posLFP);  % LFP
    S.ax2 = axes('Parent',S.fig,'Position',posRAS);  % Raster
    S.ax3 = axes('Parent',S.fig,'Position',posSUM);  % Summation
    if ~S.haveRaster, set(S.ax2,'Visible','off'); set(S.ax3,'Visible','off'); end

    % Controls (note: NO per-control KeyPressFcn to avoid double triggers)
    mk = @(str,pos,cb) uicontrol(S.fig,'Style','pushbutton','String',str, ...
        'Units','normalized','Position',pos,'Callback',cb,'FontSize',10);

    % Row 1
    y1 = 0.065; h1 = 0.05; w = 0.05; gap = 0.01; x = 0.05;
    mk('<< Prev (A)',       [x         y1 w h1], @(~,~) prevEv());      x = x + w + gap;
    mk('Next (D) >>',       [x         y1 w h1], @(~,~) nextEv());      x = x + w + gap;

    % Tri-state buttons
    wS = 0.05;
    mk('Keep (q_1)',          [x         y1 wS h1], @(~,~) setLabel(1));  x = x + wS + gap;
    mk('Delete (w_0)',        [x         y1 wS h1], @(~,~) setLabel(0));  x = x + wS + gap;
    mk('Unsure (e_2)',        [x         y1 wS h1], @(~,~) setLabel(2));  x = x + wS + gap;

    % Others
    mk('Toggle Keep (T)',   [x         y1 0.09 h1], @(~,~) toggleKeep()); x = x + 0.09 + gap;
    mk('Wider +1s (W)',     [x         y1 0.12 h1], @(~,~) widenWin());   x = x + 0.12 + gap;
    mk('Narrow -1s (S)',    [x         y1 0.12 h1], @(~,~) narrowWin());  x = x + 0.12 + gap;
    mk('Global/Local (V)',  [x         y1 0.12 h1], @(~,~) toggleGlobal()); x = x + 0.12 + gap;
    mk('Go (G)',            [x         y1 0.08 h1], @(~,~) jumpDialog());  x = x + 0.08 + gap;

    % Row 2
    y2 = 0.005; h2 = 0.05;
    S.btnPlay = uicontrol(S.fig,'Style','togglebutton','String','▶ Play (Space)', ...
        'Units','normalized','Position',[0.07 y2 0.16 h2], ...
        'Callback',@(h,~) togglePlay(), 'FontSize',10);

    uicontrol(S.fig,'Style','text','String','Speed (Hz):', ...
        'Units','normalized','Position',[0.25 y2 0.09 h2], ...
        'BackgroundColor','w','HorizontalAlignment','left','FontSize',10);

    S.edtHz = uicontrol(S.fig,'Style','edit','String',num2str(S.play_hz,'%.3g'), ...
        'Units','normalized','Position',[0.34 y2 0.10 h2], ...
        'Callback',@(h,~) setSpeedFromEdit(), 'FontSize',10);

    uicontrol(S.fig,'Style','pushbutton','String','Apply', ...
        'Units','normalized','Position',[0.45 y2 0.08 h2], ...
        'Callback',@(h,~) setSpeedFromEdit(), 'FontSize',10);

    uicontrol(S.fig,'Style','pushbutton','String','Show/Hide CWT (J)', ...
        'Units','normalized','Position',[0.74 y2 0.12 h2], ...
        'Callback',@(~,~) toggleCWT(), 'FontSize',10);

    uicontrol(S.fig,'Style','pushbutton','String','Show/Hide Raster (H)', ...
        'Units','normalized','Position',[0.87 y2 0.12 h2], ...
        'Callback',@(~,~) toggleRaster(), 'FontSize',10);

    S.txt = uicontrol(S.fig,'Style','text','Units','normalized',...
        'Position',[0.07 0.93 0.90 0.05],'BackgroundColor','w',...
        'HorizontalAlignment','left','FontSize',11,'String','');

    % Initial draw + wait
    drawEvent();
    autosave('init');
    uiwait(S.fig);

    % Output (compatibility)
    if isvalidStruct(S) && isfield(S,'keep')
        keep_idx = S.keep(:);
    else
        keep_idx = true(M,1);
    end
    swd_events_filtered = swd_events(keep_idx);

    % ---- Event quality matrix (M x 4) ----
    % Columns: [event_id, t_start_s, t_end_s, label]
    if isvalidStruct(S)
        lbl = S.labels(:);
        event_quality = [(1:S.M).', S.starts(:)/S.fs, S.ends(:)/S.fs, lbl];
    else
        event_quality = [(1:M).', starts(:)/fs, ends(:)/fs, nan(M,1)];
    end

    % Also push labels and quality to base workspace (optional)
    try
        assignin('base','swd_event_labels', S.labels);
        assignin('base','swd_keep_idx', keep_idx);
        assignin('base','swd_session_file_mat', S.file_mat);
        assignin('base','swd_session_file_csv', S.file_csv);
        assignin('base','swd_event_quality', event_quality);
    catch
    end

    % ==================== Nested functions ====================
    function drawEvent()
        if ~ishandle(S.ax1), return; end

        % valid event & window
        i = S.idx;
        if isnan(S.starts(i))
            valid = find(~isnan(S.starts));
            if isempty(valid), valid = 1; end
            [~,ii] = min(abs(valid - i));
            S.idx = valid(ii); i = S.idx;
        end
        t = (1:S.N)/S.fs;

        % LFP axis
        cla(S.ax1); hold(S.ax1,'on');
        if S.view_global
            plot(S.ax1, t, S.lfp, 'Color', [0.7 0.7 0.7]);
            xs = [S.starts(i) S.ends(i)]/S.fs;
            patch(S.ax1,[xs(1) xs(2) xs(2) xs(1)], ...
                         [S.ylim_fix(1) S.ylim_fix(1) S.ylim_fix(2) S.ylim_fix(2)], ...
                         [1 0.8 0.8],'EdgeColor','none','FaceAlpha',0.35);
            xlim(S.ax1, [t(1) t(end)]);
            ylim(S.ax1, S.ylim_fix);
            lo = 1; hi = S.N;
        else
            center = round((S.starts(i)+S.ends(i))/2);
            halfN  = round(S.win_half * S.fs);
            lo = max(1, center - halfN);
            hi = min(S.N, center + halfN);

            plot(S.ax1, t(lo:hi), S.lfp(lo:hi), 'Color', [0.6 0.6 0.6]);
            ev_lo = max(S.starts(i), lo); ev_hi = min(S.ends(i), hi);
            if ev_hi >= ev_lo
                idx = ev_lo:ev_hi;
                plot(S.ax1, t(idx), S.lfp(idx), 'r','LineWidth',1.5);
            end
            xlim(S.ax1, [t(lo) t(hi)]);
            ylim(S.ax1, S.ylim_fix);
        end
        ylabel(S.ax1,'Amplitude'); title(S.ax1,'SWD Curation'); grid(S.ax1,'on');

        % CWT axis
        if S.cwt_show && ishandle(S.ax0)
            cla(S.ax0); hold(S.ax0,'on');
            sig_win = S.lfp(lo:hi);
            t_win   = ((lo:hi)-1)/S.fs;

            try
                [cfs, f] = cwt(sig_win, S.fs, 'amor', 'FrequencyLimits', S.cwt_freq_lim);
            catch
                [cfs, f] = cwt(sig_win, S.fs, 'amor');
                fmask = f >= S.cwt_freq_lim(1) & f <= S.cwt_freq_lim(2);
                cfs   = cfs(fmask,:);
                f     = f(fmask);
            end

            Z = 20*log10(abs(cfs) + eps);
            pr = S.cwt_robust_prct;
            clim = [prctile(Z(:),pr(1)), prctile(Z(:),pr(2))];

            imagesc(S.ax0, t_win, f, Z); axis(S.ax0,'xy');
            xlim(S.ax0, [t_win(1) t_win(end)]); ylim(S.ax0, S.cwt_freq_lim);
            ylabel(S.ax0,'Freq (Hz)'); title(S.ax0,'CWT (amor, 0.5–60 Hz)');
            colormap(S.ax0,'jet'); caxis(S.ax0,clim);
            hold(S.ax0,'off');
        end

        % Raster
        if S.haveRaster
            cla(S.ax2); hold(S.ax2,'on');
            if S.view_global
                text(0.02,0.5,'Raster hidden in Global view (press V to switch to Local)',...
                    'Units','normalized','Parent',S.ax2);
                set(S.ax2,'YTick',[]); xlim(S.ax2, [t(1) t(end)]);
                cla(S.ax3); set(S.ax3,'Visible','off');
            else
                set(S.ax3,'Visible','on');
                tlo = t(lo); thi = t(hi);

                for r = 1:numel(S.row_idx)
                    rr = S.row_idx(r);
                    v = S.spikeCell{rr};
                    if isempty(v), continue; end
                    mask = (v >= tlo) & (v <= thi);
                    if any(mask)
                        x_ = v(mask); y_ = r * ones(sum(mask),1);
                        plot(S.ax2, x_, y_, opt.marker, 'MarkerSize', opt.markersize, 'Color', [0.1 0.1 0.1]);
                    end
                end
                xlim(S.ax2, [tlo thi]);
                ylim(S.ax2, [0 max(1,numel(S.row_idx))+1]);
                set(S.ax2,'YDir','normal'); grid(S.ax2,'on');
                ylabel(S.ax2, sprintf('Neurons (1..%d)', numel(S.row_idx)));
                set(S.ax2,'XTickLabel',[]);

                % Summation
                cla(S.ax3); hold(S.ax3,'on');
                edges = tlo:S.bin_w:thi;
                if numel(edges) < 2
                    edges = linspace(tlo, thi, max(2, ceil((thi-tlo)/S.bin_w)));
                end
                counts = zeros(1, numel(edges)-1);
                for r = 1:numel(S.row_idx)
                    rr = S.row_idx(r);
                    v = S.spikeCell{rr};
                    if isempty(v), continue; end
                    mask = (v >= tlo) & (v <= thi);
                    if any(mask)
                        counts = counts + histcounts(v(mask), edges);
                    end
                end
                ctrs = (edges(1:end-1) + edges(2:end))/2;
                stairs(S.ax3, ctrs, counts, 'LineWidth',1.2);
                xlim(S.ax3, [tlo thi]);
                grid(S.ax3,'on');
                ylabel(S.ax3, 'Spikes/bin');
                xlabel(S.ax3, sprintf('Time (s)  |  bin=%.3g s', S.bin_w));
            end

            if ~S.show_raster
                set(S.ax2,'Visible','off'); set(S.ax3,'Visible','off');
            else
                set(S.ax2,'Visible','on'); if ~S.view_global, set(S.ax3,'Visible','on'); end
            end
        end

        % Status bar
        dur_s = S.lens(i)/S.fs;
        lab   = S.labels(i);
        if isnan(lab), lab_str = 'unlabeled'; else, lab_str = num2str(lab); end
        set(S.txt,'String',sprintf(['Event %d/%d | start=%.3fs, end=%.3fs, dur=%.3fs ',...
            '| label=%s (0=del,1=keep,2=unsure) | keep=%d | win=%.2fs | global=%d | raster=%d/%d | bin=%.3gs | spd=%.3g Hz | play=%d | CWT=%d'],...
            i, S.M, S.starts(i)/S.fs, S.ends(i)/S.fs, dur_s, lab_str, S.keep(i), ...
            2*S.win_half, S.view_global, ...
            (S.haveRaster)*numel(S.row_idx), (S.haveRaster)*nNeurons, ...
            S.bin_w, S.play_hz, S.play, S.cwt_show));

        % link x-axes
        axs = [S.ax1 S.ax2 S.ax3]; if S.cwt_show, axs = [S.ax0 axs]; end
        try, linkaxes(axs,'x'); catch, end
        set([S.ax0 S.ax1 S.ax2 S.ax3],'HitTest','off','PickableParts','none');
    end

    function setLabel(val)
        % 0=delete, 1=keep, 2=unsure
        i = S.idx;
        S.labels(i) = val;
        S.keep(i)   = ~(val == 0);
        drawEvent();
        autosave('label');
    end

    function nextEv()
        S.idx = min(S.M, S.idx+1);
        drawEvent();
        autosave('nav');
        if S.idx >= S.M && S.play
            S.play = false; updatePlayButton(); stopTimer();
        end
    end

    function prevEv()
        S.idx = max(1, S.idx-1);
        drawEvent();
        autosave('nav');
    end

    function toggleKeep()
        i = S.idx;
        S.keep(i) = ~S.keep(i);
        if isnan(S.labels(i)) || S.labels(i)==1 || S.labels(i)==0
            S.labels(i) = double(S.keep(i)); % 1 or 0
        end
        drawEvent(); autosave('toggle');
    end

    function widenWin(),  S.win_half = min(30, S.win_half + 0.5); drawEvent(); end
    function narrowWin(), S.win_half = max(0.5, S.win_half - 0.5); drawEvent(); end
    function toggleGlobal(), S.view_global = ~S.view_global; drawEvent(); end
    function toggleRaster(), S.show_raster = ~S.show_raster; drawEvent(); end

    function toggleCWT()
        S.cwt_show = ~S.cwt_show;
        set(S.ax0,'Visible', tern(S.cwt_show,'on','off'));
        drawEvent();
    end

    function togglePlay()
        S.play = ~S.play; updatePlayButton();
        if S.play, startTimer(); else, stopTimer(); end
    end

    function updatePlayButton()
        if S.play, set(S.btnPlay,'Value',1,'String','⏸ Pause (Space)');
        else,      set(S.btnPlay,'Value',0,'String','▶ Play (Space)'); end
    end

    function setSpeedFromEdit()
        str = strtrim(get(S.edtHz,'String'));
        v = str2double(str); if isnan(v) || ~isfinite(v), v = S.play_hz; end
        v = max(S.play_min_hz, min(S.play_max_hz, v));
        S.play_hz = v;
        set(S.edtHz,'String',num2str(S.play_hz,'%.3g'));
        if S.play, startTimer(); end
        drawEvent();
    end

    function startTimer()
        stopTimer();
        dt = max(S.min_dt, 1.0 / S.play_hz);
        S.timer = timer('ExecutionMode','fixedSpacing','Period',dt, ...
                        'TimerFcn',@(~,~) nextEv(), ...
                        'StartDelay',dt,'Tag','SWD_Autoplay_Hz');
        try, start(S.timer); catch, end
    end

    function stopTimer()
        if ~isempty(S.timer) && isvalid(S.timer)
            try, stop(S.timer); delete(S.timer); catch, end
        end
        S.timer = [];
    end

    function autosave(reason)
        try
            S_out = struct();
            S_out.session_id = S.session_id;
            S_out.saved_at   = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
            S_out.reason     = reason;
            S_out.fs         = S.fs;
            S_out.N          = S.N;
            S_out.M          = S.M;
            S_out.starts     = S.starts;
            S_out.ends       = S.ends;
            S_out.labels     = S.labels;
            S_out.keep       = S.keep;
            S_out.win_half   = S.win_half;
            S_out.cwt_freq   = S.cwt_freq_lim;
            S_out.bin_w      = S.bin_w;

            save(S.file_mat,'-struct','S_out');

            T = table((1:S.M).', S.starts(:)/S.fs, S.ends(:)/S.fs, S.labels(:), S.keep(:), ...
                      'VariableNames',{'event_id','t_start_s','t_end_s','label','keep'});
            writetable(T, S.file_csv);

            assignin('base','swd_event_labels', S.labels);
            assignin('base','swd_keep_idx', S.keep);
            assignin('base','swd_session_file_mat', S.file_mat);
            assignin('base','swd_session_file_csv', S.file_csv);
        catch ME
            warning('Autosave failed: %s', ME.message);
        end
    end

    function onClose(~)
        stopTimer();
        autosave('close');
        if ishandle(S.fig)
            try, uiresume(S.fig); catch, end
            delete(S.fig);
        end
    end

    % ------- Jump dialog & helpers -------
    function jumpDialog()
        prompt = {'Event # (1..M):', 'Time (s or mm:ss or hh:mm:ss):'};
        def    = {num2str(S.idx), ''};
        answ   = inputdlg(prompt,'Go to',1,def);
        if isempty(answ), return; end

        ev_str = strtrim(answ{1});
        tm_str = strtrim(answ{2});

        % Prefer time if provided
        if ~isempty(tm_str)
            t = parseTimeString(tm_str);
            if ~isnan(t)
                k = nearestEventAtOrAroundTime(t);
                if ~isnan(k)
                    S.idx = k; drawEvent(); autosave('nav'); return;
                end
            end
        end

        % Fallback to event number
        if ~isempty(ev_str)
            j = str2double(ev_str);
            if ~isnan(j) && j>=1 && j<=S.M
                S.idx = round(j); drawEvent(); autosave('nav'); return;
            end
        end
    end

    function k = nearestEventAtOrAroundTime(t_s)
        if any(~isnan(S.starts))
            t_start = S.starts / S.fs;
            t_end   = S.ends   / S.fs;
            inEv = (t_s >= t_start) & (t_s <= t_end);
            if any(inEv)
                k = find(inEv,1,'first');  % inside event
                return;
            end
            centers = (S.starts + S.ends) / (2*S.fs);
            [~,k] = min(abs(centers - t_s));
        else
            k = NaN;
        end
    end

    function t = parseTimeString(s)
        s = strtrim(s);
        if isempty(s), t = NaN; return; end
        if contains(s,':')
            tokens = regexp(s,':','split');
            nums = cellfun(@str2double, tokens);
            if any(isnan(nums)), t = NaN; return; end
            switch numel(nums)
                case 2  % mm:ss
                    t = nums(1)*60 + nums(2);
                case 3  % hh:mm:ss
                    t = nums(1)*3600 + nums(2)*60 + nums(3);
                otherwise
                    t = NaN;
            end
        else
            t = str2double(s);
            if ~isfinite(t), t = NaN; end
        end
    end

    function out = tern(cond, a, b)
        if cond, out = a; else, out = b; end
    end

    % ------- Keyboard handler (q/w/e => label & auto-next) with debounce -------
    function keyHandler(e)
        % Debounce: ignore same key repeated within ~120 ms
        persistent lastKeyName lastKeyTS;
        if isempty(lastKeyTS), lastKeyTS = now; end
        if isempty(lastKeyName), lastKeyName = ''; end

        k = lower(e.Key);
        mods = e.Modifier;                 % cell array: 'shift','control','alt',...
        hasShift = ~isempty(mods) && any(strcmpi(mods,'shift'));

        % compute elapsed seconds since last key
        dt_sec = (now - lastKeyTS) * 86400;    % days -> seconds
        if strcmpi(k,lastKeyName) && dt_sec < 0.12
            return; % too soon, likely duplicate
        end
        lastKeyName = k;
        lastKeyTS   = now;

        switch k
            % ---- q/w/e => label 0/1/2 then auto-advance ----
            case 'q'      % label 0 (delete) + next
                setLabel(1); nextEv();
            case 'w'
                if hasShift
                    widenWin();            % Shift+W keeps widen behavior
                else
                    setLabel(0); nextEv(); % w => keep + next
                end
            case 'e'      % label 2 (unsure) + next
                setLabel(2); nextEv();

            % ---- numeric labels (no auto-advance) ----
            case '0'
                setLabel(0);
            case '1'
                setLabel(1);
            case '2'
                setLabel(2);

            % ---- navigation ----
            case 'a'
                prevEv();
            case 'd'
                nextEv();

            % ---- toggles / view ----
            case 't'
                toggleKeep();
            case 's'
                narrowWin();
            case 'v'
                toggleGlobal();
            case 'g'
                jumpDialog();
            case 'j'
                toggleCWT();
            case 'h'
                toggleRaster();

            % ---- playback ----
            case 'space'
                togglePlay();

            otherwise
                % no-op
        end
    end
end

% ===== file-scope helpers =====
function v = ifdef(s, f, d)
    if isstruct(s) && isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
function tf = isvalidStruct(S)
    tf = ~isempty(S) && isstruct(S);
end

%%
% function [swd_events_filtered, keep_idx] = curate_swd_events( ...
%     lfp_data, fs, swd_events, win_sec, timepoint_array, neuron_id, opts)
% % curate_swd_events — Interactive GUI for SWD event curation (CWT + tri-state labeling)
% %
% % Additions:
% %   • Go (G): jump to event by number or approximate time (s / mm:ss / hh:mm:ss[.fff])
% %   • Everything else keeps the original behavior
% %
% % Outputs:
% %   swd_events_filtered : cell array of kept events (by keep_idx)
% %   keep_idx            : logical Mx1 (true = keep)
% %
% % Requires: Signal Processing Toolbox (cwt, butter/filtfilt, etc.)
% 
%     if nargin < 4 || isempty(win_sec), win_sec = 10; end
%     if nargin < 5, timepoint_array = []; end
%     if nargin < 6, neuron_id = []; end
%     if nargin < 7 || isempty(opts), opts = struct(); end
% 
%     % -------- Options --------
%     opt.max_neurons     = ifdef(opts,'max_neurons', 200);
%     opt.marker          = ifdef(opts,'marker','.');
%     opt.markersize      = ifdef(opts,'markersize',6);
%     opt.show_raster     = ifdef(opts,'show_raster', true);
%     opt.bin_w           = ifdef(opts,'bin_w', 0.05);
%     opt.ylim_lfp        = ifdef(opts,'ylim_lfp', []);
%     opt.cwt_freq_lim    = ifdef(opts,'cwt_freq_lim', [0.5 60]);
%     opt.cwt_robust_prct = ifdef(opts,'cwt_robust_prct', [5 95]);
%     opt.autosave_dir    = ifdef(opts,'autosave_dir','SWD_curation_autosave');
% 
%     % -------- Input reshape / guards --------
%     lfp_data = lfp_data(:);
%     N = numel(lfp_data);
%     M = numel(swd_events);
%     if M == 0
%         warning('swd_events is empty; returning empty results.');
%         swd_events_filtered = {};
%         keep_idx = false(0,1);
%         return;
%     end
% 
%     starts = zeros(M,1); ends = zeros(M,1); lens = zeros(M,1);
%     for i = 1:M
%         ev = swd_events{i};
%         if isempty(ev)
%             starts(i) = NaN; ends(i) = NaN; lens(i) = NaN;
%         else
%             starts(i) = max(1, ev(1));
%             ends(i)   = min(N, ev(end));
%             lens(i)   = ends(i) - starts(i) + 1;
%         end
%     end
% 
%     % ---- Raster preprocessing ----
%     haveRaster = ~isempty(timepoint_array);
%     if haveRaster
%         nNeurons = size(timepoint_array,1);
%         spikeCell = cell(nNeurons,1);
%         for r = 1:nNeurons
%             v = timepoint_array(r,:);
%             v = v(~isnan(v) & v>0);
%             spikeCell{r} = v(:);    % seconds
%         end
%         if isempty(neuron_id), neuron_id = (1:nNeurons).'; end
%         plot_rows = min(opt.max_neurons, numel(spikeCell));
%         row_idx   = (1:plot_rows).';
%     else
%         nNeurons = 0; row_idx = []; spikeCell = {};
%     end
% 
%     % ====== State ======
%     S.idx         = find(~isnan(starts),1,'first'); if isempty(S.idx), S.idx = 1; end
%     S.view_global = false;
%     S.show_raster = opt.show_raster;
%     S.win_half    = max(0.5, win_sec/2);
%     S.fs          = fs;
%     S.lfp         = lfp_data;
%     S.starts      = starts; S.ends = ends; S.lens = lens;
%     S.M = M; S.N = N;
%     S.swd_events  = swd_events;
% 
%     % Tri-state labels: NaN=unlabeled, 0=delete, 1=keep, 2=unsure
%     S.labels      = nan(M,1);
%     % keep compatibility (unlabeled/unsure/keep => true; delete => false)
%     S.keep        = true(M,1);
% 
%     % LFP y-axis range
%     if isempty(opt.ylim_lfp)
%         ypad = 0.05 * range(lfp_data); if ypad == 0, ypad = 1; end
%         S.ylim_fix = [min(lfp_data)-ypad, max(lfp_data)+ypad];
%     else
%         S.ylim_fix = opt.ylim_lfp;
%     end
% 
%     % Raster / bins
%     S.haveRaster  = haveRaster; S.spikeCell = spikeCell; S.row_idx = row_idx; S.neuron_id = neuron_id;
%     S.bin_w       = opt.bin_w;
% 
%     % CWT options
%     S.cwt_freq_lim    = opt.cwt_freq_lim;
%     S.cwt_robust_prct = opt.cwt_robust_prct;
%     S.cwt_show        = true;
% 
%     % Playback
%     S.play        = false;
%     S.play_hz     = 1.0;
%     S.play_min_hz = 0.1;
%     S.play_max_hz = 20.0;
%     S.timer       = [];
%     S.min_dt      = 0.05;
% 
%     % Session & autosave
%     S.session_id  = char(datetime('now','Format','yyyyMMdd_HHmmss'));
%     S.autosave_dir = opt.autosave_dir;
%     if ~exist(S.autosave_dir,'dir'), mkdir(S.autosave_dir); end
%     S.save_base   = fullfile(S.autosave_dir, ['SWD_curation_', S.session_id]);
%     S.file_mat    = [S.save_base, '.mat'];
%     S.file_csv    = [S.save_base, '.csv'];
% 
%     % ====== Figure / GUI ======
%     S.fig = figure('Name','SWD curation','Color','w', ...
%         'NumberTitle','off','Units','normalized','Position',[0.08 0.08 0.84 0.78], ...
%         'WindowKeyPressFcn', @(~,e) keyHandler(e), ...
%         'CloseRequestFcn',   @(~,~) onClose(true));
% 
%     % Axes
%     posCWT = [0.07 0.72 0.90 0.20];
%     posLFP = [0.07 0.50 0.90 0.19];
%     posRAS = [0.07 0.33 0.90 0.12];
%     posSUM = [0.07 0.24 0.90 0.07];
% 
%     S.ax0 = axes('Parent',S.fig,'Position',posCWT);  % CWT
%     S.ax1 = axes('Parent',S.fig,'Position',posLFP);  % LFP
%     S.ax2 = axes('Parent',S.fig,'Position',posRAS);  % Raster
%     S.ax3 = axes('Parent',S.fig,'Position',posSUM);  % Summation
%     if ~S.haveRaster, set(S.ax2,'Visible','off'); set(S.ax3,'Visible','off'); end
% 
%     % Controls
%     mk = @(str,pos,cb) uicontrol(S.fig,'Style','pushbutton','String',str, ...
%         'Units','normalized','Position',pos,'Callback',cb, ...
%         'KeyPressFcn',@(h,e) keyHandler(e),'FontSize',10);
% 
%     % Row 1
%     y1 = 0.065; h1 = 0.05; w = 0.05; gap = 0.01; x = 0.05;
%     mk('<< Prev (A)',       [x         y1 w h1], @(~,~) prevEv());      x = x + w + gap;
%     mk('Next (D) >>',       [x         y1 w h1], @(~,~) nextEv());      x = x + w + gap;
% 
%     % Tri-state buttons
%     wS = 0.05;
%     mk('Keep (1)',          [x         y1 wS h1], @(~,~) setLabel(1));  x = x + wS + gap;
%     mk('Delete (0)',        [x         y1 wS h1], @(~,~) setLabel(0));  x = x + wS + gap;
%     mk('Unsure (2)',        [x         y1 wS h1], @(~,~) setLabel(2));  x = x + wS + gap;
% 
%     % Others
%     mk('Toggle Keep (T)',   [x         y1 0.09 h1], @(~,~) toggleKeep()); x = x + 0.09 + gap;
%     mk('Wider +1s (W)',     [x         y1 0.12 h1], @(~,~) widenWin());   x = x + 0.12 + gap;
%     mk('Narrow -1s (S)',    [x         y1 0.12 h1], @(~,~) narrowWin());  x = x + 0.12 + gap;
%     mk('Global/Local (V)',  [x         y1 0.12 h1], @(~,~) toggleGlobal()); x = x + 0.12 + gap;
%     mk('Go (G)',            [x         y1 0.08 h1], @(~,~) jumpDialog());  x = x + 0.08 + gap;
% 
%     % Row 2
%     y2 = 0.005; h2 = 0.05;
%     S.btnPlay = uicontrol(S.fig,'Style','togglebutton','String','▶ Play (Space)', ...
%         'Units','normalized','Position',[0.07 y2 0.16 h2], ...
%         'Callback',@(h,~) togglePlay(), 'KeyPressFcn',@(h,e) keyHandler(e), ...
%         'FontSize',10);
% 
%     uicontrol(S.fig,'Style','text','String','Speed (Hz):', ...
%         'Units','normalized','Position',[0.25 y2 0.09 h2], ...
%         'BackgroundColor','w','HorizontalAlignment','left','FontSize',10);
% 
%     S.edtHz = uicontrol(S.fig,'Style','edit','String',num2str(S.play_hz,'%.3g'), ...
%         'Units','normalized','Position',[0.34 y2 0.10 h2], ...
%         'Callback',@(h,~) setSpeedFromEdit(), 'KeyPressFcn',@(h,e) keyHandler(e), ...
%         'FontSize',10);
% 
%     uicontrol(S.fig,'Style','pushbutton','String','Apply', ...
%         'Units','normalized','Position',[0.45 y2 0.08 h2], ...
%         'Callback',@(h,~) setSpeedFromEdit(), 'KeyPressFcn',@(h,e) keyHandler(e), ...
%         'FontSize',10);
% 
%     uicontrol(S.fig,'Style','pushbutton','String','Show/Hide CWT (J)', ...
%         'Units','normalized','Position',[0.74 y2 0.12 h2], ...
%         'Callback',@(~,~) toggleCWT(), 'KeyPressFcn',@(h,e) keyHandler(e), ...
%         'FontSize',10);
% 
%     uicontrol(S.fig,'Style','pushbutton','String','Show/Hide Raster (H)', ...
%         'Units','normalized','Position',[0.87 y2 0.12 h2], ...
%         'Callback',@(~,~) toggleRaster(), 'KeyPressFcn',@(h,e) keyHandler(e), ...
%         'FontSize',10);
% 
%     S.txt = uicontrol(S.fig,'Style','text','Units','normalized',...
%         'Position',[0.07 0.93 0.90 0.05],'BackgroundColor','w',...
%         'HorizontalAlignment','left','FontSize',11,'String','');
% 
%     % Initial draw + wait
%     drawEvent();
%     autosave('init');
%     uiwait(S.fig);
% 
%     % Output (compatibility)
%     if isvalidStruct(S) && isfield(S,'keep')
%         keep_idx = S.keep(:);
%     else
%         keep_idx = true(M,1);
%     end
%     swd_events_filtered = swd_events(keep_idx);
% 
%     % Also push labels into base workspace (optional)
%     try
%         assignin('base','swd_event_labels', S.labels);
%         assignin('base','swd_keep_idx', keep_idx);
%         assignin('base','swd_session_file_mat', S.file_mat);
%         assignin('base','swd_session_file_csv', S.file_csv);
%     catch
%     end
% 
%     % ==================== Nested functions ====================
%     function drawEvent()
%         if ~ishandle(S.ax1), return; end
% 
%         % valid event & window
%         i = S.idx;
%         if isnan(S.starts(i))
%             valid = find(~isnan(S.starts));
%             if isempty(valid), valid = 1; end
%             [~,ii] = min(abs(valid - i));
%             S.idx = valid(ii); i = S.idx;
%         end
%         t = (1:S.N)/S.fs;
% 
%         % LFP axis
%         cla(S.ax1); hold(S.ax1,'on');
%         if S.view_global
%             plot(S.ax1, t, S.lfp, 'Color', [0.7 0.7 0.7]);
%             xs = [S.starts(i) S.ends(i)]/S.fs;
%             patch(S.ax1,[xs(1) xs(2) xs(2) xs(1)], ...
%                          [S.ylim_fix(1) S.ylim_fix(1) S.ylim_fix(2) S.ylim_fix(2)], ...
%                          [1 0.8 0.8],'EdgeColor','none','FaceAlpha',0.35);
%             xlim(S.ax1, [t(1) t(end)]);
%             ylim(S.ax1, S.ylim_fix);
%             lo = 1; hi = S.N;
%         else
%             center = round((S.starts(i)+S.ends(i))/2);
%             halfN  = round(S.win_half * S.fs);
%             lo = max(1, center - halfN);
%             hi = min(S.N, center + halfN);
% 
%             plot(S.ax1, t(lo:hi), S.lfp(lo:hi), 'Color', [0.6 0.6 0.6]);
%             ev_lo = max(S.starts(i), lo); ev_hi = min(S.ends(i), hi);
%             if ev_hi >= ev_lo
%                 idx = ev_lo:ev_hi;
%                 plot(S.ax1, t(idx), S.lfp(idx), 'r','LineWidth',1.5);
%             end
%             xlim(S.ax1, [t(lo) t(hi)]);
%             ylim(S.ax1, S.ylim_fix);
%         end
%         ylabel(S.ax1,'Amplitude'); title(S.ax1,'SWD Curation'); grid(S.ax1,'on');
% 
%         % CWT axis
%         if S.cwt_show && ishandle(S.ax0)
%             cla(S.ax0); hold(S.ax0,'on');
%             sig_win = S.lfp(lo:hi);
%             t_win   = ((lo:hi)-1)/S.fs;
% 
%             try
%                 [cfs, f] = cwt(sig_win, S.fs, 'amor', 'FrequencyLimits', S.cwt_freq_lim);
%             catch
%                 [cfs, f] = cwt(sig_win, S.fs, 'amor');
%                 fmask = f >= S.cwt_freq_lim(1) & f <= S.cwt_freq_lim(2);
%                 cfs   = cfs(fmask,:);
%                 f     = f(fmask);
%             end
% 
%             Z = 20*log10(abs(cfs) + eps);
%             pr = S.cwt_robust_prct;
%             clim = [prctile(Z(:),pr(1)), prctile(Z(:),pr(2))];
% 
%             imagesc(S.ax0, t_win, f, Z); axis(S.ax0,'xy');
%             xlim(S.ax0, [t_win(1) t_win(end)]); ylim(S.ax0, S.cwt_freq_lim);
%             ylabel(S.ax0,'Freq (Hz)'); title(S.ax0,'CWT (amor, 0.5–60 Hz)');
%             colormap(S.ax0,'jet'); caxis(S.ax0,clim);
%             hold(S.ax0,'off');
%         end
% 
%         % Raster
%         if S.haveRaster
%             cla(S.ax2); hold(S.ax2,'on');
%             if S.view_global
%                 text(0.02,0.5,'Raster hidden in Global view (press V to switch to Local)',...
%                     'Units','normalized','Parent',S.ax2);
%                 set(S.ax2,'YTick',[]); xlim(S.ax2, [t(1) t(end)]);
%                 cla(S.ax3); set(S.ax3,'Visible','off');
%             else
%                 set(S.ax3,'Visible','on');
%                 tlo = t(lo); thi = t(hi);
% 
%                 for r = 1:numel(S.row_idx)
%                     rr = S.row_idx(r);
%                     v = S.spikeCell{rr};
%                     if isempty(v), continue; end
%                     mask = (v >= tlo) & (v <= thi);
%                     if any(mask)
%                         x_ = v(mask); y_ = r * ones(sum(mask),1);
%                         plot(S.ax2, x_, y_, opt.marker, 'MarkerSize', opt.markersize, 'Color', [0.1 0.1 0.1]);
%                     end
%                 end
%                 xlim(S.ax2, [tlo thi]);
%                 ylim(S.ax2, [0 max(1,numel(S.row_idx))+1]);
%                 set(S.ax2,'YDir','normal'); grid(S.ax2,'on');
%                 ylabel(S.ax2, sprintf('Neurons (1..%d)', numel(S.row_idx)));
%                 set(S.ax2,'XTickLabel',[]);
% 
%                 % Summation
%                 cla(S.ax3); hold(S.ax3,'on');
%                 edges = tlo:S.bin_w:thi;
%                 if numel(edges) < 2
%                     edges = linspace(tlo, thi, max(2, ceil((thi-tlo)/S.bin_w)));
%                 end
%                 counts = zeros(1, numel(edges)-1);
%                 for r = 1:numel(S.row_idx)
%                     rr = S.row_idx(r);
%                     v = S.spikeCell{rr};
%                     if isempty(v), continue; end
%                     mask = (v >= tlo) & (v <= thi);
%                     if any(mask)
%                         counts = counts + histcounts(v(mask), edges);
%                     end
%                 end
%                 ctrs = (edges(1:end-1) + edges(2:end))/2;
%                 stairs(S.ax3, ctrs, counts, 'LineWidth',1.2);
%                 xlim(S.ax3, [tlo thi]);
%                 grid(S.ax3,'on');
%                 ylabel(S.ax3, 'Spikes/bin');
%                 xlabel(S.ax3, sprintf('Time (s)  |  bin=%.3g s', S.bin_w));
%             end
% 
%             if ~S.show_raster
%                 set(S.ax2,'Visible','off'); set(S.ax3,'Visible','off');
%             else
%                 set(S.ax2,'Visible','on'); if ~S.view_global, set(S.ax3,'Visible','on'); end
%             end
%         end
% 
%         % Status bar
%         dur_s = S.lens(i)/S.fs;
%         lab   = S.labels(i);
%         if isnan(lab), lab_str = 'unlabeled'; else, lab_str = num2str(lab); end
%         set(S.txt,'String',sprintf(['Event %d/%d | start=%.3fs, end=%.3fs, dur=%.3fs ',...
%             '| label=%s (0=del,1=keep,2=unsure) | keep=%d | win=%.2fs | global=%d | raster=%d/%d | bin=%.3gs | spd=%.3g Hz | play=%d | CWT=%d'],...
%             i, S.M, S.starts(i)/S.fs, S.ends(i)/S.fs, dur_s, lab_str, S.keep(i), ...
%             2*S.win_half, S.view_global, ...
%             (S.haveRaster)*numel(S.row_idx), (S.haveRaster)*nNeurons, ...
%             S.bin_w, S.play_hz, S.play, S.cwt_show));
% 
%         % link x-axes
%         axs = [S.ax1 S.ax2 S.ax3]; if S.cwt_show, axs = [S.ax0 axs]; end
%         try, linkaxes(axs,'x'); catch, end
%         set([S.ax0 S.ax1 S.ax2 S.ax3],'HitTest','off','PickableParts','none');
%     end
% 
%     function setLabel(val)
%         % 0=delete, 1=keep, 2=unsure
%         i = S.idx;
%         S.labels(i) = val;
%         S.keep(i)   = ~(val == 0);
%         drawEvent();
%         autosave('label');
%     end
% 
%     function nextEv()
%         S.idx = min(S.M, S.idx+1);
%         drawEvent();
%         autosave('nav');
%         if S.idx >= S.M && S.play
%             S.play = false; updatePlayButton(); stopTimer();
%         end
%     end
% 
%     function prevEv()
%         S.idx = max(1, S.idx-1);
%         drawEvent();
%         autosave('nav');
%     end
% 
%     function toggleKeep()
%         i = S.idx;
%         S.keep(i) = ~S.keep(i);
%         if isnan(S.labels(i)) || S.labels(i)==1 || S.labels(i)==0
%             S.labels(i) = double(S.keep(i)); % 1 or 0
%         end
%         drawEvent(); autosave('toggle');
%     end
% 
%     function widenWin(),  S.win_half = min(30, S.win_half + 0.5); drawEvent(); end
%     function narrowWin(), S.win_half = max(0.5, S.win_half - 0.5); drawEvent(); end
%     function toggleGlobal(), S.view_global = ~S.view_global; drawEvent(); end
%     function toggleRaster(), S.show_raster = ~S.show_raster; drawEvent(); end
% 
%     function toggleCWT()
%         S.cwt_show = ~S.cwt_show;
%         set(S.ax0,'Visible', tern(S.cwt_show,'on','off'));
%         drawEvent();
%     end
% 
%     function togglePlay()
%         S.play = ~S.play; updatePlayButton();
%         if S.play, startTimer(); else, stopTimer(); end
%     end
% 
%     function updatePlayButton()
%         if S.play, set(S.btnPlay,'Value',1,'String','⏸ Pause (Space)');
%         else,      set(S.btnPlay,'Value',0,'String','▶ Play (Space)'); end
%     end
% 
%     function setSpeedFromEdit()
%         str = strtrim(get(S.edtHz,'String'));
%         v = str2double(str); if isnan(v) || ~isfinite(v), v = S.play_hz; end
%         v = max(S.play_min_hz, min(S.play_max_hz, v));
%         S.play_hz = v;
%         set(S.edtHz,'String',num2str(S.play_hz,'%.3g'));
%         if S.play, startTimer(); end
%         drawEvent();
%     end
% 
%     function startTimer()
%         stopTimer();
%         dt = max(S.min_dt, 1.0 / S.play_hz);
%         S.timer = timer('ExecutionMode','fixedSpacing','Period',dt, ...
%                         'TimerFcn',@(~,~) nextEv(), ...
%                         'StartDelay',dt,'Tag','SWD_Autoplay_Hz');
%         try, start(S.timer); catch, end
%     end
% 
%     function stopTimer()
%         if ~isempty(S.timer) && isvalid(S.timer)
%             try, stop(S.timer); delete(S.timer); catch, end
%         end
%         S.timer = [];
%     end
% 
%     function autosave(reason)
%         try
%             S_out = struct();
%             S_out.session_id = S.session_id;
%             S_out.saved_at   = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
%             S_out.reason     = reason;
%             S_out.fs         = S.fs;
%             S_out.N          = S.N;
%             S_out.M          = S.M;
%             S_out.starts     = S.starts;
%             S_out.ends       = S.ends;
%             S_out.labels     = S.labels;
%             S_out.keep       = S.keep;
%             S_out.win_half   = S.win_half;
%             S_out.cwt_freq   = S.cwt_freq_lim;
%             S_out.bin_w      = S.bin_w;
% 
%             save(S.file_mat,'-struct','S_out');
% 
%             T = table((1:S.M).', S.starts(:)/S.fs, S.ends(:)/S.fs, S.labels(:), S.keep(:), ...
%                       'VariableNames',{'event_id','t_start_s','t_end_s','label','keep'});
%             writetable(T, S.file_csv);
% 
%             assignin('base','swd_event_labels', S.labels);
%             assignin('base','swd_keep_idx', S.keep);
%             assignin('base','swd_session_file_mat', S.file_mat);
%             assignin('base','swd_session_file_csv', S.file_csv);
%         catch ME
%             warning('Autosave failed: %s', ME.message);
%         end
%     end
% 
%     function onClose(~)
%         stopTimer();
%         autosave('close');
%         if ishandle(S.fig)
%             try, uiresume(S.fig); catch, end
%             delete(S.fig);
%         end
%     end
% 
%     % ------- Jump dialog & helpers -------
%     function jumpDialog()
%         prompt = {'Event # (1..M):', 'Time (s or mm:ss or hh:mm:ss):'};
%         def    = {num2str(S.idx), ''};
%         answ   = inputdlg(prompt,'Go to',1,def);
%         if isempty(answ), return; end
% 
%         ev_str = strtrim(answ{1});
%         tm_str = strtrim(answ{2});
% 
%         % Prefer time if provided
%         if ~isempty(tm_str)
%             t = parseTimeString(tm_str);
%             if ~isnan(t)
%                 k = nearestEventAtOrAroundTime(t);
%                 if ~isnan(k)
%                     S.idx = k; drawEvent(); autosave('nav'); return;
%                 end
%             end
%         end
% 
%         % Fallback to event number
%         if ~isempty(ev_str)
%             j = str2double(ev_str);
%             if ~isnan(j) && j>=1 && j<=S.M
%                 S.idx = round(j); drawEvent(); autosave('nav'); return;
%             end
%         end
%     end
% 
%     function k = nearestEventAtOrAroundTime(t_s)
%         if any(~isnan(S.starts))
%             t_start = S.starts / S.fs;
%             t_end   = S.ends   / S.fs;
%             inEv = (t_s >= t_start) & (t_s <= t_end);
%             if any(inEv)
%                 k = find(inEv,1,'first');  % inside event
%                 return;
%             end
%             centers = (S.starts + S.ends) / (2*S.fs);
%             [~,k] = min(abs(centers - t_s));
%         else
%             k = NaN;
%         end
%     end
% 
%     function t = parseTimeString(s)
%         s = strtrim(s);
%         if isempty(s), t = NaN; return; end
%         if contains(s,':')
%             tokens = regexp(s,':','split');
%             nums = cellfun(@str2double, tokens);
%             if any(isnan(nums)), t = NaN; return; end
%             switch numel(nums)
%                 case 2  % mm:ss
%                     t = nums(1)*60 + nums(2);
%                 case 3  % hh:mm:ss
%                     t = nums(1)*3600 + nums(2)*60 + nums(3);
%                 otherwise
%                     t = NaN;
%             end
%         else
%             t = str2double(s);
%             if ~isfinite(t), t = NaN; end
%         end
%     end
% 
%     function out = tern(cond, a, b)
%         if cond, out = a; else, out = b; end
%     end
% end
% 
% % ===== file-scope helpers =====
% function v = ifdef(s, f, d)
%     if isstruct(s) && isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
% end
% function tf = isvalidStruct(S)
%     tf = ~isempty(S) && isstruct(S);
% end



%%

function [unsure_ids, onset_samples, onset_times] = pick_unsure_onsets( ...
    lfp_data, fs, swd_events, labels, varargin)
% pick_unsure_onsets — 交互式给 label==2 的事件标“原点”（鼠标点一下）
% 取消 guidata/gcbf 依赖，使用嵌套函数共享状态，避免 “Object must be a figure...” 错误。
%
% 用法：
%   [unsure_ids, onset_samples, onset_times] = pick_unsure_onsets( ...
%       lfp_data, fs, swd_events, labels, ...
%       'RowsPerPage',8,'PreSec',1,'PostSec',2,'AutosaveDir','SWD_onset_autosave');
%
% 输入：
%   lfp_data   : 列向量 LFP
%   fs         : 采样率 Hz
%   swd_events : 1xM cell，每个是该事件的样本索引（递增）
%   labels     : 长度 M（NaN/0/1/2），本函数仅显示 label==2 的事件
%
% 输出：
%   unsure_ids     : 原始事件的索引（在 swd_events 中的序号）
%   onset_samples  : 对应的“原点”绝对样本索引（NaN=未标）
%   onset_times    : 对应的“原点”相对时间（秒，事件起点为 0；NaN=未标）
%
% 交互：
%   左键：打点；右键：清除；A/← 上一页；D/→ 下一页；S 保存快照；Enter 保存并退出；Esc/Q 退出

% ---------- 参数 ----------
ip = inputParser;
ip.addParameter('RowsPerPage', 8, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
ip.addParameter('PreSec', 1, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
ip.addParameter('PostSec', 2, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
ip.addParameter('AutosaveDir', 'SWD_onset_autosave', @(x)ischar(x)||isstring(x));
ip.parse(varargin{:});
rows_per_page = ip.Results.RowsPerPage;
pre_sec       = ip.Results.PreSec;
post_sec      = ip.Results.PostSec;
autosave_dir  = char(ip.Results.AutosaveDir);

% ---------- 守卫 ----------
lfp_data = lfp_data(:);
N = numel(lfp_data);
M = numel(swd_events);
if M==0
    warning('swd_events 为空。'); unsure_ids=[]; onset_samples=[]; onset_times=[]; return;
end
if numel(labels) ~= M
    error('labels 长度(%d)与 swd_events 数量(%d)不一致。', numel(labels), M);
end

% ---------- 选择 label==2 的事件 ----------
unsure_ids = find(labels==2);
K = numel(unsure_ids);
if K==0
    warning('没有 label==2 的事件。');
    onset_samples = []; onset_times = []; return;
end

% ---------- 会话与自动保存 ----------
session_id = char(datetime('now','Format','yyyyMMdd_HHmmss'));
if ~exist(autosave_dir,'dir'), mkdir(autosave_dir); end
file_mat = fullfile(autosave_dir, ['pick_unsure_onsets_' session_id '.mat']);
file_csv = fullfile(autosave_dir, ['pick_unsure_onsets_' session_id '.csv']);

% ---------- 状态结构（嵌套函数共享） ----------
S.fs            = fs;
S.lfp           = lfp_data;
S.N             = N;
S.swd_events    = swd_events;
S.unsure_ids    = unsure_ids;
S.K             = K;
S.cur_page      = 1;
S.rows_per_page = rows_per_page;
S.preS          = pre_sec;
S.postS         = post_sec;
S.onset_samples = nan(K,1);
S.onset_times   = nan(K,1);
S.file_mat      = file_mat;
S.file_csv      = file_csv;
S.session_id    = session_id;

% ---------- 构建图形（不使用 guidata/gcbf） ----------
S.fig = figure('Name',sprintf('Pick Onsets | %d unsure events', K), ...
    'Color','w','Units','normalized','Position',[0.07 0.05 0.86 0.88], ...
    'NumberTitle','off','WindowButtonDownFcn',@mouseClick, ...
    'KeyPressFcn', @keyHandler, 'CloseRequestFcn', @onClose);

% 顶部提示
S.txt = uicontrol(S.fig,'Style','text','Units','normalized','BackgroundColor','w',...
    'Position',[0.02 0.95 0.96 0.04],'HorizontalAlignment','left','FontSize',11, ...
    'String','左键：打点 | 右键：清除 | A/← 上一页 | D/→ 下一页 | S 保存 | Enter 保存并退出 | Esc/Q 退出');

% 底部按钮
S.btnPrev = uicontrol(S.fig,'Style','pushbutton','String','← 上一页 (A)', ...
    'Units','normalized','Position',[0.02 0.01 0.12 0.05],'FontSize',10, ...
    'Callback',@(~,~)prevPage);
S.btnNext = uicontrol(S.fig,'Style','pushbutton','String','下一页 (D) →', ...
    'Units','normalized','Position',[0.16 0.01 0.12 0.05],'FontSize',10, ...
    'Callback',@(~,~)nextPage);
S.btnSave = uicontrol(S.fig,'Style','pushbutton','String','保存 (S)', ...
    'Units','normalized','Position',[0.30 0.01 0.10 0.05],'FontSize',10, ...
    'Callback',@(~,~)autosave('manual'));
S.btnDone = uicontrol(S.fig,'Style','pushbutton','String','保存并退出 (Enter)', ...
    'Units','normalized','Position',[0.42 0.01 0.14 0.05],'FontSize',10, ...
    'Callback',@(~,~)closeAndSave);

% 放置多个子图（按行）
S.ax = gobjects(rows_per_page,1);
top = 0.90; bot = 0.08; vgap = 0.01;
H = (top-bot - (rows_per_page-1)*vgap)/rows_per_page;
for r = 1:rows_per_page
    y = top - r*H - (r-1)*vgap;
    S.ax(r) = axes('Parent',S.fig,'Position',[0.06 y 0.90 H], 'Box','on');
end

% 初次绘制 & 阻塞等待
redrawPage();
uiwait(S.fig);

% ====== 返回输出（从 S 读，因使用嵌套函数，S 在此仍可见）======
if ishghandle(S.fig), delete(S.fig); end
onset_samples = S.onset_samples;
onset_times   = S.onset_times;

% 抛到 base（便于直接用）
try
    assignin('base','unsure_ids', S.unsure_ids);
    assignin('base','unsure_onset_samples', S.onset_samples);
    assignin('base','unsure_onset_times', S.onset_times);
    assignin('base','unsure_session_file_mat', S.file_mat);
    assignin('base','unsure_session_file_csv', S.file_csv);
catch
end

% ================= 内部函数（全部共享 S） =================
    function redrawPage()
        K = S.K; R = S.rows_per_page;
        page = S.cur_page;
        first = (page-1)*R + 1;
        last  = min(page*R, K);

        for r = 1:R
            ax = S.ax(r); cla(ax); hold(ax,'on');
            idx = first + (r-1);
            if idx <= last
                ev_id = S.unsure_ids(idx);
                ev    = S.swd_events{ev_id};
                if isempty(ev), continue; end
                ev_start = ev(1); ev_end = ev(end);
                i0 = max(1, ev_start - round(S.preS*S.fs));
                i1 = min(S.N, ev_start + round(S.postS*S.fs));
                t  = ((i0:i1)-ev_start)/S.fs;  % 事件起点对齐 = 0 s
                sig = S.lfp(i0:i1);

                plot(ax, t, sig, 'Color', [0.3 0.3 0.3]);

                t0 = 0;
                t1 = (ev_end - ev_start)/S.fs;
                if t1>t0
                    yl = [min(sig) max(sig)];
                    patch(ax, [t0 t1 t1 t0], [yl(1) yl(1) yl(2) yl(2)], ...
                          [1 0.9 0.9], 'EdgeColor','none','FaceAlpha',0.25);
                end

                if ~isnan(S.onset_times(idx))
                    x = S.onset_times(idx);
                    yl = ylim(ax);
                    plot(ax,[x x], yl, 'r-','LineWidth',1.2);
                    plot(ax, x, interp1(t,sig,x,'linear','extrap'), 'ro','MarkerSize',5,'MarkerFaceColor','r');
                end

                title(ax, sprintf('Event #%d (unsure %d/%d)', ev_id, idx, K));
                xlabel(ax,'Time from event start (s)'); ylabel(ax,'LFP');
                xlim(ax, [-S.preS, S.postS]);
            else
                set(ax,'XTick',[],'YTick',[]);
            end
            hold(ax,'off');
        end

        try, linkaxes(S.ax,'x'); catch, end
        set(S.txt,'String',sprintf('左键：打点 | 右键：清除 | Page %d / %d | 共 %d unsure 事件 | A/← 上一页 | D/→ 下一页 | S 保存 | Enter 保存并退出 | Esc/Q 退出', ...
            S.cur_page, ceil(K/R), K));
    end

    function mouseClick(~,~)
        % 当前轴
        cp_ax = gca;
        r = find(S.ax == cp_ax, 1);
        if isempty(r), return; end

        idx = (S.cur_page-1)*S.rows_per_page + r;
        if idx < 1 || idx > S.K, return; end

        sel = get(S.fig,'SelectionType'); % 'normal'=左键, 'alt'=右键
        ax = S.ax(r);

        if strcmp(sel,'normal') % 左键：打点
            pt = get(ax,'CurrentPoint'); x = pt(1,1); % 相对起点的秒
            x = max(-S.preS, min(S.postS, x));

            ev_id = S.unsure_ids(idx);
            ev    = S.swd_events{ev_id};
            if isempty(ev), return; end
            ev_start = ev(1);
            abs_sample = ev_start + round(x * S.fs);

            S.onset_times(idx)   = x;
            S.onset_samples(idx) = abs_sample;

            redrawPage();
            autosave('click');

        elseif strcmp(sel,'alt') % 右键：清除
            S.onset_times(idx)   = NaN;
            S.onset_samples(idx) = NaN;
            redrawPage();
            autosave('clear');
        end
    end

    function prevPage, S.cur_page = max(1, S.cur_page - 1); redrawPage(); end
    function nextPage, S.cur_page = min(ceil(S.K/S.rows_per_page), S.cur_page + 1); redrawPage(); end

    function keyHandler(~,e)
        switch lower(e.Key)
            case {'leftarrow','a'}, prevPage;
            case {'rightarrow','d'}, nextPage;
            case {'s'}, autosave('manual');
            case {'return','enter'}, closeAndSave;
            case {'escape','q'}, onClose();
        end
    end

    function autosave(reason)
        try
            out = struct();
            out.session_id     = S.session_id;
            out.saved_at       = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
            out.reason         = reason;
            out.fs             = S.fs;
            out.pre_post_sec   = [S.preS S.postS];
            out.unsure_ids     = S.unsure_ids(:);
            out.onset_samples  = S.onset_samples(:);
            out.onset_times    = S.onset_times(:);

            save(S.file_mat,'-struct','out');

            T = table(S.unsure_ids(:), S.onset_samples(:), S.onset_times(:), ...
                'VariableNames', {'event_id','onset_sample','onset_time_s'});
            writetable(T, S.file_csv);

            % 抛到 base
            assignin('base','unsure_ids', S.unsure_ids(:));
            assignin('base','unsure_onset_samples', S.onset_samples(:));
            assignin('base','unsure_onset_times', S.onset_times(:));
            assignin('base','unsure_session_file_mat', S.file_mat);
            assignin('base','unsure_session_file_csv', S.file_csv);
        catch ME
            warning('Autosave failed: %s', ME.message);
        end
    end

    function closeAndSave, autosave('done'); onClose(); end

    function onClose(~,~)
        try, autosave('close'); catch, end
        if ishghandle(S.fig)
            uiresume(S.fig);
            delete(S.fig);
        end
    end
end

%%
% function [unsure_ids, onset_samples, onset_times] = pick_unsure_onsets_single( ...
%     lfp_data, fs, swd_events, labels, timepoint_array, neuron_id, win_sec, opts)
% % pick_unsure_onsets_single — Single-event viewer: manually mark "onset" for events with label==2
% % Layout: CWT (0.5–60 Hz) | LFP (clickable to set onset) | Raster | Raster Summary
% %
% % Usage:
% %   [unsure_ids, onset_samples, onset_times] = pick_unsure_onsets_single( ...
% %       lfp_clean3, fs, swd_events, labels, timepoint_array, neuron_id, 10, struct(...));
% %
% % Inputs:
% %   lfp_data        : column vector LFP (pre-cleaned)
% %   fs              : sampling rate (Hz)
% %   swd_events      : 1xM cell, each cell contains increasing sample indices of the event
% %   labels          : length-M vector (NaN/0/1/2), only iterate events with label==2
% %   timepoint_array : (Nneurons x K) matrix, seconds; each row is a neuron's spike times
% %   neuron_id       : (Nneurons x 1) neuron IDs for display (optional)
% %   win_sec         : total width (sec) of the single-event viewing window, default 10
% %   opts            : struct, optional fields:
% %       .max_neurons      default 200
% %       .bin_w            histogram bin width for raster summary (sec), default 0.05
% %       .cwt_freq_lim     CWT frequency limits, default [0.5 60]
% %       .cwt_prct         robust color limits percentiles for CWT, default [5 95]
% %       .ylim_lfp         y-limits for LFP axis, default [] (auto)
% %       .autosave_dir     autosave directory, default 'SWD_onset_autosave'
% %
% % Outputs:
% %   unsure_ids     : original indices in swd_events (only those with label==2)
% %   onset_samples  : absolute sample indices for the marked onset (NaN = not set)
% %   onset_times    : absolute times (sec) for the marked onset (NaN = not set)
% %
% % Interactions:
% %   LFP axis left-click: set onset; right-click: clear onset
% %   Keyboard: A/← previous; D/→ next; G go to; S save; Enter save & exit; Esc/Q exit
% 
%     if nargin < 7 || isempty(win_sec), win_sec = 10; end
%     if nargin < 8 || isempty(opts),    opts    = struct(); end
%     if isempty(timepoint_array), timepoint_array = []; end
%     if nargin < 6 || isempty(neuron_id), neuron_id = []; end
% 
%     % --- Guards & pre-processing ---
%     lfp_data = lfp_data(:);
%     N = numel(lfp_data);
%     M = numel(swd_events);
%     if M==0, warning('swd_events is empty.'); unsure_ids=[]; onset_samples=[]; onset_times=[]; return; end
%     if numel(labels) ~= M
%         error('labels length (%d) mismatches swd_events count (%d).', numel(labels), M);
%     end
% 
%     unsure_ids = find(labels==2);
%     K = numel(unsure_ids);
%     if K==0
%         warning('No events with label==2.');
%         onset_samples = []; onset_times = []; return;
%     end
% 
%     % --- Options ---
%     O.max_neurons   = getdef(opts,'max_neurons', 200);
%     O.bin_w         = getdef(opts,'bin_w', 0.05);
%     O.cwt_freq      = getdef(opts,'cwt_freq_lim', [0.5 60]);
%     O.cwt_prct      = getdef(opts,'cwt_prct', [5 95]);
%     O.ylim_lfp      = getdef(opts,'ylim_lfp', []);
%     O.autosave_dir  = char(getdef(opts,'autosave_dir', 'SWD_onset_autosave'));
% 
%     % --- Raster pre-processing ---
%     haveRaster = ~isempty(timepoint_array);
%     if haveRaster
%         nNeurons = size(timepoint_array,1);
%         spikeCell = cell(nNeurons,1);
%         for r = 1:nNeurons
%             v = timepoint_array(r,:);
%             v = v(~isnan(v) & v>0);
%             spikeCell{r} = v(:); % seconds
%         end
%         if isempty(neuron_id), neuron_id = (1:nNeurons).'; end
%         plot_rows = min(O.max_neurons, numel(spikeCell));
%         row_idx   = (1:plot_rows).';
%     else
%         nNeurons = 0; spikeCell = {}; row_idx = [];
%     end
% 
%     % --- State ---
%     S.fs         = fs;
%     S.lfp        = lfp_data;
%     S.N          = N;
%     S.M          = M;
%     S.win_half   = max(0.5, win_sec/2);
%     S.idx        = 1;               % pointer into unsure_ids (1..K)
%     S.unsure_ids = unsure_ids(:);
%     S.K          = K;
% 
%     S.spikeCell  = spikeCell;
%     S.row_idx    = row_idx;
%     S.haveRaster = haveRaster;
% 
%     % Onset storage (absolute samples & times)
%     S.onset_samp = nan(K,1);
%     S.onset_time = nan(K,1);
% 
%     % LFP y-limits
%     if isempty(O.ylim_lfp)
%         ypad = 0.05 * range(lfp_data); if ypad==0, ypad=1; end
%         S.ylim_lfp = [min(lfp_data)-ypad, max(lfp_data)+ypad];
%     else
%         S.ylim_lfp = O.ylim_lfp;
%     end
% 
%     % --- Session & autosave ---
%     S.session_id = char(datetime('now','Format','yyyyMMdd_HHmmss'));
%     if ~exist(O.autosave_dir,'dir'), mkdir(O.autosave_dir); end
%     S.file_mat = fullfile(O.autosave_dir, ['onset_single_' S.session_id '.mat']);
%     S.file_csv = fullfile(O.autosave_dir, ['onset_single_' S.session_id '.csv']);
% 
%     % --- Figure & axes layout (CWT | LFP | Raster | Summary) ---
%     S.fig = figure('Name',sprintf('Unsure Onset Picker | %d events (label==2)', K), ...
%         'Color','w','Units','normalized','Position',[0.08 0.08 0.84 0.78], ...
%         'NumberTitle','off','WindowButtonDownFcn',@mouseClick, ...
%         'KeyPressFcn',@keyHandler, 'CloseRequestFcn',@onClose);
% 
%     posCWT = [0.07 0.72 0.90 0.20];
%     posLFP = [0.07 0.50 0.90 0.19];
%     posRAS = [0.07 0.33 0.90 0.12];
%     posSUM = [0.07 0.24 0.90 0.07];
% 
%     S.ax0 = axes('Parent',S.fig,'Position',posCWT);  % CWT
%     S.ax1 = axes('Parent',S.fig,'Position',posLFP);  % LFP (clickable)
%     S.ax2 = axes('Parent',S.fig,'Position',posRAS);  % Raster
%     S.ax3 = axes('Parent',S.fig,'Position',posSUM);  % Summary
% 
%     % --- Controls ---
%     mk = @(str,pos,cb) uicontrol(S.fig,'Style','pushbutton','String',str, ...
%         'Units','normalized','Position',pos,'Callback',cb,'FontSize',10);
% 
%     y1=0.065; h1=0.07; w=0.11; gap=0.018; x=0.07;
%     mk('<< Previous (A)', [x y1 w h1], @(~,~) prevEv()); x = x + w + gap;
%     mk('Next (D) >>',     [x y1 w h1], @(~,~) nextEv()); x = x + w + gap;
%     mk('Clear Onset',     [x y1 0.10 h1], @(~,~) clearOnset()); x = x + 0.10 + gap;
%     mk('Go To (G)',       [x y1 0.10 h1], @(~,~) gotoEv());    x = x + 0.10 + gap;
% 
%     y2=0.005; h2=0.05;
%     mk('Save (S)',                 [0.07 y2 0.12 h2], @(~,~) autosave('manual'));
%     mk('Save & Exit (Enter)',      [0.21 y2 0.16 h2], @(~,~) closeAndSave());
% 
%     S.txt = uicontrol(S.fig,'Style','text','Units','normalized','BackgroundColor','w',...
%         'Position',[0.07 0.93 0.90 0.05],'HorizontalAlignment','left','FontSize',11, ...
%         'String','Left-click on LFP to set event onset; right-click to clear. A/← prev; D/→ next; G go-to; S save; Enter save & exit; Esc/Q exit.');
% 
%     % --- Initial draw & wait ---
%     drawEvent();
%     uiwait(S.fig);
% 
%     % --- Outputs ---
%     onset_samples = S.onset_samp;
%     onset_times   = S.onset_time;
% 
%     % assign to base for convenience
%     try
%         assignin('base','unsure_ids', S.unsure_ids);
%         assignin('base','unsure_onset_samples', S.onset_samp);
%         assignin('base','unsure_onset_times', S.onset_time);
%         assignin('base','unsure_single_file_mat', S.file_mat);
%         assignin('base','unsure_single_file_csv', S.file_csv);
%     catch
%     end
% 
%     % ================== Internals ==================
%     function drawEvent()
%         k  = S.idx;            % k-th unsure
%         id = S.unsure_ids(k);  % original event id
%         ev = swd_events{id};
%         if isempty(ev), ev = [1 1]; end
%         t_all = ((1:S.N)-1)/S.fs;  % absolute time (s)
% 
%         % Local window (centered at event midpoint)
%         center = round((ev(1)+ev(end))/2);
%         halfN  = round(S.win_half * S.fs);
%         lo = max(1, center - halfN);
%         hi = min(S.N, center + halfN);
% 
%         % ===== Top: CWT =====
%         cla(S.ax0); hold(S.ax0,'on');
%         sig_win = S.lfp(lo:hi);
%         t_win   = ((lo:hi)-1)/S.fs;
% 
%         try
%             [cfs, f] = cwt(sig_win, S.fs, 'amor', 'FrequencyLimits', O.cwt_freq);
%         catch
%             [cfs, f] = cwt(sig_win, S.fs, 'amor');
%             fmask = f>=O.cwt_freq(1) & f<=O.cwt_freq(2);
%             cfs = cfs(fmask,:); f = f(fmask);
%         end
%         Z = 20*log10(abs(cfs)+eps);
%         clim = prctile(Z(:), O.cwt_prct);
%         imagesc(S.ax0, t_win, f, Z); axis(S.ax0,'xy');
%         xlim(S.ax0, [t_win(1) t_win(end)]); ylim(S.ax0, O.cwt_freq);
%         ylabel(S.ax0,'Freq (Hz)'); title(S.ax0,'CWT (amor, 0.5–60 Hz)');
%         colormap(S.ax0,'jet'); caxis(S.ax0,clim); hold(S.ax0,'off');
% 
%         % ===== LFP (clickable onset) =====
%         cla(S.ax1); hold(S.ax1,'on');
%         plot(S.ax1, t_all(lo:hi), S.lfp(lo:hi), 'Color', [0.4 0.4 0.4]);
%         % highlight event segment
%         ev_lo = max(ev(1), lo); ev_hi = min(ev(end), hi);
%         if ev_hi >= ev_lo
%             tx = ((ev_lo:ev_hi)-1)/S.fs;
%             plot(S.ax1, tx, S.lfp(ev_lo:ev_hi), 'r','LineWidth',1.5);
%         end
%         % existing onset (vertical line + marker)
%         if ~isnan(S.onset_samp(k))
%             x = (S.onset_samp(k)-1)/S.fs;
%             yl = ylim(S.ax1);
%             plot(S.ax1, [x x], yl, 'm-','LineWidth',1.4);
%             plot(S.ax1, x, S.lfp(max(1,min(S.N,S.onset_samp(k)))), 'mo','MarkerFaceColor','m','MarkerSize',5);
%         end
%         xlim(S.ax1, [(lo-1)/S.fs, (hi-1)/S.fs]);
%         ylim(S.ax1, S.ylim_lfp);
%         ylabel(S.ax1,'LFP'); title(S.ax1,'SWD Curation (click to set onset)');
%         hold(S.ax1,'off');
% 
%         % ===== Raster =====
%         cla(S.ax2); hold(S.ax2,'on');
%         if S.haveRaster
%             tlo = (lo-1)/S.fs; thi = (hi-1)/S.fs;
%             for r = 1:numel(S.row_idx)
%                 rr = S.row_idx(r);
%                 v = S.spikeCell{rr};
%                 if isempty(v), continue; end
%                 mask = (v >= tlo) & (v <= thi);
%                 if any(mask)
%                     plot(S.ax2, v(mask), r*ones(sum(mask),1), '.', 'MarkerSize',6, 'Color',[0.1 0.1 0.1]);
%                 end
%             end
%             xlim(S.ax2, [tlo thi]); ylim(S.ax2,[0 max(1,numel(S.row_idx))+1]);
%             set(S.ax2,'YDir','normal'); grid(S.ax2,'on');
%             ylabel(S.ax2,sprintf('Neurons (1..%d)', numel(S.row_idx)));
%             set(S.ax2,'XTickLabel',[]); % hide x labels (shared with bottom axes)
%         else
%             text(0.02,0.5,'No raster input','Units','normalized','Parent',S.ax2);
%             set(S.ax2,'YTick',[]);
%         end
%         hold(S.ax2,'off');
% 
%         % ===== Raster Summary =====
%         cla(S.ax3); hold(S.ax3,'on');
%         if S.haveRaster
%             tlo = (lo-1)/S.fs; thi = (hi-1)/S.fs;
%             edges = tlo:O.bin_w:thi;
%             if numel(edges)<2
%                 edges = linspace(tlo, thi, max(2, ceil((thi-tlo)/O.bin_w)));
%             end
%             counts = zeros(1, numel(edges)-1);
%             for r = 1:numel(S.row_idx)
%                 rr = S.row_idx(r);
%                 v = S.spikeCell{rr};
%                 if isempty(v), continue; end
%                 mask = (v >= tlo) & (v <= thi);
%                 if any(mask)
%                     counts = counts + histcounts(v(mask), edges);
%                 end
%             end
%             ctrs = (edges(1:end-1)+edges(2:end))/2;
%             stairs(S.ax3, ctrs, counts, 'LineWidth',1.2);
%             xlim(S.ax3, [tlo thi]); grid(S.ax3,'on');
%             ylabel(S.ax3,'Spikes/bin'); xlabel(S.ax3, sprintf('Time (s)  |  bin=%.3g s', O.bin_w));
%         else
%             set(S.ax3,'Visible','off');
%         end
%         hold(S.ax3,'off');
% 
%         % Link X axes
%         try, linkaxes([S.ax0 S.ax1 S.ax2 S.ax3],'x'); catch, end
% 
%         % Top status text
%         s0 = (ev(1)-1)/S.fs; s1 = (ev(end)-1)/S.fs;
%         hasOn = ~isnan(S.onset_samp(k));
%         onStr = 'Not set';
%         if hasOn, onStr = sprintf('%.3fs', S.onset_time(k)); end
%         set(S.txt,'String',sprintf(['Event %d/%d (orig id=%d) | start=%.3fs end=%.3fs dur=%.3fs | onset=%s ', ...
%             '| A/← prev | D/→ next | G go-to | left-click set | right-click clear | S save | Enter save & exit | Esc/Q exit'], ...
%             k, S.K, id, s0, s1, s1-s0, onStr));
%     end
% 
%     function mouseClick(~,~)
%         % only respond when clicking on the LFP axis
%         if gca ~= S.ax1, return; end
%         sel = get(S.fig,'SelectionType');   % 'normal' left | 'alt' right
%         if strcmp(sel,'normal')
%             pt = get(S.ax1,'CurrentPoint'); x = pt(1,1);     % absolute time (s)
%             % clamp to current xlim, convert to sample index (t=(i-1)/fs)
%             xl = xlim(S.ax1);
%             x = max(xl(1), min(xl(2), x));
%             samp = round(x * S.fs) + 1;
%             samp = max(1, min(S.N, samp));
%             % record
%             S.onset_samp(S.idx) = samp;
%             S.onset_time(S.idx) = (samp-1)/S.fs;
%             drawEvent(); autosave('click');
%         elseif strcmp(sel,'alt')  % right-click clear
%             S.onset_samp(S.idx) = NaN;
%             S.onset_time(S.idx) = NaN;
%             drawEvent(); autosave('clear');
%         end
%     end
% 
%     function prevEv()
%         S.idx = max(1, S.idx-1);
%         drawEvent(); autosave('nav');
%     end
%     function nextEv()
%         S.idx = min(S.K, S.idx+1);
%         drawEvent(); autosave('nav');
%     end
%     function gotoEv()
%         answ = inputdlg({'Go to "unsure" event index (1..K):'},'Go to',1,{num2str(S.idx)});
%         if isempty(answ), return; end
%         j = str2double(answ{1});
%         if isnan(j) || j<1 || j>S.K, return; end
%         S.idx = round(j); drawEvent(); autosave('nav');
%     end
%     function clearOnset()
%         S.onset_samp(S.idx) = NaN;
%         S.onset_time(S.idx) = NaN;
%         drawEvent(); autosave('clear');
%     end
% 
%     function keyHandler(~,e)
%         switch lower(e.Key)
%             case {'leftarrow','a'}, prevEv();
%             case {'rightarrow','d'}, nextEv();
%             case {'g'}, gotoEv();
%             case {'s'}, autosave('manual');
%             case {'return','enter'}, closeAndSave();
%             case {'escape','q'}, onClose();
%         end
%     end
% 
%     function autosave(reason)
%         try
%             out = struct();
%             out.session_id   = S.session_id;
%             out.saved_at     = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
%             out.reason       = reason;
%             out.fs           = S.fs;
%             out.win_half     = S.win_half;
%             out.unsure_ids   = S.unsure_ids;
%             out.onset_sample = S.onset_samp;
%             out.onset_time   = S.onset_time;
%             save(S.file_mat,'-struct','out');
% 
%             T = table(S.unsure_ids, S.onset_samp, S.onset_time, ...
%                 'VariableNames',{'event_id','onset_sample','onset_time_s'});
%             writetable(T, S.file_csv);
% 
%             % also assign to base
%             assignin('base','unsure_ids', S.unsure_ids);
%             assignin('base','unsure_onset_samples', S.onset_samp);
%             assignin('base','unsure_onset_times', S.onset_time);
%             assignin('base','unsure_single_file_mat', S.file_mat);
%             assignin('base','unsure_single_file_csv', S.file_csv);
%         catch ME
%             warning('Autosave failed: %s', ME.message);
%         end
%     end
%     function closeAndSave(), autosave('done'); onClose(); end
%     function onClose(~,~)
%         try, autosave('close'); catch, end
%         if ishghandle(S.fig)
%             uiresume(S.fig);
%             delete(S.fig);
%         end
%     end
% end
% 
% % ---- Helper ----
% function v = getdef(s, f, d)
%     if isstruct(s) && isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
% end
% 




%%
function [unsure_ids, onset_samples, onset_times] = pick_unsure_onsets_single( ...
    lfp_data, fs, swd_events, labels, timepoint_array, neuron_id, win_sec, opts)
% pick_unsure_onsets_single — Single-event viewer: manually mark "onset" for events with label==2
% Layout: CWT (0.5–60 Hz) | LFP (clickable to set onset) | Raster | Raster Summary
%
% Usage:
%   [unsure_ids, onset_samples, onset_times] = pick_unsure_onsets_single( ...
%       lfp_clean3, fs, swd_events, labels, timepoint_array, neuron_id, 10, struct(...));
%
% Inputs:
%   lfp_data        : column vector LFP (pre-cleaned)
%   fs              : sampling rate (Hz)
%   swd_events      : 1xM cell, each cell contains increasing sample indices of the event
%   labels          : length-M vector (NaN/0/1/2), only iterate events with label==2
%   timepoint_array : (Nneurons x K) matrix, seconds; each row is a neuron's spike times
%   neuron_id       : (Nneurons x 1) neuron IDs for display (optional)
%   win_sec         : total width (sec) of the single-event viewing window, default 10
%   opts            : struct, optional fields:
%       .max_neurons      default 200
%       .bin_w            histogram bin width for raster summary (sec), default 0.05
%       .cwt_freq_lim     CWT frequency limits, default [0.5 60]
%       .cwt_prct         robust color limits percentiles for CWT, default [5 95]
%       .ylim_lfp         y-limits for LFP axis, default [] (auto)
%       .autosave_dir     autosave directory, default 'SWD_onset_autosave'
%
% Outputs (unchanged):
%   unsure_ids     : original indices in swd_events (only those with label==2)
%   onset_samples  : absolute sample indices for the marked onset (NaN = not set)
%   onset_times    : absolute times (sec) for the marked onset (NaN = not set)
%
% New feature:
%   • GUI buttons "Relabel as 0/1" to re-label current unsure event.
%   • On every save, write merged results to base workspace:
%       - unsure_relabel_ids, unsure_relabel_values
%       - merged_labels, merged_keep_idx
%
% Interactions:
%   LFP axis left-click: set onset; right-click: clear onset
%   Keyboard: A/← previous; D/→ next; G go to; S save; Enter save & exit; Esc/Q exit

    if nargin < 7 || isempty(win_sec), win_sec = 10; end
    if nargin < 8 || isempty(opts),    opts    = struct(); end
    if isempty(timepoint_array), timepoint_array = []; end
    if nargin < 6 || isempty(neuron_id), neuron_id = []; end

    % --- Guards & pre-processing ---
    lfp_data = lfp_data(:);
    N = numel(lfp_data);
    M = numel(swd_events);
    if M==0, warning('swd_events is empty.'); unsure_ids=[]; onset_samples=[]; onset_times=[]; return; end
    if numel(labels) ~= M
        error('labels length (%d) mismatches swd_events count (%d).', numel(labels), M);
    end

    unsure_ids = find(labels==2);
    K = numel(unsure_ids);
    if K==0
        warning('No events with label==2.');
        onset_samples = []; onset_times = []; return;
    end

    % --- Options ---
    O.max_neurons   = getdef(opts,'max_neurons', 200);
    O.bin_w         = getdef(opts,'bin_w', 0.05);
    O.cwt_freq      = getdef(opts,'cwt_freq_lim', [0.5 60]);
    O.cwt_prct      = getdef(opts,'cwt_prct', [5 95]);
    O.ylim_lfp      = getdef(opts,'ylim_lfp', []);
    O.autosave_dir  = char(getdef(opts,'autosave_dir', 'SWD_onset_autosave'));

    % --- Raster pre-processing ---
    haveRaster = ~isempty(timepoint_array);
    if haveRaster
        nNeurons = size(timepoint_array,1);
        spikeCell = cell(nNeurons,1);
        for r = 1:nNeurons
            v = timepoint_array(r,:);
            v = v(~isnan(v) & v>0);
            spikeCell{r} = v(:); % seconds
        end
        if isempty(neuron_id), neuron_id = (1:nNeurons).'; end
        plot_rows = min(O.max_neurons, numel(spikeCell));
        row_idx   = (1:plot_rows).';
    else
        nNeurons = 0; spikeCell = {}; row_idx = [];
    end

    % --- State ---
    S.fs         = fs;
    S.lfp        = lfp_data;
    S.N          = N;
    S.M          = M;
    S.win_half   = max(0.5, win_sec/2);
    S.idx        = 1;               % pointer into unsure_ids (1..K)
    S.unsure_ids = unsure_ids(:);
    S.K          = K;

    S.spikeCell  = spikeCell;
    S.row_idx    = row_idx;
    S.haveRaster = haveRaster;

    % Onset storage (absolute samples & times)
    S.onset_samp = nan(K,1);
    S.onset_time = nan(K,1);

    % New: relabel storage (for unsure events only)
    % NaN = not set; 0 or 1 = new label
    S.relabel_val = nan(K,1);

    % LFP y-limits
    if isempty(O.ylim_lfp)
        ypad = 0.05 * range(lfp_data); if ypad==0, ypad=1; end
        S.ylim_lfp = [min(lfp_data)-ypad, max(lfp_data)+ypad];
    else
        S.ylim_lfp = O.ylim_lfp;
    end

    % --- Session & autosave ---
    S.session_id = char(datetime('now','Format','yyyyMMdd_HHmmss'));
    if ~exist(O.autosave_dir,'dir'), mkdir(O.autosave_dir); end
    S.file_mat = fullfile(O.autosave_dir, ['onset_single_' S.session_id '.mat']);
    S.file_csv = fullfile(O.autosave_dir, ['onset_single_' S.session_id '.csv']);

    % --- Figure & axes layout (CWT | LFP | Raster | Summary) ---
    S.fig = figure('Name',sprintf('Unsure Onset Picker | %d events (label==2)', K), ...
        'Color','w','Units','normalized','Position',[0.08 0.08 0.84 0.78], ...
        'NumberTitle','off','WindowButtonDownFcn',@mouseClick, ...
        'KeyPressFcn',@keyHandler, 'CloseRequestFcn',@onClose);

    posCWT = [0.07 0.72 0.90 0.20];
    posLFP = [0.07 0.50 0.90 0.19];
    posRAS = [0.07 0.33 0.90 0.12];
    posSUM = [0.07 0.24 0.90 0.07];

    S.ax0 = axes('Parent',S.fig,'Position',posCWT);  % CWT
    S.ax1 = axes('Parent',S.fig,'Position',posLFP);  % LFP (clickable)
    S.ax2 = axes('Parent',S.fig,'Position',posRAS);  % Raster
    S.ax3 = axes('Parent',S.fig,'Position',posSUM);  % Summary

    % --- Controls ---
    mk = @(str,pos,cb) uicontrol(S.fig,'Style','pushbutton','String',str, ...
        'Units','normalized','Position',pos,'Callback',cb,'FontSize',10);

    % Row 1 (left block)
    y1=0.065; h1=0.07; w=0.11; gap=0.018; x=0.07;
    mk('<< Previous (A)', [x y1 w h1], @(~,~) prevEv()); x = x + w + gap;
    mk('Next (D) >>',     [x y1 w h1], @(~,~) nextEv()); x = x + w + gap;
    mk('Clear Onset',     [x y1 0.10 h1], @(~,~) clearOnset()); x = x + 0.10 + gap;
    mk('Go To (G)',       [x y1 0.10 h1], @(~,~) gotoEv());    x = x + 0.10 + gap;

    % Row 1 (right block) — New relabel buttons
    mk('Delete',    [x y1 0.12 h1], @(~,~) setRelabel(0)); x = x + 0.12 + gap;
    mk('Keep',    [x y1 0.12 h1], @(~,~) setRelabel(1)); x = x + 0.12 + gap;

    % Row 2
    y2=0.005; h2=0.05;
    mk('Save (S)',                 [0.07 y2 0.12 h2], @(~,~) autosave('manual'));
    mk('Save & Exit (Enter)',      [0.21 y2 0.16 h2], @(~,~) closeAndSave());

    S.txt = uicontrol(S.fig,'Style','text','Units','normalized','BackgroundColor','w',...
        'Position',[0.07 0.93 0.90 0.05],'HorizontalAlignment','left','FontSize',11, ...
        'String','Left-click on LFP to set onset; right-click to clear. A/← prev; D/→ next; G go to; S save; Enter save & exit; Esc/Q exit. Relabel unsure via buttons.');

    % --- Initial draw & wait ---
    drawEvent();
    uiwait(S.fig);

    % --- Outputs (unchanged) ---
    onset_samples = S.onset_samp;
    onset_times   = S.onset_time;

    % also assign to base for convenience (final push after UI closes)
    try
        pushToBase('done');
    catch
    end

    % ================== Internals ==================
    function drawEvent()
        k  = S.idx;            % k-th unsure
        id = S.unsure_ids(k);  % original event id
        ev = swd_events{id};
        if isempty(ev), ev = [1 1]; end
        t_all = ((1:S.N)-1)/S.fs;  % absolute time (s)

        % Local window (centered at event midpoint)
        center = round((ev(1)+ev(end))/2);
        halfN  = round(S.win_half * S.fs);
        lo = max(1, center - halfN);
        hi = min(S.N, center + halfN);

        % ===== Top: CWT =====
        cla(S.ax0); hold(S.ax0,'on');
        sig_win = S.lfp(lo:hi);
        t_win   = ((lo:hi)-1)/S.fs;

        try
            [cfs, f] = cwt(sig_win, S.fs, 'amor', 'FrequencyLimits', O.cwt_freq);
        catch
            [cfs, f] = cwt(sig_win, S.fs, 'amor');
            fmask = f>=O.cwt_freq(1) & f<=O.cwt_freq(2);
            cfs = cfs(fmask,:); f = f(fmask);
        end
        Z = 20*log10(abs(cfs)+eps);
        clim = prctile(Z(:), O.cwt_prct);
        imagesc(S.ax0, t_win, f, Z); axis(S.ax0,'xy');
        xlim(S.ax0, [t_win(1) t_win(end)]); ylim(S.ax0, O.cwt_freq);
        ylabel(S.ax0,'Freq (Hz)'); title(S.ax0,'CWT (amor, 0.5–60 Hz)');
        colormap(S.ax0,'jet'); caxis(S.ax0,clim); hold(S.ax0,'off');

        % ===== LFP (clickable onset) =====
        cla(S.ax1); hold(S.ax1,'on');
        plot(S.ax1, t_all(lo:hi), S.lfp(lo:hi), 'Color', [0.4 0.4 0.4]);
        % highlight event segment
        ev_lo = max(ev(1), lo); ev_hi = min(ev(end), hi);
        if ev_hi >= ev_lo
            tx = ((ev_lo:ev_hi)-1)/S.fs;
            plot(S.ax1, tx, S.lfp(ev_lo:ev_hi), 'r','LineWidth',1.5);
        end
        % existing onset (vertical line + marker)
        if ~isnan(S.onset_samp(k))
            x = (S.onset_samp(k)-1)/S.fs;
            yl = ylim(S.ax1);
            plot(S.ax1, [x x], yl, 'm-','LineWidth',1.4);
            plot(S.ax1, x, S.lfp(max(1,min(S.N,S.onset_samp(k)))), 'mo','MarkerFaceColor','m','MarkerSize',5);
        end
        xlim(S.ax1, [(lo-1)/S.fs, (hi-1)/S.fs]);
        ylim(S.ax1, S.ylim_lfp);
        ylabel(S.ax1,'LFP'); title(S.ax1,'SWD Curation (click to set onset)');
        hold(S.ax1,'off');

        % ===== Raster =====
        cla(S.ax2); hold(S.ax2,'on');
        if S.haveRaster
            tlo = (lo-1)/S.fs; thi = (hi-1)/S.fs;
            for r = 1:numel(S.row_idx)
                rr = S.row_idx(r);
                v = S.spikeCell{rr};
                if isempty(v), continue; end
                mask = (v >= tlo) & (v <= thi);
                if any(mask)
                    plot(S.ax2, v(mask), r*ones(sum(mask),1), '.', 'MarkerSize',6, 'Color',[0.1 0.1 0.1]);
                end
            end
            xlim(S.ax2, [tlo thi]); ylim(S.ax2,[0 max(1,numel(S.row_idx))+1]);
            set(S.ax2,'YDir','normal'); grid(S.ax2,'on');
            ylabel(S.ax2,sprintf('Neurons (1..%d)', numel(S.row_idx)));
            set(S.ax2,'XTickLabel',[]); % hide x labels (shared with bottom axes)
        else
            text(0.02,0.5,'No raster input','Units','normalized','Parent',S.ax2);
            set(S.ax2,'YTick',[]);
        end
        hold(S.ax2,'off');

        % ===== Raster Summary =====
        cla(S.ax3); hold(S.ax3,'on');
        if S.haveRaster
            tlo = (lo-1)/S.fs; thi = (hi-1)/S.fs;
            edges = tlo:O.bin_w:thi;
            if numel(edges)<2
                edges = linspace(tlo, thi, max(2, ceil((thi-tlo)/O.bin_w)));
            end
            counts = zeros(1, numel(edges)-1);
            for r = 1:numel(S.row_idx)
                rr = S.row_idx(r);
                v = S.spikeCell{rr};
                if isempty(v), continue; end
                mask = (v >= tlo) & (v <= thi);
                if any(mask)
                    counts = counts + histcounts(v(mask), edges);
                end
            end
            ctrs = (edges(1:end-1)+edges(2:end))/2;
            stairs(S.ax3, ctrs, counts, 'LineWidth',1.2);
            xlim(S.ax3, [tlo thi]); grid(S.ax3,'on');
            ylabel(S.ax3,'Spikes/bin'); xlabel(S.ax3, sprintf('Time (s)  |  bin=%.3g s', O.bin_w));
        else
            set(S.ax3,'Visible','off');
        end
        hold(S.ax3,'off');

        % Link X axes
        try, linkaxes([S.ax0 S.ax1 S.ax2 S.ax3],'x'); catch, end

        % Top status text
        s0 = (ev(1)-1)/S.fs; s1 = (ev(end)-1)/S.fs;
        hasOn = ~isnan(S.onset_samp(k));
        onStr = 'Not set';
        if hasOn, onStr = sprintf('%.3fs', S.onset_time(k)); end
        relStr = 'Relabel: N/A';
        if ~isnan(S.relabel_val(k)), relStr = sprintf('Relabel: %d', S.relabel_val(k)); end
        set(S.txt,'String',sprintf(['Event %d/%d (orig id=%d) | start=%.3fs end=%.3fs dur=%.3fs | onset=%s | %s ', ...
            '| A/← prev | D/→ next | G go-to | left-click set | right-click clear | S save | Enter save & exit | Esc/Q exit'], ...
            k, S.K, id, s0, s1, s1-s0, onStr, relStr));
    end

    function mouseClick(~,~)
        % only respond when clicking on the LFP axis
        if gca ~= S.ax1, return; end
        sel = get(S.fig,'SelectionType');   % 'normal' left | 'alt' right
        if strcmp(sel,'normal')
            pt = get(S.ax1,'CurrentPoint'); x = pt(1,1);     % absolute time (s)
            % clamp to current xlim, convert to sample index (t=(i-1)/fs)
            xl = xlim(S.ax1);
            x = max(xl(1), min(xl(2), x));
            samp = round(x * S.fs) + 1;
            samp = max(1, min(S.N, samp));
            % record
            S.onset_samp(S.idx) = samp;
            S.onset_time(S.idx) = (samp-1)/S.fs;
            drawEvent(); autosave('click');
        elseif strcmp(sel,'alt')  % right-click clear
            S.onset_samp(S.idx) = NaN;
            S.onset_time(S.idx) = NaN;
            drawEvent(); autosave('clear');
        end
    end

    function prevEv()
        S.idx = max(1, S.idx-1);
        drawEvent(); autosave('nav');
    end
    function nextEv()
        S.idx = min(S.K, S.idx+1);
        drawEvent(); autosave('nav');
    end
    function gotoEv()
        answ = inputdlg({'Go to "unsure" event index (1..K):'},'Go to',1,{num2str(S.idx)});
        if isempty(answ), return; end
        j = str2double(answ{1});
        if isnan(j) || j<1 || j>S.K, return; end
        S.idx = round(j); drawEvent(); autosave('nav');
    end
    function clearOnset()
        S.onset_samp(S.idx) = NaN;
        S.onset_time(S.idx) = NaN;
        drawEvent(); autosave('clear');
    end

    % ---- New: set relabel value (0/1) for current unsure event ----
    function setRelabel(v01)
        if ~(isequal(v01,0) || isequal(v01,1)), return; end
        S.relabel_val(S.idx) = v01;
        drawEvent(); autosave('relabel');
    end

    function keyHandler(~,e)
        switch lower(e.Key)
            case {'leftarrow','a'}, prevEv();
            case {'rightarrow','d'}, nextEv();
            case {'g'}, gotoEv();
            case {'s'}, autosave('manual');
            case {'return','enter'}, closeAndSave();
            case {'escape','q'}, onClose();
        end
    end

    function autosave(reason)
        try
            % build merged labels (apply relabels to unsure positions)
            merged_labels = labels;
            mask = ~isnan(S.relabel_val);
            if any(mask)
                merged_labels(S.unsure_ids(mask)) = S.relabel_val(mask);
            end
            merged_keep_idx = merged_labels ~= 0;

            out = struct();
            out.session_id     = S.session_id;
            out.saved_at       = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
            out.reason         = reason;
            out.fs             = S.fs;
            out.win_half       = S.win_half;
            out.unsure_ids     = S.unsure_ids;
            out.onset_sample   = S.onset_samp;
            out.onset_time     = S.onset_time;
            out.relabel_values = S.relabel_val;
            out.merged_labels  = merged_labels(:);
            out.merged_keep    = merged_keep_idx(:);
            save(S.file_mat,'-struct','out');

            T = table(S.unsure_ids, S.onset_samp, S.onset_time, S.relabel_val, ...
                'VariableNames',{'event_id','onset_sample','onset_time_s','relabel_0_1'});
            % for convenience append a second table to same CSV (sectioned by a blank line)
            writetable(T, S.file_csv);
            fid = fopen(S.file_csv,'a'); fprintf(fid,'\n'); fclose(fid);
            T2 = table((1:M).', merged_labels(:), merged_keep_idx(:), ...
                'VariableNames',{'event_id','merged_label','merged_keep'});
            writetable(T2, S.file_csv, 'WriteMode','append');

            % push to base (so你不需要改下游代码签名)
            pushToBase(reason, merged_labels, merged_keep_idx);
        catch ME
            warning('Autosave failed: %s', ME.message);
        end
    end

    function pushToBase(reason, merged_labels, merged_keep_idx)
        try
            assignin('base','unsure_ids', S.unsure_ids);
            assignin('base','unsure_onset_samples', S.onset_samp);
            assignin('base','unsure_onset_times', S.onset_time);
            assignin('base','unsure_relabel_ids', S.unsure_ids(~isnan(S.relabel_val)));
            assignin('base','unsure_relabel_values', S.relabel_val(~isnan(S.relabel_val)));
            if nargin < 2
                % if not provided, rebuild here
                merged_labels = labels;
                mask = ~isnan(S.relabel_val);
                if any(mask), merged_labels(S.unsure_ids(mask)) = S.relabel_val(mask); end
                merged_keep_idx = merged_labels ~= 0;
            end
            assignin('base','merged_labels', merged_labels);
            assignin('base','merged_keep_idx', merged_keep_idx);
            assignin('base','unsure_single_file_mat', S.file_mat);
            assignin('base','unsure_single_file_csv', S.file_csv);
            assignin('base','unsure_save_reason', reason);
        catch
        end
    end

    function closeAndSave(), autosave('done'); onClose(); end
    function onClose(~,~)
        try, autosave('close'); catch, end
        if ishghandle(S.fig)
            uiresume(S.fig);
            delete(S.fig);
        end
    end
end

% ---- Helper ----
function v = getdef(s, f, d)
    if isstruct(s) && isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end

