%% cusimized for PTZ IID recording : detection on mean(d) instead of frequncy specific swd 

loading_lfp

%%
d_shift = [d(:, 11747:end), zeros(size(d,1), 11747)];
%%
d = d_shift;
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
swd_events_30000Hz = cell(size(swd_events))% Replace this with your actual variable

% Pop_ulate each component of the cell array
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
% Load WT Spike Times and Templates AND getting spiketime array 
spike_time_wt = readNPY('D:\2025PTZ\2025-05-01_1019_HOM-female-adult\2025-05-01_13-03-35\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\spike_times.npy');
spike_time_full_wt = double(spike_time_wt) / 30000;
spike_templates_wt = readNPY('D:\2025PTZ\2025-05-01_1019_HOM-female-adult\2025-05-01_13-03-35\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\spike_templates.npy');
%spike_templates_wt = readNPY('/Volumes/GZ_NPXL_25/seizure/2024-01-02_WT_HOM-male-adult/2024-01-02_13-43-59/Record Node 101/experiment1/recording2/continuous/Neuropix-PXI-100.ProbeA-AP/spike_templates.npy');
spike_templates_full_wt = double(spike_templates_wt) + 1;
spike_channel_wt = [spike_time_full_wt, spike_templates_full_wt];

% 
[~, sortIdx_wt] = sort(spike_channel_wt(:, 2));
sortedArray2D_wt = spike_channel_wt(sortIdx_wt, :);


% Group Elements by Unique Values (WT and HOM)
uniqueValues_wt = unique(sortedArray2D_wt(:, 2));
groupedRows_wt = accumarray(sortedArray2D_wt(:, 2), sortedArray2D_wt(:, 1), [], @(x) {x'});

% Initialize Result Arrays with NaNs
maxGroupSize_wt = max(cellfun(@length, groupedRows_wt));
resultArray_wt = nan(length(uniqueValues_wt), maxGroupSize_wt + 1);
for i = 1:length(uniqueValues_wt)
    resultArray_wt(i, 1) = uniqueValues_wt(i);
    resultArray_wt(i, 2:length(groupedRows_wt{i}) + 1) = groupedRows_wt{i};
end

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

swd_events = swd_events_QualityM; 

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
% Load variables (ensure logical_array and resultArray_wt are in your workspace)
% logical_array: 1x90793140 logical array (30000 Hz)
% resultArray_wt: 337x112319 double array (spike times in seconds)

% Parameters262
sampling_rate = 30000; % Sampling rate in Hz
min_burst_duration = 0.01; % Minimum burst duration in seconds %%%%%%%%%%%%%%%
min_burst_samples = min_burst_duration * sampling_rate; % Minimum burst duration in samples
bin_size = 0.1; % Bin size in seconds (0.1 s)
bin_samples = bin_size * sampling_rate;

% Identify burst events lasting longer than the threshold
burst_diff = diff([0, logic_array, 0]); % Add 0 at boundaries to detect edges
burst_starts = find(burst_diff == 1);
burst_ends = find(burst_diff == -1) - 1;
burst_durations = burst_ends - burst_starts + 1;

% Filter bursts lasting at least 0.5 seconds
valid_bursts = find(burst_durations >= min_burst_samples);
burst_starts = burst_starts(valid_bursts);
burst_ends = burst_ends(valid_bursts);

% Preallocate arrays for storing spike rates
num_neurons = size(resultArray_wt, 1);
burst_spike_rate = zeros(num_neurons, 1);
non_burst_spike_rate = zeros(num_neurons, 1);

% Analyze spikes during burst and non-burst periods
total_burst_time = sum((burst_ends - burst_starts + 1)) / sampling_rate; % Total burst time in seconds
total_non_burst_time = length(logic_array) / sampling_rate - total_burst_time; % Total non-burst time

for neuron = 1:num_neurons
    spike_times = resultArray_wt(neuron, :);
    spike_times = spike_times(spike_times > 0); % Remove zero padding if any
    spike_samples = round(spike_times * sampling_rate); % Convert spike times to sample indices

    % Count spikes during burst events
    burst_spike_count = 0;
    for b = 1:length(burst_starts)
        burst_range = burst_starts(b):burst_ends(b);
        burst_spike_count = burst_spike_count + sum(ismember(spike_samples, burst_range));
    end

    % Count spikes during non-burst events
    all_burst_ranges = cell2mat(arrayfun(@(s, e) s:e, burst_starts, burst_ends, 'UniformOutput', false));
    non_burst_spike_count = sum(~ismember(spike_samples, all_burst_ranges));

    % Calculate spike rates (spikes per 0.1 seconds)
    burst_spike_rate(neuron) = burst_spike_count / total_burst_time;
    non_burst_spike_rate(neuron) = non_burst_spike_count / total_non_burst_time;
end

% Compute burst-to-non-burst spike rate ratio
burst_to_non_burst_rate_ratio = burst_spike_rate ./ (non_burst_spike_rate + 1e-6); % Add small value to avoid division by zero
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Select neurons with higher burst spike rate
selected_neurons = find(burst_to_non_burst_rate_ratio >1);  %(<0.5 meaning off,    >2 meaing 0n )

% Determine the overall x-axis range
x_start = 0; % Starting time in seconds
x_end = max(resultArray_wt(:)); % Ending time based on the spike data
% Visualization
figure;
% Define the positions of the subplots
subplot_positions = [0.1, 0.85, 0.8, 0.1;  % Position for first subplot
                     0.1, 0.7, 0.8, 0.1;  % Position for second subplot
                     0.1, 0.55, 0.8, 0.1; % Position for third subplot
                     0.1, 0.4, 0.8, 0.1;  % Position for fourth subplot
                     0.1, 0.25, 0.8, 0.13; % Position for fifth subplot
                     0.1, 0.1, 0.8, 0.07]; % Position for sixth subplot

ax = zeros(6,1);

% --- Plot in the first subplot (CWT 0–60 Hz, log-power + per-frequency z-score) ---
ax(1) = subplot('Position', subplot_positions(1,:)); hold on

% 采样率
fs = 2500;
if exist('original_sampling_rate','var') && ~isempty(original_sampling_rate)
    fs = original_sampling_rate;
end

% 把信号换成 µV（若 d 是原始单位；按你前面用过的 0.195）
scale_factor = 0.195;
x = mean(d(: , :), 1) * scale_factor;   % 简单地对通道取均值；按你需要调整
x = x(:);

% 预处理：带通 + 去趋势（避免直流/慢漂移把热度抹平）
bp = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',1,'HalfPowerFrequency2',60, ...
    'SampleRate',fs);
x = filtfilt(bp, x);
x = detrend(x, 'linear');

% 只计算你的可视时间窗
ix0 = max(1, floor(x_start*fs)+1);
ix1 = min(numel(x), ceil(x_end*fs));
sig_win = x(ix0:ix1);
t_win   = ((ix0:ix1)-1)/fs;

% 建 CWT 滤波器组（Morse，小波参数可调以增强集中度）
fb = cwtfilterbank('SignalLength',numel(sig_win), 'SamplingFrequency',fs, ...
    'Wavelet','morse','VoicesPerOctave',24,'TimeBandwidth',60, ...
    'FrequencyLimits',[0.5 60]);

[cfs, f] = wt(fb, sig_win);

% 功率 -> log，再做“逐频 z-score”（让振荡增强更凸显）
P = abs(cfs).^2;                    % µV^2
Zlog = log10(P + eps);              % log(power)
% 逐频标准化：每一行（某个频率）减去该行均值/除以该行std
mu = mean(Zlog, 2);
sd = std(Zlog, 0, 2) + eps;
Zz = (Zlog - mu) ./ sd;             % 每个频率 z-score

% 稳健色阶：例如 [-2, +4] 或按百分位裁剪
clim = [-2, 4]; % 也可：clim = prctile(Zz(:), [5 95]);

% 画图
imagesc(t_win, f, Zz); axis xy
xlim([x_start, x_end]); ylim([0.5 60])
xlabel('Time (s)'); ylabel('Frequency (Hz)')
title('CWT (Morse, 0–60 Hz) — per-freq z-score of log(power)')
colormap(jet); caxis(clim)
cb = colorbar; ylabel(cb, 'z (per-freq log power)')

hold off


%Plot in the second subplot (Filtered LFP signal)
ax(2)=subplot('Position', subplot_positions(2, :));
    hold on
%function plot_filtered_lfp(lfp_data, fs)
    % Bandpass Filter (5 to 60 Hz)
    low_cutoff = 5;   % Set a small non-zero value for the lower cutoff
    high_cutoff = 60;
    [b, a] = butter(4, [low_cutoff high_cutoff] / (2500 / 2), 'bandpass');
    filtered_5_60_lfp = filtfilt(b, a, mean(d));

    % Apply the scale factor to convert to microvolts (µV)
    scale_factor = 0.195;  % Conversion scale factor
    filtered_lfp_microV = filtered_5_60_lfp * scale_factor;

    % Create a time vector
    time_vector = (1:length(filtered_5_60_lfp)) / 2500;

    % Plot the filtered LFP signal in microvolts
  
    plot(time_vector, filtered_lfp_microV, 'b');
    xlabel('Time (s)');
    ylabel('Amplitude (µV)');
    title('Filtered LFP Signal (5-60 Hz) in µV');
    ylim([-2500 2500]);
    xlim([x_start, x_end])
    hold off
    %grid on;
%end



% Plot in the third subplot (IID detection)
ax(3)=subplot('Position', subplot_positions(3, :));
time_in_seconds = (1:length(mean(d))) / original_sampling_rate;
    
plot(time_in_seconds, lfp_clean3, 'Color', [0.6 0.6 0.6]); % Plot the LFP signal in a lighter color
hold on;
    dmean = mean(d);
    for i = 1:length(swd_events)
        event = swd_events{i};
        event_time = event / original_sampling_rate; % Convert event indices to time in seconds
        plot(event_time, dmean(event), 'r', 'LineWidth', 2); % Plot SWD events in red with thicker lines
    end
    
    xlabel('Time (seconds)'); % Update x-axis label
    ylabel('Amplitude');
    title('SWD Detection in LFP Signal');
    xlim([x_start, x_end])
    ylim([-20000 20000]); % Set y-axis limits
    hold off;


% Plot in the fourth subplot (Burst detection)
ax(4)=subplot('Position', subplot_positions(4, :));
hold on;
for b = 1:length(burst_starts)
    start_time = burst_starts(b) / sampling_rate;
    end_time = burst_ends(b) / sampling_rate;
    fill([start_time, end_time, end_time, start_time], [0, 0, 1, 1], 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
end
title('Burst Detection (1 s or more)');
ylabel('Burst');
ylim([0, 1]);
xlim([x_start, x_end]); % Set x-axis limits
hold off;

% Plot in the fifth subplot (Raster plot)
ax(5)=subplot('Position', subplot_positions(5, :));
hold on;
for i = 1:length(selected_neurons)
    neuron = selected_neurons(i);
    spike_times = resultArray_wt(neuron, :);
    spike_times = spike_times(spike_times > 0); % Remove zero padding
    scatter(spike_times, ones(size(spike_times)) * neuron, 10, '.');
end
xlabel('Time (s)');
ylabel('Neuron Index');
title('Raster Plot for Selected Neurons');
xlim([x_start, x_end]); % Set x-axis limits
hold off;

% Plot in the sixth subplot (Binned spike rates)
ax(6)=subplot('Position', subplot_positions(6, :));
bin_width = 0.1; % 0.1 second bins
time_bins = x_start:bin_width:x_end;

% Collect spike times of selected neurons
selected_spike_times = [];
for i = 1:length(selected_neurons)
    neuron = selected_neurons(i);
    spike_times = resultArray_wt(neuron, :);
    spike_times = spike_times(spike_times > 0); % Remove zero padding
    selected_spike_times = [selected_spike_times, spike_times]; % Append spike times
end

% Calculate histogram based on selected neurons
binned_spikes = histcounts(selected_spike_times, time_bins) / bin_width;

% Plot histogram
bar(time_bins(1:end-1), binned_spikes, 'k');
xlabel('Time (s)');
ylabel('Spike Rate (spikes/s)');
title('Binned Spike Rates for Selected Neurons');
xlim([x_start, x_end]); % Set x-axis limits

linkaxes(ax, 'x');

% Adjust the figure window size as needed
set(gcf, 'Position', [100, 100, 800, 900]);  % Example figure size

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

