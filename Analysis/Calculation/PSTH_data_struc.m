% ===== 计算每只老鼠的 baseline 和 response amplitude above baseline =====
nMice = numel(mousePSTH_all);

for iM = 1:nMice
    
    % ---------- 1. 读取 peak_value ----------
    % finalPeak_xy 的第 2 列是 y 值（可能有多行，这里取最大值作为该 PSTH 的 peak）
    peak_vals  = mousePSTH_all(iM).finalPeak_xy(:,2);
    peak_value = max(peak_vals);   % response peak (spikes/s/neuron)
    
    % ---------- 2. 读取对应的 PSTH raw trace ----------
    % M_PSTH: nTimeBins x nTrace（如果只有一列则直接使用）
    psth_mat = mousePSTH_all(iM).M_PSTH;
    if isvector(psth_mat)
        psth_trace = psth_mat(:);              % 保证是列向量
    else
        % 多列时对所有列取平均，得到 population-averaged PSTH
        psth_trace = mean(psth_mat, 2, 'omitnan');
    end
    
    % 时间轴
    t = mousePSTH_all(iM).t(:);   % nTimeBins x 1
    
    % ---------- 3. 计算 baseline ----------
    % 这里用 [-0.1, 0) s 作为 baseline 窗口，可根据需要修改
    idx_base = (t >= -0.1) & (t < 0);
    baseline = mean(psth_trace(idx_base), 'omitnan');
    
    % ---------- 4. 计算 amplitude ----------
    amp = peak_value - baseline;
    
    % ---------- 5. 写回 struct ----------
    % 在 mousePSTH_all 中新增两个 field，作为“第 10 / 11 列”
    mousePSTH_all(iM).baseline = baseline;
    mousePSTH_all(iM).amp      = amp;
end



%% ===== Baseline & response amplitude per PSTH (no averaging across PSTHs) =====

nMice = numel(mousePSTH_all);

for iM = 1:nMice
    
    % 时间轴 & PSTH 矩阵
    t        = mousePSTH_all(iM).t(:);        % [nBins x 1]
    psth_mat = mousePSTH_all(iM).M_PSTH;      % [nBins x nPSTH]
    
    if isvector(psth_mat)
        psth_mat = psth_mat(:);              % [nBins x 1]
    end
    
    % 对应每个 PSTH 的 peak 值（finalPeak_xy 第 2 列）
    peak_vals = mousePSTH_all(iM).finalPeak_xy(:,2);   % [nPSTH x 1]
    
    % 安全检查：列数要和 peak 数量一致
    nPSTH = size(psth_mat, 2);
    if numel(peak_vals) ~= nPSTH
        error('Mouse %s: M_PSTH 列数 (%d) 与 finalPeak_xy 行数 (%d) 不一致。',...
            mousePSTH_all(iM).mouseSheet, nPSTH, numel(peak_vals));
    end
    
    % ===== 1. 计算每个 PSTH 的 baseline =====
    % baseline 窗口：[-0.1, 0) s，如需修改直接改这里
    idx_base = (t >= -0.1) & (t < 0);
    
    % 对 baseline window 内的行，在每一列上求 mean
    % 得到 1 x nPSTH，再转成 nPSTH x 1 的 double
    baseline_each = mean(psth_mat(idx_base, :), 1, 'omitnan');  % [1 x nPSTH]
    baseline_each = baseline_each(:);                           % [nPSTH x 1]
    
    % ===== 2. 计算每个 PSTH 的 amplitude =====
    % amp_i = peak_value_i - baseline_i
    amp_each = peak_vals(:) - baseline_each;                    % [nPSTH x 1]
    
    % ===== 3. 写回 mousePSTH_all（每只老鼠一个 n x 1 double） =====
    % 作为新的“第 10、11 列”字段
    mousePSTH_all(iM).baseline = baseline_each;   % nPSTH x 1 double
    mousePSTH_all(iM).amp      = amp_each;        % nPSTH x 1 double
end

%% ===== 从峰降到 50% 峰值所需时间（per PSTH），保存到 mousePSTH_all 第13列 =====

nMice = numel(mousePSTH_all);

for iM = 1:nMice
    
    t        = mousePSTH_all(iM).t(:);           % [nBins x 1]
    psth_mat = mousePSTH_all(iM).M_PSTH;         % [nBins x nPSTH]
    peak_xy  = mousePSTH_all(iM).finalPeak_xy;   % [nPSTH x 2]  (col1: t_peak, col2: y_peak)
    
    if isvector(psth_mat)
        psth_mat = psth_mat(:);                  % [nBins x 1]
    end
    
    nPSTH = size(psth_mat, 2);
    if size(peak_xy,1) ~= nPSTH
        error('Mouse %s: M_PSTH 列数 (%d) 与 finalPeak_xy 行数 (%d) 不一致。',...
            mousePSTH_all(iM).mouseSheet, nPSTH, size(peak_xy,1));
    end
    
    % 存每个 PSTH 的“从峰到 50%% 峰值”的时间
    t_halfDecay = nan(nPSTH,1);
    
    for p = 1:nPSTH
        peak_time = peak_xy(p,1);   % 峰的时间
        peak_val  = peak_xy(p,2);   % 峰的 y 值
        
        % 峰在 t 里的 index
        [~, idx_peak] = min(abs(t - peak_time));
        
        trace_p = psth_mat(:,p);
        
        % 50% 峰值阈值（如需考虑 baseline，可改成 baseline + 0.5*(peak-baseline)）
        target_val = 0.5 * peak_val;
        
        % 从峰之后开始找，第一次下降到 <= 50% 峰值的位置
        post_trace = trace_p(idx_peak:end);
        idx_rel = find(post_trace <= target_val, 1, 'first');
        
        if ~isempty(idx_rel)
            idx_half = idx_peak + idx_rel - 1;
            t_halfDecay(p) = t(idx_half) - t(idx_peak);   % 相对峰的时间差
        else
            t_halfDecay(p) = NaN;   % 如果没降到 50%，记 NaN
        end
    end
    
    % 写入 struct（第13列）
    mousePSTH_all(iM).t_halfDecay = t_halfDecay;   % nPSTH x 1 double
end
%%
%% ===== time-to-baseline：从峰到回到 baseline±ε 的时间 (per PSTH) =====
% 这里 ε 使用各自 PSTH 在 baseline 窗口 [-0.1,0) 内的标准差
% 结果保存在 mousePSTH_all 第14列：timeToBaseline

nMice = numel(mousePSTH_all);

for iM = 1:nMice
    
    t        = mousePSTH_all(iM).t(:);           % [nBins x 1]
    psth_mat = mousePSTH_all(iM).M_PSTH;         % [nBins x nPSTH]
    peak_xy  = mousePSTH_all(iM).finalPeak_xy;   % [nPSTH x 2]
    base_vec = mousePSTH_all(iM).baseline(:);    % [nPSTH x 1]，之前算好的 baseline
    
    if isvector(psth_mat)
        psth_mat = psth_mat(:);                  % [nBins x 1]
    end
    
    nPSTH = size(psth_mat, 2);
    if size(peak_xy,1) ~= nPSTH || numel(base_vec) ~= nPSTH
        error('Mouse %s: M_PSTH / finalPeak_xy / baseline 数量不一致。',...
            mousePSTH_all(iM).mouseSheet);
    end
    
    timeToBaseline = nan(nPSTH,1);
    
    % baseline 窗口
    idx_base_win = (t >= -0.1) & (t < 0);
    
    for p = 1:nPSTH
        trace_p   = psth_mat(:,p);
        baseline  = base_vec(p);
        
        % ε = baseline window 的 std
        base_seg  = trace_p(idx_base_win);
        eps_p     = std(base_seg, 'omitnan');
        if isnan(eps_p) || eps_p == 0
            eps_p = 1e-6;   % 防止极端情况
        end
        
        % 峰 index
        peak_time = peak_xy(p,1);
        [~, idx_peak] = min(abs(t - peak_time));
        
        % 从峰之后开始找第一次回到 baseline±ε 的点
        post_trace = trace_p(idx_peak:end);
        idx_rel = find( abs(post_trace - baseline) <= eps_p , 1, 'first');
        
        if ~isempty(idx_rel)
            idx_back = idx_peak + idx_rel - 1;
            timeToBaseline(p) = t(idx_back) - t(idx_peak);  % 相对峰的时间差
        else
            timeToBaseline(p) = NaN;  % 没有回到 baseline±ε
        end
    end
    
    % 写回 struct（第14列）
    mousePSTH_all(iM).timeToBaseline = timeToBaseline;  % [nPSTH x 1 double]
end
%
%% ===== Sustained plateau rate (0.3–0.5 s mean firing rate, per PSTH) =====
% 结果保存在 mousePSTH_all 第15列：plateauRate

nMice = numel(mousePSTH_all);

for iM = 1:nMice
    
    t        = mousePSTH_all(iM).t(:);      % [nBins x 1]
    psth_mat = mousePSTH_all(iM).M_PSTH;    % [nBins x nPSTH 或 [nBins x 1]]
    
    if isvector(psth_mat)
        psth_mat = psth_mat(:);            % [nBins x 1]
    end
    
    % 平台期时间窗口：0.3–0.5 s
    idx_plateau = (t >= 0.3) & (t <= 0.5);
    
    % 对每个 PSTH 在这个时间窗口内求平均 firing rate
    % 得到 1 x nPSTH，再转成 nPSTH x 1 double
    plateauRate = mean(psth_mat(idx_plateau, :), 1, 'omitnan');   % [1 x nPSTH]
    plateauRate = plateauRate(:);                                 % [nPSTH x 1]
    
    % 写回 struct（第15列）
    mousePSTH_all(iM).plateauRate = plateauRate;
end
%%%% ===== Sustained plateau rate (0.3–0.5 s mean firing rate, per PSTH) =====
% 结果保存在 mousePSTH_all 第15列：plateauRate

nMice = numel(mousePSTH_all);

for iM = 1:nMice
    
    t        = mousePSTH_all(iM).t(:);      % [nBins x 1]
    psth_mat = mousePSTH_all(iM).M_PSTH;    % [nBins x nPSTH 或 [nBins x 1]]
    
    if isvector(psth_mat)
        psth_mat = psth_mat(:);            % [nBins x 1]
    end
    
    % 平台期时间窗口：0.3–0.5 s
    idx_plateau = (t >= 0.3) & (t <= 0.5);
    
    % 对每个 PSTH 在这个时间窗口内求平均 firing rate
    % 得到 1 x nPSTH，再转成 nPSTH x 1 double
    plateauRate = mean(psth_mat(idx_plateau, :), 1, 'omitnan');   % [1 x nPSTH]
    plateauRate = plateauRate(:);                                 % [nPSTH x 1]
    
    % 写回 struct（第15列）
    mousePSTH_all(iM).plateauRate = plateauRate;
end

%%

%% ===== Sustained plateau rate (0.15–0.5 s mean firing rate, per PSTH) =====
% 结果保存在 mousePSTH_all 第15列：plateauRate

nMice = numel(mousePSTH_all);

for iM = 1:nMice
    
    t        = mousePSTH_all(iM).t(:);      % [nBins x 1]
    psth_mat = mousePSTH_all(iM).M_PSTH;    % [nBins x nPSTH 或 [nBins x 1]]
    
    if isvector(psth_mat)
        psth_mat = psth_mat(:);            % [nBins x 1]
    end
    
    % 平台期时间窗口：0.3–0.5 s
    idx_plateau = (t >= 0.15) & (t <= 0.5);
    
    % 对每个 PSTH 在这个时间窗口内求平均 firing rate
    % 得到 1 x nPSTH，再转成 nPSTH x 1 double
    plateauRate = mean(psth_mat(idx_plateau, :), 1, 'omitnan');   % [1 x nPSTH]
    plateauRate = plateauRate(:);                                 % [nPSTH x 1]
    
    % 写回 struct（第16列）
    mousePSTH_all(iM).plateauRate = plateauRate;
end
%%
%% ===== Sustained plateau rate (0.15–0.5 s mean firing rate, per PSTH) =====
% 用于体现 HOMcon 那条“拖得比较长”的尾巴
% 结果保存在 mousePSTH_all 第16列：plateauRate_long

nMice = numel(mousePSTH_all);

for iM = 1:nMice
    
    t        = mousePSTH_all(iM).t(:);      % [nBins x 1]
    psth_mat = mousePSTH_all(iM).M_PSTH;    % [nBins x nPSTH 或 [nBins x 1]]
    
    if isvector(psth_mat)
        psth_mat = psth_mat(:);            % [nBins x 1]
    end
    
    % 平台期时间窗口：0.15–0.5 s
    idx_plateau_long = (t >= 0.15) & (t <= 0.5);
    
    % 对每个 PSTH 在这个时间窗口内求平均 firing rate
    % 得到 1 x nPSTH，再转成 nPSTH x 1 double
    plateauRate_long = mean(psth_mat(idx_plateau_long, :), 1, 'omitnan');  % [1 x nPSTH]
    plateauRate_long = plateauRate_long(:);                                % [nPSTH x 1]
    
    % 写回 struct（第16列）
    mousePSTH_all(iM).plateauRate_long = plateauRate_long;
end


%% ===== Area under the curve (AUC, 0–0.3 s, per PSTH) =====
% 结果保存在 mousePSTH_all 第17列：AUC_0_0p3

nMice = numel(mousePSTH_all);

for iM = 1:nMice
    
    t        = mousePSTH_all(iM).t(:);      % [nBins x 1]
    psth_mat = mousePSTH_all(iM).M_PSTH;    % [nBins x nPSTH 或 nBins x 1]
    
    if isvector(psth_mat)
        psth_mat = psth_mat(:);            % [nBins x 1]
    end
    
    % AUC 窗口：0–0.3 s
    idx_auc = (t >= 0) & (t <= 0.3);
    
    % 估计时间步长 dt（假设 t 近似等间隔）
    t_win = t(idx_auc);
    if numel(t_win) < 2
        error('Mouse %s: 0–0.3 s 窗口内的时间点太少，无法计算 AUC。',...
            mousePSTH_all(iM).mouseSheet);
    end
    dt = median(diff(t_win));   % s
    
    % 对每个 PSTH 在该窗口内做离散积分：sum(rate * dt)
    auc_0_0p3 = sum(psth_mat(idx_auc, :), 1, 'omitnan') * dt;   % [1 x nPSTH]
    auc_0_0p3 = auc_0_0p3(:);                                  % [nPSTH x 1]
    
    % 写回 struct（第17列）
    mousePSTH_all(iM).AUC_0_0p3 = auc_0_0p3;
end
%%
%% ===== Area under the curve (AUC) for Early (0–0.15 s) & Late (0.15–0.5 s) windows =====
% 结果保存在 mousePSTH_all：
%   第18列：AUC_early_0_0p15
%   第19列：AUC_late_0p15_0p5

nMice = numel(mousePSTH_all);

for iM = 1:nMice
    
    t        = mousePSTH_all(iM).t(:);      % [nBins x 1]
    psth_mat = mousePSTH_all(iM).M_PSTH;    % [nBins x nPSTH 或 nBins x 1]
    
    if isvector(psth_mat)
        psth_mat = psth_mat(:);            % [nBins x 1]
    end
    
    % ===== 时间窗口 =====
    idx_early = (t >= 0)    & (t <= 0.15);   % Early: 0–0.15 s
    idx_late  = (t >  0.15) & (t <= 0.5);    % Late : 0.15–0.5 s
    
    % 检查时间点是否足够
    if sum(idx_early) < 2 || sum(idx_late) < 2
        error('Mouse %s: early 或 late 窗口内的时间点太少，无法计算 AUC。',...
            mousePSTH_all(iM).mouseSheet);
    end
    
    % 估计 dt（假设 time bin 近似等间隔）
    dt_early = median(diff(t(idx_early)));
    dt_late  = median(diff(t(idx_late)));
    
    % ===== 每个 PSTH 的 Early AUC: 0–0.15 s =====
    AUC_early = sum(psth_mat(idx_early, :), 1, 'omitnan') * dt_early;   % [1 x nPSTH]
    AUC_early = AUC_early(:);                                          % [nPSTH x 1]
    
    % ===== 每个 PSTH 的 Late AUC: 0.15–0.5 s =====
    AUC_late = sum(psth_mat(idx_late, :), 1, 'omitnan') * dt_late;     % [1 x nPSTH]
    AUC_late = AUC_late(:);                                            % [nPSTH x 1]
    
    % ===== 写回 struct（第18、19列） =====
    mousePSTH_all(iM).AUC_early_0_0p15   = AUC_early;  % 第18列
    mousePSTH_all(iM).AUC_late_0p15_0p5  = AUC_late;   % 第19列
end

%% ===== Baseline firing rate per PSTH，保存到第20列 =====
% 窗口：[-0.1, 0) s，结果为每个 PSTH 一个 baseline firing rate

nMice = numel(mousePSTH_all);

for iM = 1:nMice
    
    t        = mousePSTH_all(iM).t(:);      % [nBins x 1]
    psth_mat = mousePSTH_all(iM).M_PSTH;    % [nBins x nPSTH 或 nBins x 1]
    
    if isvector(psth_mat)
        psth_mat = psth_mat(:);            % [nBins x 1]
    end
    
    % baseline 时间窗口：[-0.1, 0) s
    idx_base = (t >= -0.1) & (t < 0);
    
    % 对每个 PSTH 在 baseline 窗口内求平均 firing rate
    % 得到 1 x nPSTH，再转成 nPSTH x 1 double
    baselineRate = mean(psth_mat(idx_base, :), 1, 'omitnan');   % [1 x nPSTH]
    baselineRate = baselineRate(:);                             % [nPSTH x 1]
    
    % 写回 struct（第20列）
    mousePSTH_all(iM).baselineRate = baselineRate;
end

%%
%% ===== 每只老鼠的 baseline firing rate 均值，保存到第21列 =====
% 假设第20列已经存了每个 PSTH 的 baselineRate（nPSTH x 1 double）

nMice = numel(mousePSTH_all);

for iM = 1:nMice
    
    % 如果已经有 per-PSTH 的 baselineRate，直接用
    if isfield(mousePSTH_all, "baselineRate") && ~isempty(mousePSTH_all(iM).baselineRate)
        br_vec = mousePSTH_all(iM).baselineRate(:);   % nPSTH x 1
    else
        % 兜底：如果还没算过 baselineRate，就临时算一遍（窗口 [-0.1, 0) s）
        t        = mousePSTH_all(iM).t(:);
        psth_mat = mousePSTH_all(iM).M_PSTH;
        if isvector(psth_mat)
            psth_mat = psth_mat(:);
        end
        idx_base = (t >= -0.1) & (t < 0);
        br_vec   = mean(psth_mat(idx_base,:), 1, 'omitnan').';  % nPSTH x 1
    end
    
    % 每只老鼠的 baseline firing rate 均值
    baselineRate_mean = mean(br_vec, 'omitnan');   % 标量
    
    % 写回 struct（第21列）
    mousePSTH_all(iM).baselineRate_mean = baselineRate_mean;
end

%% ===== Baseline firing rate per PSTH（baseline: -1 ~ 0 s），保存到第21列 =====
% 结果为每个 PSTH 一个 baseline firing rate（nPSTH x 1 double）

nMice = numel(mousePSTH_all);

for iM = 1:nMice
    
    t        = mousePSTH_all(iM).t(:);      % [nBins x 1]
    psth_mat = mousePSTH_all(iM).M_PSTH;    % [nBins x nPSTH 或 nBins x 1]
    
    if isvector(psth_mat)
        psth_mat = psth_mat(:);            % [nBins x 1]
    end
    
    % baseline 时间窗口：[-1, 0) s
    idx_base = (t >= -1) & (t < 0);
    
    % 对每个 PSTH 在 baseline 窗口内求平均 firing rate
    % 得到 1 x nPSTH，再转成 nPSTH x 1 double
    baselineRate_m1_0 = mean(psth_mat(idx_base, :), 1, 'omitnan');   % [1 x nPSTH]
    baselineRate_m1_0 = baselineRate_m1_0(:);                        % [nPSTH x 1]
    
    % 写回 struct（第21列）
    mousePSTH_all(iM).baselineRate_m1_0 = baselineRate_m1_0;
end

