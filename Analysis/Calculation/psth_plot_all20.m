%% ==== 依次选择 4 个 genotype 的 PSTH Excel 文件 ====
% 顺序：WT, HOM, HOMcon, HOMkv1.1
genoDispNames = {'WT','HOM','HOMcon','HOMkv1.1'};   % 用来提示
genotypeNames = {'WT','HOM','HOMcon','HOMkv1_1'};   % 变量名版本（不能有点）
nGenos = numel(genotypeNames);

file_in = cell(nGenos,1);   % 每个 genotype 的 Excel 完整路径
fp_list = cell(nGenos,1);   % 每个文件所在的文件夹路径

for g = 1:nGenos
    promptStr = sprintf('选择 %s 的 PSTH Excel 文件', genoDispNames{g});
    [fn, fp] = uigetfile('*.xlsx', promptStr);
    if isequal(fn,0)
        error('没有选择 %s 的 Excel 文件。', genoDispNames{g});
    end
    file_in{g} = fullfile(fp, fn);
    fp_list{g} = fp;
end

%% ==== 预先读出每个 Excel 的 sheet 名，并为全局 baseline 做准备 ====
sheetNames_all = cell(nGenos,1);
nSheets_all    = zeros(nGenos,1);

allBaselineVals = [];      % 所有 genotype、所有 PSTH 的前 50 个点
maxRows_global  = 0;       % 所有 sheet 中最长的 PSTH 长度

for g = 1:nGenos
    [~, sheetNames] = xlsfinfo(file_in{g});
    sheetNames_all{g} = sheetNames;
    nSheets_all(g)    = numel(sheetNames);

    for iS = 1:nSheets_all(g)
        M = readmatrix(file_in{g}, 'Sheet', sheetNames{iS});  % 行=time bin, 列=PSTH

        if size(M,1) < 50
            error('文件 "%s" 的 sheet "%s" 行数少于 50，请检查数据。', ...
                file_in{g}, sheetNames{iS});
        end

        baseSeg = M(1:50, :);                 % 50 x nPSTH
        allBaselineVals = [allBaselineVals; baseSeg(:)]; %#ok<AGROW>

        maxRows_global = max(maxRows_global, size(M,1));
    end
end

%% ==== 计算全局 baseline_val（所有 genotype 合在一起） ====
baseline_val = mean(allBaselineVals, 'omitnan');
fprintf('全局 baseline_val = %.4f\n', baseline_val);

%% ==== 第二步：对每条 PSTH 按 “local_baseline vs baseline_val 差值” 做平移 ====
%   local_baseline_j = mean(前50点)
%   shift_j          = local_baseline_j - baseline_val
%   M_aligned(:,j)   = M(:,j) - shift_j

dt = 0.5 / 50;                 % bin 宽度（前 50 点对应 -1 ~ -0.5 s）
M_aligned_all = cell(nGenos,1);     % 每个 genotype 里再放一个 cell（每个 sheet 一个矩阵）
geno_totalPSTH = zeros(nGenos,1);   % 每个 genotype 的 PSTH 总数

for g = 1:nGenos
    thisFile  = file_in{g};
    thisFP    = fp_list{g};
    [~, name_g, ext_g] = fileparts(thisFile);
    file_out_g = fullfile(thisFP, [name_g '_perPSTHshiftAligned' ext_g]);

    sheetNames = sheetNames_all{g};
    nSheets    = nSheets_all(g);

    M_aligned_all{g} = cell(nSheets,1);

    % 画图：这个 genotype 的 PSTH（每只鼠一种颜色）
    figure; hold on;
    cmap_mouse = lines(nSheets);

    for iS = 1:nSheets
        M = readmatrix(thisFile, 'Sheet', sheetNames{iS});
        [nRows, nCols] = size(M);

        % 每条 PSTH 自己的 baseline
        local_baseline = mean(M(1:50, :), 1, 'omitnan');  % 1 x nCols
        shift          = local_baseline - baseline_val;   % 1 x nCols
        shift_mat      = repmat(shift, nRows, 1);         % nRows x nCols

        M_aligned = M - shift_mat;                        % 对齐后的 PSTH

        M_aligned_all{g}{iS} = M_aligned;
        geno_totalPSTH(g) = geno_totalPSTH(g) + nCols;

        % 写回新的 Excel 文件（可选）
        writematrix(M_aligned, file_out_g, 'Sheet', sheetNames{iS});

        % ==== 画这个 sheet（这只鼠）的 PSTH ====
        t_local   = -1 + (0:nRows-1)' * dt;
        thisColor = cmap_mouse(iS, :);

        % 细线：该鼠所有 PSTH
        for j = 1:nCols
            plot(t_local, M_aligned(:, j), ...
                'Color', thisColor * 0.6 + 0.4, 'LineWidth', 0.5);
        end

        % 粗线：该鼠的平均 PSTH
        mean_mouse = mean(M_aligned, 2, 'omitnan');
        plot(t_local, mean_mouse, 'Color', thisColor, 'LineWidth', 2, ...
            'DisplayName', sheetNames{iS});
    end

    xline(0, 'k-', 'LineWidth', 1.2);
    xlabel('Time (s)');
    ylabel(sprintf('%s PSTH (aligned)', genotypeNames{g}));
    title(sprintf('%s per-PSTH baseline 对齐后的 PSTH', genotypeNames{g}), ...
        'Interpreter', 'none');
    box off;
    set(gca, 'LineWidth', 1.5, 'FontSize', 12);
    legend('Location', 'bestoutside');
end

%% ==== 构造统一时间轴（所有 genotype 共用） ====
t = -1 + (0:maxRows_global-1)' * dt;    % [maxRows_global x 1]

%% ==== 第三步：按 genotype 分别生成 mean / SEM / n & 0–0.3 s 的 peak_xy ====
% 输出变量：
%   T_prism_WT, T_prism_HOM, T_prism_HOMcon, T_prism_HOMkv1_1
%   peak_xy_WT, peak_xy_HOM, peak_xy_HOMcon, peak_xy_HOMkv1_1

idx_win = (t >= 0) & (t <= 0.3);
t_seg   = t(idx_win);

for g = 1:nGenos
    sheetNames = sheetNames_all{g};
    nSheets    = nSheets_all(g);

    % 把这个 genotype 的所有对齐后 PSTH 拼成 [time x nPSTH]，短的用 NaN 补
    nCols_g    = geno_totalPSTH(g);
    PSTH_all_g = nan(maxRows_global, nCols_g);

    colIdx = 0;
    for iS = 1:nSheets
        M_aligned = M_aligned_all{g}{iS};
        [nRows, nCols] = size(M_aligned);

        rows_to_use = 1:nRows;
        cols_to_use = (colIdx+1):(colIdx+nCols);

        PSTH_all_g(rows_to_use, cols_to_use) = M_aligned;
        colIdx = colIdx + nCols;
    end

    % ==== genotype 级别的 mean / SEM / n ====
    mean_g  = mean(PSTH_all_g, 2, 'omitnan');              % [maxRows x 1]
    n_eff_g = sum(~isnan(PSTH_all_g), 2);                  % 每个 bin 实际参与的 PSTH 数
    sem_g   = std(PSTH_all_g, 0, 2, 'omitnan') ./ sqrt(n_eff_g);

    n_total_g = nCols_g;                                   % 这个 genotype 的 PSTH 总数
    N_col_g   = repmat(n_total_g, maxRows_global, 1);      % Prism 里方便记 sample size

    T_prism_g = table(t, mean_g, sem_g, N_col_g, ...
        'VariableNames', {'Time_s','Mean','SEM','N'});

    % ==== 0–0.3 s 区间内每条 PSTH 的 peak (x,y) ====
    nPSTH_g = size(PSTH_all_g, 2);
    peak_xy = nan(nPSTH_g, 2);      % 第1列: time, 第2列: value

    for j = 1:nPSTH_g
        y_seg = PSTH_all_g(idx_win, j);

        if all(isnan(y_seg))
            continue;
        end

        [y_max, idx_local] = max(y_seg);
        x_max = t_seg(idx_local);

        peak_xy(j,:) = [x_max, y_max];
    end

    % ==== 根据 genotype 写入对应变量名 ====
    switch g
        case 1
            T_prism_WT    = T_prism_g;
            peak_xy_WT    = peak_xy;
        case 2
            T_prism_HOM   = T_prism_g;
            peak_xy_HOM   = peak_xy;
        case 3
            T_prism_HOMcon = T_prism_g;
            peak_xy_HOMcon = peak_xy;
        case 4
            T_prism_HOMkv1_1 = T_prism_g;
            peak_xy_HOMkv1_1 = peak_xy;
    end
end

%% ==== （可选）把各 genotype 的 T_prism 存成 CSV，方便 Prism 导入 ====
for g = 1:nGenos
    thisFile = file_in{g};
    thisFP   = fp_list{g};
    [~, name_g, ~] = fileparts(thisFile);

    switch g
        case 1
            Tsave = T_prism_WT;
        case 2
            Tsave = T_prism_HOM;
        case 3
            Tsave = T_prism_HOMcon;
        case 4
            Tsave = T_prism_HOMkv1_1;
    end

    file_prism_g = fullfile(thisFP, [name_g '_T_prism.csv']);
    writetable(Tsave, file_prism_g);
    fprintf('Genotype %s 的 T_prism 已保存到：%s\n', genotypeNames{g}, file_prism_g);
end

%%
%% ==== 第四步：画每个 genotype 的总体平均 trace ＋ SEM ====
% 利用刚刚生成的 T_prism_* 变量
T_all = {T_prism_WT, T_prism_HOM, T_prism_HOMcon, T_prism_HOMkv1_1};

% 为 4 个 genotype 定义颜色（顺序：WT, HOM, HOMcon, HOMkv1.1）
colorGenos = [  0 114 189;   % WT
               217  83  25;  % HOM
               237 177  32;  % HOM+CtrlV (HOMcon)
                 0 158 115]; % HOM+Kv1.1
colorGenos = colorGenos / 255;

figure; hold on;

for g = 1:nGenos
    Tg = T_all{g};

    % 取 time / mean / sem，并确保是列向量
    t_plot   = Tg.Time_s(:);
    mean_g   = Tg.Mean(:);
    sem_g    = Tg.SEM(:);

    % 如果有 NaN 或 n=0 对应的 SEM，可以先去掉首尾全 NaN 的点（可选）
    % 这里保留全部点，NaN 会自动不画

    % ---- 画 SEM 阴影带 ----
    y_lower = mean_g - sem_g;
    y_upper = mean_g + sem_g;

    x_patch = [t_plot; flipud(t_plot)];
    y_patch = [y_lower; flipud(y_upper)];

    fill(x_patch, y_patch, colorGenos(g,:), ...
        'FaceAlpha', 0.2, ...          % 半透明
        'EdgeColor', 'none');

    % ---- 画平均 trace ----
    plot(t_plot, mean_g, ...
        'Color', colorGenos(g,:), ...
        'LineWidth', 2, ...
        'DisplayName', genoDispNames{g});
end

xline(0, 'k-', 'LineWidth', 1.2);  % SWD onset
xlabel('Time (s)');
ylabel('PSTH (aligned, spikes/s/neuron)');
title('Genotype-level mean PSTH \pm SEM (baseline aligned)');
box off;
set(gca, 'LineWidth', 1.5, 'FontSize', 12);
legend('Location', 'bestoutside');
%%    %%%%% GUI

%% ==== GUI：逐条浏览 PSTH + 手动选择 peak，并按“每只老鼠”保存 ====
% 依赖的变量（上面脚本已经有）：
% M_aligned_all{g}{iS} : 该 genotype 第 iS 个 sheet(一只鼠) 的对齐后 PSTH (time x nPSTH)
% sheetNames_all{g}{iS}: 每个 sheet 名（通常是一只鼠的 ID）
% file_in{g}, fp_list{g}, genoDispNames, genotypeNames, dt, nGenos, nSheets_all

% ---------- 预先构造：每只鼠自己的时间轴 + 0–0.3s 区间自动 peak ----------
GUI = struct();
GUI.M_aligned_all   = M_aligned_all;
GUI.sheetNames_all  = sheetNames_all;
GUI.file_in         = file_in;
GUI.fp_list         = fp_list;
GUI.genoDispNames   = genoDispNames;
GUI.genotypeNames   = genotypeNames;
GUI.nGenos          = nGenos;
GUI.nSheets_all     = nSheets_all;
GUI.dt              = dt;

GUI.t_local_all   = cell(nGenos,1);   % t_local_all{g}{iS} = [nRows x 1]
GUI.autoPeak_xy   = cell(nGenos,1);   % autoPeak_xy{g}{iS} = [nPSTH x 2]
GUI.manualPeak_xy = cell(nGenos,1);   % manualPeak_xy{g}{iS} = [nPSTH x 2] (NaN 初始)

for g = 1:nGenos
    nSheets = nSheets_all(g);
    GUI.t_local_all{g}   = cell(nSheets,1);
    GUI.autoPeak_xy{g}   = cell(nSheets,1);
    GUI.manualPeak_xy{g} = cell(nSheets,1);

    for iS = 1:nSheets
        M = M_aligned_all{g}{iS};
        [nRows, nCols] = size(M);

        % 这只鼠自己的时间轴
        t_local = -1 + (0:nRows-1)' * dt;
        GUI.t_local_all{g}{iS} = t_local;

        % 0–0.3 s 区间自动peak
        idx_win = (t_local >= 0) & (t_local <= 0.3);
        t_seg   = t_local(idx_win);

        auto_xy = nan(nCols, 2);   % 每列一个PSTH，存 [t_peak, y_peak]
        for j = 1:nCols
            y_seg = M(idx_win, j);
            if all(isnan(y_seg))
                continue;
            end
            [y_max, idx_local] = max(y_seg);
            auto_xy(j,:) = [t_seg(idx_local), y_max];
        end
        GUI.autoPeak_xy{g}{iS}   = auto_xy;
        GUI.manualPeak_xy{g}{iS} = nan(size(auto_xy));   % 手动peak，初始为 NaN
    end
end

% ---------- 初始浏览位置：第1个 genotype，第1只鼠，第1条 PSTH ----------
GUI.g  = 4;   % genotype index
GUI.iS = 1;   % sheet(老鼠) index
GUI.j  = 1;   % PSTH index（列）

% ---------- 创建 Figure 和控件 ----------
fig = figure('Name','PSTH 手动 peak 选择 GUI', ...
    'NumberTitle','off', ...
    'Units','normalized', ...
    'Position',[0.07 0.1 0.8 0.8]);

% 画 PSTH 的坐标轴
GUI.axTrace = axes('Parent',fig, ...
    'Units','normalized', ...
    'Position',[0.08 0.2 0.6 0.75]);
hold(GUI.axTrace,'on');

% 顶部信息文本：当前 genotype / 老鼠 / PSTH 索引
GUI.txtInfo = uicontrol('Parent',fig, ...
    'Style','text', ...
    'Units','normalized', ...
    'Position',[0.08 0.94 0.6 0.04], ...
    'HorizontalAlignment','left', ...
    'FontSize',11, ...
    'String','');

% 右侧显示自动 & 手动 peak 坐标
GUI.txtPeak = uicontrol('Parent',fig, ...
    'Style','text', ...
    'Units','normalized', ...
    'Position',[0.7 0.7 0.27 0.2], ...
    'HorizontalAlignment','left', ...
    'FontSize',10, ...
    'String','');

% 按钮：上一条 PSTH
uicontrol('Parent',fig, ...
    'Style','pushbutton', ...
    'Units','normalized', ...
    'Position',[0.7 0.6 0.12 0.06], ...
    'String','<< Prev PSTH', ...
    'FontSize',10, ...
    'Callback',@(src,evt) onPrevPSTH(src,evt,fig));

% 按钮：下一条 PSTH
uicontrol('Parent',fig, ...
    'Style','pushbutton', ...
    'Units','normalized', ...
    'Position',[0.85 0.6 0.12 0.06], ...
    'String','Next PSTH >>', ...
    'FontSize',10, ...
    'Callback',@(src,evt) onNextPSTH(src,evt,fig));

% 按钮：上一只鼠
uicontrol('Parent',fig, ...
    'Style','pushbutton', ...
    'Units','normalized', ...
    'Position',[0.7 0.52 0.12 0.06], ...
    'String','<< Prev Mouse', ...
    'FontSize',10, ...
    'Callback',@(src,evt) onPrevMouse(src,evt,fig));

% 按钮：下一只鼠
uicontrol('Parent',fig, ...
    'Style','pushbutton', ...
    'Units','normalized', ...
    'Position',[0.85 0.52 0.12 0.06], ...
    'String','Next Mouse >>', ...
    'FontSize',10, ...
    'Callback',@(src,evt) onNextMouse(src,evt,fig));

% 按钮：手动选择 peak（在图上点一下）
uicontrol('Parent',fig, ...
    'Style','pushbutton', ...
    'Units','normalized', ...
    'Position',[0.7 0.4 0.27 0.07], ...
    'String','在当前 PSTH 上手动点选 peak', ...
    'FontSize',11, ...
    'Callback',@(src,evt) onPickPeak(src,evt,fig));

% 按钮：保存当前“老鼠”的所有 peak (.mat + .xlsx)
uicontrol('Parent',fig, ...
    'Style','pushbutton', ...
    'Units','normalized', ...
    'Position',[0.7 0.3 0.27 0.07], ...
    'String','保存当前老鼠的 peak (MAT + Excel)', ...
    'FontSize',11, ...
    'BackgroundColor',[0.9 0.9 1], ...
    'Callback',@(src,evt) onSaveCurrentMouse(src,evt,fig));

% 按钮：关闭 GUI
uicontrol('Parent',fig, ...
    'Style','pushbutton', ...
    'Units','normalized', ...
    'Position',[0.7 0.18 0.27 0.07], ...
    'String','关闭 GUI', ...
    'FontSize',10, ...
    'Callback',@(src,evt) close(fig));

% 把 GUI 结构体存到 figure 的 appdata 中，方便回调函数使用
setappdata(fig, 'GUI', GUI);

% 初次更新绘图
updatePlot(fig);

%% ==================== 回调函数定义（放在同一文件末尾即可） ====================
function updatePlot(fig)
    GUI = getappdata(fig, 'GUI');

    g  = GUI.g;
    iS = GUI.iS;
    j  = GUI.j;

    M        = GUI.M_aligned_all{g}{iS};
    t_local  = GUI.t_local_all{g}{iS};
    auto_xy  = GUI.autoPeak_xy{g}{iS};
    manual_xy= GUI.manualPeak_xy{g}{iS};

    nCols = size(M,2);

    % 防止越界
    j = max(1, min(j, nCols));
    GUI.j = j;

    cla(GUI.axTrace); hold(GUI.axTrace,'on');

    % 画当前 PSTH
    plot(GUI.axTrace, t_local, M(:,j), 'k-', 'LineWidth', 1.2);
    xline(GUI.axTrace, 0, 'k-', 'LineWidth', 1);

    % 自动 peak (0–0.3 s)
    if ~isnan(auto_xy(j,1))
        plot(GUI.axTrace, auto_xy(j,1), auto_xy(j,2), ...
            'bo', 'MarkerFaceColor','b', 'DisplayName','Auto peak 0–0.3s');
    end

    % 手动 peak（如果已有）
    if ~isnan(manual_xy(j,1))
        plot(GUI.axTrace, manual_xy(j,1), manual_xy(j,2), ...
            'rx', 'MarkerSize',8, 'LineWidth',1.5, ...
            'DisplayName','Manual peak');
    end

    xlabel(GUI.axTrace, 'Time (s)');
    ylabel(GUI.axTrace, 'PSTH (aligned)');
    title(GUI.axTrace, sprintf('%s - %s | PSTH %d / %d', ...
        GUI.genoDispNames{g}, GUI.sheetNames_all{g}{iS}, ...
        j, nCols), 'Interpreter','none');
    box(GUI.axTrace,'off');
    set(GUI.axTrace, 'LineWidth',1.5, 'FontSize',11);

    % 更新顶部信息
    set(GUI.txtInfo, 'String', sprintf('Genotype: %s   Mouse(sheet): %s   PSTH: %d / %d', ...
        GUI.genoDispNames{g}, GUI.sheetNames_all{g}{iS}, j, nCols));

    % 更新右侧文字：显示 auto & manual peak 坐标
    if ~isnan(auto_xy(j,1))
        s1 = sprintf('Auto peak (0–0.3 s): [%.3f s, %.3f]', auto_xy(j,1), auto_xy(j,2));
    else
        s1 = 'Auto peak (0–0.3 s): NaN';
    end

    if ~isnan(manual_xy(j,1))
        s2 = sprintf('Manual peak: [%.3f s, %.3f]', manual_xy(j,1), manual_xy(j,2));
    else
        s2 = 'Manual peak: (尚未设置)';
    end
    set(GUI.txtPeak, 'String', {s1; s2});

    % 把更新后的 GUI 写回
    setappdata(fig, 'GUI', GUI);
end

function onNextPSTH(~, ~, fig)
    GUI = getappdata(fig, 'GUI');
    g  = GUI.g;
    iS = GUI.iS;

    M = GUI.M_aligned_all{g}{iS};
    nCols = size(M,2);

    if GUI.j < nCols
        GUI.j = GUI.j + 1;
    else
        % 自动跳到下一只鼠的第1条PSTH
        nSheets = GUI.nSheets_all(g);
        if iS < nSheets
            GUI.iS = iS + 1;
            GUI.j  = 1;
        else
            beep;  % 已经是最后一条
        end
    end

    setappdata(fig, 'GUI', GUI);
    updatePlot(fig);
end

function onPrevPSTH(~, ~, fig)
    GUI = getappdata(fig, 'GUI');
    g  = GUI.g;
    iS = GUI.iS;

    if GUI.j > 1
        GUI.j = GUI.j - 1;
    else
        % 回到前一只鼠的最后一条 PSTH
        if iS > 1
            GUI.iS = iS - 1;
            Mprev = GUI.M_aligned_all{g}{GUI.iS};
            GUI.j = size(Mprev,2);
        else
            beep;  % 已经是第一条
        end
    end

    setappdata(fig, 'GUI', GUI);
    updatePlot(fig);
end

function onNextMouse(~, ~, fig)
    GUI = getappdata(fig, 'GUI');
    g  = GUI.g;
    iS = GUI.iS;

    nSheets = GUI.nSheets_all(g);
    if iS < nSheets
        GUI.iS = iS + 1;
        GUI.j  = 1;
        setappdata(fig, 'GUI', GUI);
        updatePlot(fig);
    else
        beep;
    end
end

function onPrevMouse(~, ~, fig)
    GUI = getappdata(fig, 'GUI');
    g  = GUI.g;
    iS = GUI.iS;

    if iS > 1
        GUI.iS = iS - 1;
        M = GUI.M_aligned_all{g}{GUI.iS};
        GUI.j  = min(GUI.j, size(M,2));
        setappdata(fig, 'GUI', GUI);
        updatePlot(fig);
    else
        beep;
    end
end

function onPickPeak(~, ~, fig)
    GUI = getappdata(fig, 'GUI');
    g  = GUI.g;
    iS = GUI.iS;
    j  = GUI.j;

    axes(GUI.axTrace); %#ok<LAXES>
    title(GUI.axTrace, '用鼠标单击选择手动 peak（左键）', 'Color',[0.8 0 0]);

    try
        [x,y] = ginput(1);
    catch
        % 用户可能关掉 figure / ESC 之类
        return;
    end

    GUI.manualPeak_xy{g}{iS}(j,:) = [x, y];

    setappdata(fig, 'GUI', GUI);
    updatePlot(fig);
end

function onSaveCurrentMouse(~, ~, fig)
    GUI = getappdata(fig, 'GUI');
    g  = GUI.g;
    iS = GUI.iS;

    M         = GUI.M_aligned_all{g}{iS};
    t_local   = GUI.t_local_all{g}{iS};
    auto_xy   = GUI.autoPeak_xy{g}{iS};
    manual_xy = GUI.manualPeak_xy{g}{iS};

    thisFile = GUI.file_in{g};
    [~, name_g, ~] = fileparts(thisFile);
    sheetName = GUI.sheetNames_all{g}{iS};
    outFolder = GUI.fp_list{g};

    % 文件名：带 genotype 文件名 + sheet 名，防止覆盖
    matFile = fullfile(outFolder, sprintf('%s_%s_manualPeak.mat', name_g, sheetName));
    xlsFile = fullfile(outFolder, sprintf('%s_%s_manualPeak.xlsx', name_g, sheetName));

    % ===== 保存 MAT：一个 struct =====
    manualPeakStruct = struct();
    manualPeakStruct.genotype      = GUI.genotypeNames{g};
    manualPeakStruct.genotypeDisp  = GUI.genoDispNames{g};
    manualPeakStruct.mouseSheet    = sheetName;
    manualPeakStruct.t             = t_local;
    manualPeakStruct.M_PSTH        = M;
    manualPeakStruct.autoPeak_xy   = auto_xy;
    manualPeakStruct.manualPeak_xy = manual_xy;

    save(matFile, '-struct', 'manualPeakStruct');

    % ===== 保存 Excel：每条 PSTH 一行 =====
    idx = (1:size(M,2))';
    T = table(idx, ...
        auto_xy(:,1), auto_xy(:,2), ...
        manual_xy(:,1), manual_xy(:,2), ...
        'VariableNames', {'PSTH_index','AutoPeak_t','AutoPeak_val','ManualPeak_t','ManualPeak_val'});

    writetable(T, xlsFile);

    fprintf('已保存 %s (%s) 的手动 peak 到:\n  MAT:   %s\n  Excel: %s\n', ...
        sheetName, GUI.genoDispNames{g}, matFile, xlsFile);
end


