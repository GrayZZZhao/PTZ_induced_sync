function run_slope_group_scaled_ptz()
    % RUN_SLOPE_GROUP_SCALED_PTZ
    % 功能：
    % 1. 处理 PTZ 数据，筛选 Peak > 3.0。
    % 2. 对齐：最大上升斜率点。
    % 3. 【关键修改】组级缩放 (Group Scaling)：
    %    不归一化单条线，而是将该组的"平均曲线"归一化到0-1，
    %    然后将同样的缩放比例应用到该组的所有原始线条上。
    
    clc; close all;

    %% ================= 1. 参数设置 (可在下面修改) =================
    % 输入路径 (PTZ)
    ROOT_DIR = 'D:\suite2p_output\analysis\merged\resize\intensity\analysis_summary - X4000R0.5\02_dff\ptz';
    
    % 输出目录
    OUT_DIR = fullfile(ROOT_DIR, '..', '09_Group_Scaled_Slope_Aligned_PTZ');
    if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

    % --- 绘图坐标轴限制 ---
    PLOT_X_LIM    = [-1, 5];       % X轴范围 (秒)
    PLOT_Y_REAL   = [-1, 10];      % Real Value 图的 Y轴范围
    
    % Scaled 图的 Y轴范围
    % 因为平均线在 0-1，个体差异会导致数值超过1或低于0，所以范围要给大一点
    PLOT_Y_SCALED = [-2, 6];   

    % --- 信号参数 ---
    Fs = 3.26;              
    BASELINE_WIN_SEC = 30;  
    
    % --- 事件检测与筛选 ---
    PEAK_PROM_THR = 0.20;   
    MIN_DIST_SEC = 2.0;     
    FILTER_PEAK_MIN =1.5;  % 筛选阈值
    
    % --- 对齐参数 ---
    ALIGN_WIN_SEC = 5.0;    
    SLOPE_SEARCH_SEC = 3.0; 
    
    %% ================= 2. 数据处理循环 (Step 1: 收集 Raw) =================
    files = dir(fullfile(ROOT_DIR, '*_dff.xlsx'));
    if isempty(files), error('未找到文件: %s', ROOT_DIR); end
    fprintf('开始处理 PTZ 数据 (收集原始数据)... \n');

    % 临时容器 (只存 Raw)
    raw_traces = struct();
    raw_traces.WT = []; raw_traces.HOM = [];
    
    trace_info = []; 

    for i = 1:length(files)
        fname = files(i).name;
        fpath = fullfile(files(i).folder, fname);
        
        group_tag = 'UNK';
        if contains(lower(fname), 'wt'), group_tag = 'WT';
        elseif contains(lower(fname), 'hom'), group_tag = 'HOM'; end
        
        try
            data_raw = readmatrix(fpath); 
            if size(data_raw,1) < size(data_raw,2), data_raw = data_raw'; end
        catch, continue; end
        
        [nFrames, nROI] = size(data_raw);
        if nFrames < 10, continue; end
        time_vec = (0:nFrames-1) / Fs;

        % 基线校正
        win_frames = round(BASELINE_WIN_SEC * Fs);
        base_rough = movmin(data_raw, win_frames, 1);
        smooth_win = round(win_frames / 5);
        baseline_trend = movmean(base_rough, smooth_win, 1);
        data_detrended = data_raw - baseline_trend;
        mean_trace = mean(data_detrended, 2, 'omitnan');
        
        sorted_trace = sort(mean_trace, 'ascend');
        idx_10 = ceil(length(sorted_trace) * 0.10);
        if idx_10 < 1, idx_10 = 1; end
        baseline_offset = mean(sorted_trace(1:idx_10), 'omitnan');
        mean_trace_final = mean_trace - baseline_offset;
        
        % 事件检测
        min_dist_frames = round(MIN_DIST_SEC * Fs);
        [pks, locs_idx] = findpeaks(mean_trace_final, ...
            'MinPeakProminence', PEAK_PROM_THR, ...
            'MinPeakDistance', min_dist_frames);
            
        % 筛选与对齐
        win_samples = round(ALIGN_WIN_SEC * Fs);
        search_samples = round(SLOPE_SEARCH_SEC * Fs);
        count_pass = 0;
        
        for k = 1:length(pks)
            if pks(k) <= FILTER_PEAK_MIN, continue; end
            
            pk_idx = locs_idx(k);
            
            idx_search_start = max(1, pk_idx - search_samples);
            idx_search_end = pk_idx;
            segment = mean_trace_final(idx_search_start:idx_search_end);
            
            d_seg = diff(segment); 
            [~, max_slope_local_idx] = max(d_seg);
            slope_idx_global = idx_search_start + max_slope_local_idx - 1;
            
            idx_start = slope_idx_global - win_samples;
            idx_end = slope_idx_global + win_samples;
            
            if idx_start < 1 || idx_end > nFrames, continue; end
            
            wave_raw = mean_trace_final(idx_start:idx_end)'; 
            
            % 只存储 Raw，后续统一处理缩放
            if strcmp(group_tag, 'WT')
                raw_traces.WT = [raw_traces.WT; wave_raw];
            elseif strcmp(group_tag, 'HOM')
                raw_traces.HOM = [raw_traces.HOM; wave_raw];
            end
            
            info.Group = string(group_tag);
            info.FileName = string(fname);
            info.OriginalPeakTime = time_vec(pk_idx);
            info.PeakVal = pks(k);
            trace_info = [trace_info; info]; %#ok<AGROW>
            count_pass = count_pass + 1;
        end
        fprintf('[%d/%d] %s -> %d 个事件\n', i, length(files), fname, count_pass);
    end
    
    %% ================= 3. 计算组级缩放 (Step 2) =================
    fprintf('Step 2: 执行组级缩放 (Group Scaling)...\n');
    
    if isempty(raw_traces.WT) && isempty(raw_traces.HOM)
        warning('未找到数据'); return;
    end
    
    traces_final = struct();
    traces_final.WT.raw = raw_traces.WT;
    traces_final.HOM.raw = raw_traces.HOM;
    traces_final.WT.scaled = [];
    traces_final.HOM.scaled = [];
    
    % --- WT 组处理 ---
    if ~isempty(raw_traces.WT)
        % 1. 计算平均曲线
        mu_wt = mean(raw_traces.WT, 1, 'omitnan');
        
        % 2. 找到平均曲线的极值
        min_wt = min(mu_wt);
        max_wt = max(mu_wt);
        range_wt = max_wt - min_wt;
        if range_wt == 0, range_wt = 1; end
        
        % 3. 应用到该组每一条线 (整体缩放)
        % Formula: (Trace - Min_Mean) / (Max_Mean - Min_Mean)
        traces_final.WT.scaled = (raw_traces.WT - min_wt) / range_wt;
    end
    
    % --- HOM 组处理 ---
    if ~isempty(raw_traces.HOM)
        mu_hom = mean(raw_traces.HOM, 1, 'omitnan');
        min_hom = min(mu_hom);
        max_hom = max(mu_hom);
        range_hom = max_hom - min_hom;
        if range_hom == 0, range_hom = 1; end
        
        traces_final.HOM.scaled = (raw_traces.HOM - min_hom) / range_hom;
    end
    
    %% ================= 4. 绘图 =================
    n_pts = size(raw_traces.WT, 2);
    if n_pts == 0, n_pts = size(raw_traces.HOM, 2); end
    t_axis = linspace(-ALIGN_WIN_SEC, ALIGN_WIN_SEC, n_pts);
    
    fprintf('正在生成图表...\n');
    
    % --- Fig 1: Real Values ---
    fig_raw = plot_double_panel_fixed(t_axis, traces_final, 'raw', 'dF/F (Real)', ...
        'Aligned to Max Slope', PLOT_Y_REAL, PLOT_X_LIM);
    save_all_formats(fig_raw, fullfile(OUT_DIR, '01_Real_Value'));
    close(fig_raw);
    
    % --- Fig 2: Group Scaled Values (新方法) ---
    % 此时平均曲线的 Peak 必定为 1
    fig_scaled = plot_double_panel_fixed(t_axis, traces_final, 'scaled', 'Group Scaled (Mean=0-1)', ...
        'Aligned to Max Slope', PLOT_Y_SCALED, PLOT_X_LIM);
    save_all_formats(fig_scaled, fullfile(OUT_DIR, '02_Group_Scaled'));
    close(fig_scaled);
    
    %% ================= 5. 数据导出 =================
    fprintf('正在导出数据...\n');
    
    save(fullfile(OUT_DIR, 'All_Data_GroupScaled.mat'), 'traces_final', 'trace_info', 't_axis');
    
    if ~isempty(trace_info)
        writetable(struct2table(trace_info), fullfile(OUT_DIR, 'Events_Metadata.xlsx'));
    end
    
    T_mean = table();
    T_mean.Time = t_axis';
    if ~isempty(traces_final.WT.raw)
        T_mean.WT_Real_Mean = mean(traces_final.WT.raw, 1)';
        T_mean.WT_Scaled_Mean = mean(traces_final.WT.scaled, 1)'; % 它的最大值应该是1
    end
    if ~isempty(traces_final.HOM.raw)
        T_mean.HOM_Real_Mean = mean(traces_final.HOM.raw, 1)';
        T_mean.HOM_Scaled_Mean = mean(traces_final.HOM.scaled, 1)'; % 它的最大值应该是1
    end
    writetable(T_mean, fullfile(OUT_DIR, 'Group_Mean_Curves.xlsx'));
    
    fprintf('完成！结果保存在: %s\n', OUT_DIR);
end

%% ================= 绘图函数 (固定坐标轴) =================
function fig = plot_double_panel_fixed(t, traces, type, y_label, title_suffix, y_lims, x_lims)
    fig = figure('Color','w', 'Position', [100 100 1200 500]);
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'loose'); 
    
    wt_data = []; hom_data = [];
    if isfield(traces.WT, type), wt_data = traces.WT.(type); end
    if isfield(traces.HOM, type), hom_data = traces.HOM.(type); end
    
    % --- Panel 1: All Traces ---
    nexttile; hold on;
    title(['All Traces (' title_suffix ')']);
    
    if ~isempty(wt_data)
        plot(t, wt_data', 'Color', [0.2 0.4 0.8 0.15], 'LineWidth', 0.5);
    end
    if ~isempty(hom_data)
        plot(t, hom_data', 'Color', [0.8 0.2 0.2 0.15], 'LineWidth', 0.5);
    end
    
    xline(0, '--k', 'Max Slope'); yline(0, '--k');
    
    h1 = plot(nan, nan, 'Color', [0.2 0.4 0.8], 'LineWidth', 2, 'DisplayName', 'WT');
    h2 = plot(nan, nan, 'Color', [0.8 0.2 0.2], 'LineWidth', 2, 'DisplayName', 'HOM');
    legend([h1 h2], 'Location', 'northeast');
    
    xlabel('Time (s)'); ylabel(y_label);
    ylim(y_lims); xlim(x_lims); 
    grid on; box off;
    
    % --- Panel 2: Average + SEM ---
    nexttile; hold on;
    title(['Average ± SEM (' title_suffix ')']);
    
    if ~isempty(wt_data), plot_shaded_sem(t, wt_data, [0.2 0.4 0.8], 'WT'); end
    if ~isempty(hom_data), plot_shaded_sem(t, hom_data, [0.8 0.2 0.2], 'HOM'); end
    
    xline(0, '--k', 'Max Slope'); yline(0, '--k');
    
    xlabel('Time (s)'); ylabel(y_label);
    ylim(y_lims); xlim(x_lims); 
    grid on; box off;
    legend('Location', 'northeast');
end

function plot_shaded_sem(x, data_mat, col, name)
    mu = mean(data_mat, 1);
    n = size(data_mat, 1);
    sem = std(data_mat, 0, 1) / sqrt(n);
    x_fill = [x, fliplr(x)];
    y_fill = [mu+sem, fliplr(mu-sem)];
    fill(x_fill, y_fill, col, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(x, mu, 'Color', col, 'LineWidth', 2, 'DisplayName', [name ' (n=' num2str(n) ')']);
end

function save_all_formats(fig, file_base)
    savefig(fig, [file_base '.fig']);
    exportgraphics(fig, [file_base '.png'], 'Resolution', 300);
    exportgraphics(fig, [file_base '.pdf'], 'ContentType', 'vector');
end
