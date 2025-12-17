function analyze_complete_pipeline_v2_better_viz()
% ANALYZE_COMPLETE_PIPELINE_V2_BETTER_VIZ
% 改进版：
% 1) Moving Bottom baseline (movmin + smooth)
% 2) Z-score heatmap
% 3) 固定 caxis
%
% [当前绘图窗口要求]
% baseline WT/HOM: 0-500 s
% ptz WT: 600-1100 s
% ptz HOM: 2100-2600 s
%
% [当前新增要求]
% - Mean dF/F 子图 y 轴固定为 [0 3]
% - 输出为矢量 PDF
% - 同时保存 .fig
% - 不关闭图窗（保留在 MATLAB 界面）
% - 显示 colorbar

    clc; close all;

    %% ================= CONFIGURATION =================
    ROOT_DIR = '/Users/grayzzzzzz/Desktop/19_Numbered_Charts_Docs/Fig_pannel/06050223_mapTrace';
    OUT_DIR  = fullfile(ROOT_DIR, '15_Improved_Viz_Analysis');
    if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

    MAP_DIR = fullfile(OUT_DIR, 'Improved_Map');
    if ~exist(MAP_DIR, 'dir'), mkdir(MAP_DIR); end

    Fs = 3.26;
    BASELINE_WIN_SEC = 30;

    % 事件检测参数
    MIN_PEAK_PROM = 0.20;
    MIN_DIST_SEC  = 1.0;
    HIGH_R_THR    = 0.5;

    DRAW_PLOTS = true;

    %% ================= MAIN PROCESS =================
    files_base = dir(fullfile(ROOT_DIR, 'baseline', '**', '*dff*.xlsx'));
    files_ptz  = dir(fullfile(ROOT_DIR, 'ptz',      '**', '*dff*.xlsx'));
    all_files  = [files_base; files_ptz];

    fprintf('Found baseline files: %d | ptz files: %d\n', numel(files_base), numel(files_ptz));
    if isempty(all_files), error('未找到文件'); end

    event_records = [];
    fprintf('开始分析...\n');

    for i = 1:length(all_files)
        fname = all_files(i).name;
        fpath = fullfile(all_files(i).folder, fname);

        meta = parse_filename(fname);

        % ===== Phase 强制从文件夹推断（避免 baseline+ptz 命名误判）=====
        folderLower = lower(all_files(i).folder);
        if contains(folderLower, [filesep 'baseline']) || endsWith(folderLower, [filesep 'baseline'])
            meta.Phase = 'baseline';
        elseif contains(folderLower, [filesep 'ptz']) || endsWith(folderLower, [filesep 'ptz'])
            meta.Phase = 'ptz';
        else
            meta.Phase = 'UNK';
        end

        % Group 兜底：从文件夹推断
        if strcmpi(meta.Group, 'UNK')
            if contains(folderLower, 'wt')
                meta.Group = 'WT';
            elseif contains(folderLower, 'hom')
                meta.Group = 'HOM';
            end
        end

        % 1) 读取数据
        try
            data_raw = readmatrix(fpath);
            if size(data_raw,1) < size(data_raw,2), data_raw = data_raw'; end
        catch
            fprintf('  [SKIP] readmatrix failed: %s\n', fpath);
            continue;
        end

        [nFrames, ~] = size(data_raw);
        if nFrames < 10
            fprintf('  [SKIP] too few frames: %s\n', fpath);
            continue;
        end
        time_vec = (0:nFrames-1) / Fs;

        % 2) Moving-bottom baseline detrend
        win_frames = max(3, round(BASELINE_WIN_SEC * Fs));
        base_rough = movmin(data_raw, win_frames, 1);
        smooth_win = max(3, round(win_frames/5));
        baseline_mat = movmean(base_rough, smooth_win, 1);
        data_flat = data_raw - baseline_mat;

        % 3) 事件检测（基于群体平均）
        mean_trace_flat = mean(data_flat, 2, 'omitnan');
        [pks, locs_sec, w_sec] = findpeaks(mean_trace_flat, Fs, ...
            'MinPeakProminence', MIN_PEAK_PROM, ...
            'MinPeakDistance', MIN_DIST_SEC, ...
            'WidthReference', 'halfprom');

        num_events = numel(pks);

        % 4) 指标计算
        current_file_events = [];

        for k = 1:num_events
            center_t = locs_sec(k);
            duration = w_sec(k);

            t_start = center_t - duration/2;
            t_end   = center_t + duration/2;
            idx_start = max(1, floor(t_start * Fs));
            idx_end   = min(nFrames, ceil(t_end * Fs));

            event_chunk   = data_flat(idx_start:idx_end, :);
            trace_segment = mean_trace_flat(idx_start:idx_end);

            auc_val = sum(trace_segment) * (1/Fs);

            roi_std = std(event_chunk, 0, 1);
            valid_mask = roi_std > 1e-4;

            mean_corr = NaN; frac_high_r_nodes = NaN;

            if sum(valid_mask) > 2
                c_mat = corr(event_chunk(:, valid_mask));
                mask_tri = triu(true(size(c_mat)), 1);
                mean_corr = mean(c_mat(mask_tri), 'omitnan');

                node_mean_r = (sum(c_mat, 2) - 1) / (sum(valid_mask) - 1);
                n_high_r_nodes = sum(node_mean_r > HIGH_R_THR);
                frac_high_r_nodes = n_high_r_nodes / max(1, sum(valid_mask));
            end

            ev.FileName = string(fname);
            ev.Group = string(meta.Group);
            ev.Phase = string(meta.Phase);
            ev.Animal = string(meta.Animal);
            ev.PeakTime = center_t;
            ev.PeakAmp = pks(k);
            ev.Duration = duration;
            ev.AUC_Detrended = auc_val;
            ev.Mean_Corr = mean_corr;
            ev.Frac_High_R_Nodes = frac_high_r_nodes;

            current_file_events = [current_file_events; ev]; %#ok<AGROW>
            event_records = [event_records; ev]; %#ok<AGROW>
        end

        fprintf('  [%s] %s-%s: %d events. (%s)\n', meta.Phase, meta.Group, meta.Animal, num_events, fname);

        % 5) 绘图（按指定时间窗截取）-> 保存矢量PDF + .fig（不关闭）
        if DRAW_PLOTS
            make_better_plot_windowed(data_flat, mean_trace_flat, time_vec, current_file_events, meta, MAP_DIR);
        end
    end

    %% ================= EXPORT =================
    if ~isempty(event_records)
        T = struct2table(event_records);
        writetable(T, fullfile(OUT_DIR, 'Improved_Event_Metrics.xlsx'));
        fprintf('完成。Improved_Map 已按时间窗输出 baseline & ptz 的矢量 PDF + .fig（图窗保留不关闭）。\n');
    else
        warning('未检测到事件（但 Improved_Map 仍会输出截取后的矢量 PDF + .fig）。');
    end
end


%% ================= HELPERS =================

function make_better_plot_windowed(flat, mean_flat, t, events, meta, out_dir)
    % 根据 meta 决定绘图窗口
    [t0, t1] = get_plot_window(meta);

    % clamp 到数据范围
    t0 = max(t0, t(1));
    t1 = min(t1, t(end));

    idx = (t >= t0) & (t <= t1);
    if ~any(idx)
        fprintf('  [WARN] No samples within window %.1f-%.1f for %s %s %s\n', ...
            t0, t1, meta.Phase, meta.Group, meta.Animal);
        return;
    end

    tW    = t(idx);
    flatW = flat(idx, :);
    meanW = mean_flat(idx);

    % 事件只显示窗口内的
    eventsW = events;
    if ~isempty(eventsW)
        keep = arrayfun(@(e) (e.PeakTime >= tW(1) && e.PeakTime <= tW(end)), eventsW);
        eventsW = eventsW(keep);
    end

    fig = figure('Visible','on','Color','w','Position',[100 100 900 700]);
    set(fig, 'Renderer', 'painters'); % 矢量渲染更稳
    tiledlayout(4,1, 'Padding','compact', 'TileSpacing','tight');

    % Subplot 1: Mean dF/F（y轴固定 [0 3]）
    nexttile(1);
    plot(tW, meanW, 'k', 'LineWidth', 1.2); hold on;
    ylabel('Mean dF/F');
    title(sprintf('%s %s | %s | %.0f-%.0fs', meta.Group, meta.Phase, meta.Animal, tW(1), tW(end)), 'Interpreter','none');
    xlim([tW(1) tW(end)]);
    grid on;
    ylim([0 3]);

    % Heatmap：对窗口内数据做 Z-score
    z_data = zscore(flatW, 0, 1);

    nexttile(2, [3 1]);
    imagesc(tW, 1:size(z_data,2), z_data');
    colormap(parula);
    caxis([-1 4]);
    ylabel('ROI (Z-Scored)');
    xlabel('Time (s)');
    xlim([tW(1) tW(end)]);

    % === 新增：显示 colorbar（对应当前 heatmap axes）===
    colorbar;

    hold on;
    yl = ylim;
    for k = 1:length(eventsW)
        ts = eventsW(k).PeakTime - eventsW(k).Duration/2;
        te = eventsW(k).PeakTime + eventsW(k).Duration/2;
        if te < tW(1) || ts > tW(end), continue; end
        ts = max(ts, tW(1));
        te = min(te, tW(end));
        patch([ts te te ts], [yl(1) yl(1) yl(2) yl(2)], 'w', ...
              'FaceAlpha', 0.15, 'EdgeColor', 'none');
    end

    drawnow;

    % === 保存矢量 PDF ===
    save_name_pdf = sprintf('Improved_Map_%s_%s_%s_%ds_%ds.pdf', meta.Phase, meta.Group, meta.Animal, round(tW(1)), round(tW(end)));
    exportgraphics(fig, fullfile(out_dir, save_name_pdf), 'ContentType', 'vector');

    % === 同时保存 .fig ===
    save_name_fig = sprintf('Improved_Map_%s_%s_%s_%ds_%ds.fig', meta.Phase, meta.Group, meta.Animal, round(tW(1)), round(tW(end)));
    savefig(fig, fullfile(out_dir, save_name_fig));

    % 不关闭图窗
end


function [t0, t1] = get_plot_window(meta)
    if strcmpi(meta.Phase, 'baseline')
        t0 = 0;    t1 = 500;  return;
    end
    if strcmpi(meta.Phase, 'ptz') && strcmpi(meta.Group, 'WT')
        t0 = 600;  t1 = 1100; return;
    end
    if strcmpi(meta.Phase, 'ptz') && strcmpi(meta.Group, 'HOM')
        t0 = 2100; t1 = 2600; return;
    end

    % 兜底：其他情况默认全程
    t0 = -inf; t1 = inf;
end


function meta = parse_filename(fname)
    % 解析 Group/Animal；Phase 最终会被 folder 强制覆盖
    fnameLower = lower(fname);
    meta.Group  = 'UNK';
    meta.Phase  = 'UNK';
    meta.Animal = 'UNK';

    if contains(fnameLower, 'wt')
        meta.Group='WT';
    elseif contains(fnameLower, 'hom')
        meta.Group='HOM';
    end

    if contains(fnameLower, 'ptz')
        meta.Phase='ptz';
    elseif contains(fnameLower, 'baseline')
        meta.Phase='baseline';
    end

    tok = regexp(fnameLower, '^(\d{8})', 'tokens', 'once');
    if ~isempty(tok)
        meta.Animal = strtrim(tok{1});
    else
        meta.Animal = fnameLower(1:min(10, numel(fnameLower)));
    end
end
