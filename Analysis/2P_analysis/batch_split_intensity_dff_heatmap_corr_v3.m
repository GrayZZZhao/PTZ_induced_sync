function batch_split_intensity_dff_heatmap_corr_v3()
% Batch pipeline:
% 1) Read intensity from Sheet2 (Sheet1 empty)
% 2) Split by baselineN: only 20240215->978; all others->1956
% 3) Save split intensity
% 4) Compute DFF: (FI-F0)/(F0-Fb)
%    Fb = mean(lowest 5% of bg), F0 per ROI = mean(lowest 10% of ROI)
% 5) Heatmap per file/stage: per-ROI min-max norm [0,1], sort by peak time
% 6) Correlation:
%    - baseline: FULL baseline segment (frame 1~split)
%    - PTZ: first X frames of PTZ segment (equiv. [total-split+1, total-split+X])
% 7) Export summary tables:
%    - meanR per animal per stage
%    - #neurons with any R>n per animal per stage (+ #pairs R>n)
% 8) Make large 4-col summary figures (WT base, WT PTZ, HOM base, HOM PTZ)
%
% Output root:
%   D:\suite2p_output\analysis\merged\resize\intensity\analysis_summary

clc; close all;

%% ===================== USER SETTINGS =====================
IN_ROOT  = fullfile('D:\suite2p_output\analysis\merged\resize\intensity');
OUT_ROOT = fullfile(IN_ROOT, 'analysis_summary');

X = 4000;     % <<< PTZ correlation window length (frames)
n = 0.5;    % <<< threshold for "R > n" statistics

SAVE_FIG = true;  % also save .fig besides .png
DPI      = 300;

%% ===================== OUTPUT FOLDERS =====================
ensure_dir(OUT_ROOT);

SPLIT_BASE = fullfile(OUT_ROOT, '01_split_intensity', 'baseline');
SPLIT_PTZ  = fullfile(OUT_ROOT, '01_split_intensity', 'ptz');

DFF_BASE   = fullfile(OUT_ROOT, '02_dff', 'baseline');
DFF_PTZ    = fullfile(OUT_ROOT, '02_dff', 'ptz');

HM_BASE    = fullfile(OUT_ROOT, '03_heatmap', 'baseline');
HM_PTZ     = fullfile(OUT_ROOT, '03_heatmap', 'ptz');
HM_SUM     = fullfile(OUT_ROOT, '03_heatmap', 'summary_4col');

CORR_BASE  = fullfile(OUT_ROOT, '04_corr', 'baseline');
CORR_PTZ   = fullfile(OUT_ROOT, '04_corr', 'ptz');
CORR_SUM   = fullfile(OUT_ROOT, '04_corr', 'summary_4col');

ensure_dir(SPLIT_BASE); ensure_dir(SPLIT_PTZ);
ensure_dir(DFF_BASE);   ensure_dir(DFF_PTZ);
ensure_dir(HM_BASE);    ensure_dir(HM_PTZ);    ensure_dir(HM_SUM);
ensure_dir(CORR_BASE);  ensure_dir(CORR_PTZ);  ensure_dir(CORR_SUM);

LOGFILE = fullfile(OUT_ROOT, 'log_batch_split_dff_v3.txt');
logFID = fopen(LOGFILE, 'w');
cleanupObj = onCleanup(@() fclose(logFID)); %#ok<NASGU>

fprintf(logFID, 'Batch started: %s\n', datestr(now));
fprintf(logFID, 'IN_ROOT : %s\n', IN_ROOT);
fprintf(logFID, 'OUT_ROOT: %s\n', OUT_ROOT);
fprintf(logFID, 'PTZ corr window X=%d, threshold n=%.3f\n\n', X, n);

%% ===================== FIND INPUT FILES =====================
allX = dir(fullfile(IN_ROOT, '*.xlsx'));
allX = allX(~strcmpi({allX.name}, '.') & ~strcmpi({allX.name}, '..'));

if isempty(allX)
    fprintf('No xlsx found in: %s\n', IN_ROOT);
    fprintf(logFID, 'No xlsx found.\n');
    return;
end

%% ===================== RECORDS FOR SUMMARY PLOTTING & EXPORT =====================
rec = struct( ...
    'file',{},'date',{},'animal',{},'group',{}, ...
    'baselineN',{},'nTime',{},'nROI',{}, ...
    'dff_base_path',{},'dff_ptz_path',{}, ...
    'corr_base_path',{},'corr_ptz_path',{}, ...
    'meanR_base',{},'meanR_ptz',{}, ...
    'neuronsGtN_base',{},'neuronsGtN_ptz',{}, ...
    'pairsGtN_base',{},'pairsGtN_ptz',{} );

%% ===================== MAIN LOOP =====================
for i = 1:numel(allX)
    fn = allX(i).name;
    fp = fullfile(allX(i).folder, fn);

    try
        [dateStr, animalStr] = parse_date_animal(fn);
        group = infer_mouse_type_from_name(fn); % 'wt'/'hom'/'unk'
        baselineN = get_baselineN(fn);          % only 20240215->978 else 1956

        fprintf(logFID, '\n[FILE] %s | date=%s | animal=%s | group=%s | baselineN=%d\n', ...
            fn, dateStr, animalStr, group, baselineN);

        % -------- READ sheet 2 intensity --------
        Xmat = read_sheet2_numeric(fp);
        if isempty(Xmat) || size(Xmat,2) < 2
            fprintf(logFID, '  [WARN] Empty/invalid sheet2: %s\n', fn);
            continue;
        end

        nTime = size(Xmat,1);
        if baselineN >= nTime
            fprintf(logFID, '  [WARN] baselineN (%d) >= nTime (%d). Skip.\n', baselineN, nTime);
            continue;
        end

        base_int = Xmat(1:baselineN, :);
        ptz_int  = Xmat(baselineN+1:end, :);
        nROI = size(Xmat,2) - 1;

        % -------- SAVE split intensity --------
        write_matrix_xlsx(fullfile(SPLIT_BASE, add_suffix(strip_ext(fn), '_baseline_intensity.xlsx')), base_int, 'intensity');
        write_matrix_xlsx(fullfile(SPLIT_PTZ,  add_suffix(strip_ext(fn), '_ptz_intensity.xlsx')),      ptz_int,  'intensity');

        % -------- COMPUTE DFF --------
        dff_base = compute_dff_from_intensity(base_int);  % [t x roi]
        dff_ptz  = compute_dff_from_intensity(ptz_int);

        dff_base_path = fullfile(DFF_BASE, add_suffix(strip_ext(fn), '_baseline_dff.xlsx'));
        dff_ptz_path  = fullfile(DFF_PTZ,  add_suffix(strip_ext(fn), '_ptz_dff.xlsx'));
        write_matrix_xlsx(dff_base_path, dff_base, 'dff');
        write_matrix_xlsx(dff_ptz_path,  dff_ptz,  'dff');

        % -------- HEATMAPS (readable, big) --------
        hm_base_png = fullfile(HM_BASE, add_suffix(strip_ext(fn), '_baseline_heatmap.png'));
        hm_ptz_png  = fullfile(HM_PTZ,  add_suffix(strip_ext(fn), '_ptz_heatmap.png'));

        plot_peak_sorted_heatmap_big(dff_base, sprintf('%s | %s baseline', fn, upper(group)), hm_base_png, DPI);
        plot_peak_sorted_heatmap_big(dff_ptz,  sprintf('%s | %s PTZ', fn, upper(group)),      hm_ptz_png,  DPI);

        if SAVE_FIG
            plot_peak_sorted_heatmap_big(dff_base, sprintf('%s | %s baseline', fn, upper(group)), ...
                fullfile(HM_BASE, add_suffix(strip_ext(fn), '_baseline_heatmap.fig')), DPI);
            plot_peak_sorted_heatmap_big(dff_ptz,  sprintf('%s | %s PTZ', fn, upper(group)), ...
                fullfile(HM_PTZ, add_suffix(strip_ext(fn), '_ptz_heatmap.fig')), DPI);
        end

        % -------- CORRELATIONS (baseline full; PTZ uses window X) --------
        base_win = dff_base;  % baseline full: frame 1 ~ split

        xp = min(X, size(dff_ptz,1));
        ptz_win = dff_ptz(1:xp, :); % PTZ window: first X frames

        corr_base = corrcoef(base_win, 'Rows','pairwise');
        corr_ptz  = corrcoef(ptz_win,  'Rows','pairwise');

        corr_base_path = fullfile(CORR_BASE, add_suffix(strip_ext(fn), '_baseline_corr.xlsx'));
        corr_ptz_path  = fullfile(CORR_PTZ,  add_suffix(strip_ext(fn), '_ptz_corr.xlsx'));
        write_matrix_xlsx(corr_base_path, corr_base, 'corr');
        write_matrix_xlsx(corr_ptz_path,  corr_ptz,  'corr');

        % plot corr matrices (bigger)
        cr_base_png = fullfile(CORR_BASE, add_suffix(strip_ext(fn), '_baseline_corr.png'));
        cr_ptz_png  = fullfile(CORR_PTZ,  add_suffix(strip_ext(fn), '_ptz_corr.png'));

        plot_corr_matrix_big(corr_base, sprintf('%s | %s baseline corr (full)', fn, upper(group)), cr_base_png, DPI);
        plot_corr_matrix_big(corr_ptz,  sprintf('%s | %s PTZ corr (first %d)', fn, upper(group), xp), cr_ptz_png, DPI);

        if SAVE_FIG
            plot_corr_matrix_big(corr_base, sprintf('%s | %s baseline corr (full)', fn, upper(group)), ...
                fullfile(CORR_BASE, add_suffix(strip_ext(fn), '_baseline_corr.fig')), DPI);
            plot_corr_matrix_big(corr_ptz,  sprintf('%s | %s PTZ corr (first %d)', fn, upper(group), xp), ...
                fullfile(CORR_PTZ, add_suffix(strip_ext(fn), '_ptz_corr.fig')), DPI);
        end

        % -------- METRICS EXPORT (meanR, neurons R>n, pairs R>n) --------
        [meanR_base, neuronsGtN_base, pairsGtN_base] = corr_metrics(corr_base, n);
        [meanR_ptz,  neuronsGtN_ptz,  pairsGtN_ptz]  = corr_metrics(corr_ptz,  n);

        rec(end+1) = struct( ...
            'file', fn, 'date', dateStr, 'animal', animalStr, 'group', group, ...
            'baselineN', baselineN, 'nTime', nTime, 'nROI', nROI, ...
            'dff_base_path', dff_base_path, 'dff_ptz_path', dff_ptz_path, ...
            'corr_base_path', corr_base_path, 'corr_ptz_path', corr_ptz_path, ...
            'meanR_base', meanR_base, 'meanR_ptz', meanR_ptz, ...
            'neuronsGtN_base', neuronsGtN_base, 'neuronsGtN_ptz', neuronsGtN_ptz, ...
            'pairsGtN_base', pairsGtN_base, 'pairsGtN_ptz', pairsGtN_ptz );

        fprintf(logFID, '  [OK] Done. nROI=%d\n', nROI);

    catch ME
        fprintf(logFID, '  [ERROR] %s\n', ME.message);
        fprintf(logFID, '  %s\n', getReport(ME,'extended','hyperlinks','off'));
    end
end

%% ===================== EXPORT SUMMARY TABLES =====================
if ~isempty(rec)
    T = struct2table(rec);

    % mean correlation summary
    T_mean = table( ...
        string(T.file), string(T.group), string(T.date), string(T.animal), ...
        T.nROI, T.baselineN, T.nTime, ...
        T.meanR_base, T.meanR_ptz, ...
        'VariableNames', {'file','group','date','animal','nROI','baselineN','nTime','meanR_baseline','meanR_PTZ'} );

    writetable(T_mean, fullfile(OUT_ROOT, sprintf('summary_meanR_PTZwinX%d.xlsx', X)), 'Sheet', 'meanR');

    % R>n neurons summary (+pairs)
    T_thr = table( ...
        string(T.file), string(T.group), string(T.date), string(T.animal), ...
        T.nROI, ...
        T.neuronsGtN_base, T.neuronsGtN_ptz, ...
        T.pairsGtN_base,   T.pairsGtN_ptz, ...
        'VariableNames', {'file','group','date','animal','nROI', ...
                          sprintf('neurons_Rgt%.2f_baseline',n), sprintf('neurons_Rgt%.2f_PTZ',n), ...
                          sprintf('pairs_Rgt%.2f_baseline',n),   sprintf('pairs_Rgt%.2f_PTZ',n)} );

    writetable(T_thr, fullfile(OUT_ROOT, sprintf('summary_RgtN_n%.2f_PTZwinX%d.xlsx', n, X)), 'Sheet', 'RgtN');

    fprintf(logFID, '\nExported summary tables.\n');
end

%% ===================== MAKE 4-COL SUMMARY FIGS (RE-PLOT LARGE) =====================
make_4col_heatmap_summary_from_dff(rec, HM_SUM, DPI);
make_4col_corr_summary_from_corr(rec, CORR_SUM, DPI);

fprintf(logFID, '\nBatch finished: %s\n', datestr(now));
disp('DONE. Results saved under:');
disp(OUT_ROOT);

end

%% ===================== HELPERS =====================

function baselineN = get_baselineN(fn)
% only 20240215 uses 978, all others use 1956
if startsWith(fn, '20240215')
    baselineN = 978;
else
    baselineN = 1956;
end
end

function ensure_dir(p)
if ~exist(p,'dir'); mkdir(p); end
end

function [dateStr, animalStr] = parse_date_animal(fn)
dateStr = '';
animalStr = '';
tok = regexp(fn, '^(?<date>\d{8})(?:\s+(?<id>\d{4}))?', 'names');
if ~isempty(tok)
    dateStr = tok.date;
    if isfield(tok,'id') && ~isempty(tok.id)
        animalStr = tok.id;
    end
end
end

function mtype = infer_mouse_type_from_name(fn)
f = lower(fn);
if contains(f,' hom ')
    mtype = 'hom';
elseif contains(f,' wt ')
    mtype = 'wt';
elseif contains(f,'hom')
    mtype = 'hom';
elseif contains(f,'wt')
    mtype = 'wt';
else
    mtype = 'unk';
end
end

function X = read_sheet2_numeric(fp)
% Read numeric matrix from Sheet 2
try
    T = readtable(fp, 'Sheet', 2, 'ReadVariableNames', false);
    X = table2array(T);
    X = double(X);
catch
    X = readmatrix(fp, 'Sheet', 2);
    X = double(X);
end
if isempty(X); return; end
emptyRow = all(isnan(X),2);
X = X(~emptyRow,:);
end

function out = strip_ext(fn)
[~, n, ~] = fileparts(fn);
out = n;
end

function out = add_suffix(nameOrFn, suffixWithExt)
% If nameOrFn already ends with .xlsx/.png etc. handle accordingly outside.
out = [nameOrFn, suffixWithExt];
end

function write_matrix_xlsx(outpath, M, sheetName)
% Ensure parent dir exists
[parent,~,~] = fileparts(outpath);
if ~isempty(parent); ensure_dir(parent); end
writematrix(M, outpath, 'Sheet', sheetName);
end

function dff = compute_dff_from_intensity(intensityMat)
% intensityMat: [time x (1+ROI)], col1=background
bg  = intensityMat(:,1);
roi = intensityMat(:,2:end);
nROI = size(roi,2);

% Fb: lowest 5% mean of bg
bg_sorted = sort(bg, 'ascend');
n5 = max(1, round(0.05 * numel(bg_sorted)));
Fb = mean(bg_sorted(1:n5), 'omitnan');

% F0: each ROI lowest 10% mean
F0 = zeros(1, nROI);
for j = 1:nROI
    t_sorted = sort(roi(:,j), 'ascend');
    n10 = max(1, round(0.10 * numel(t_sorted)));
    F0(j) = mean(t_sorted(1:n10), 'omitnan');
end

den = (F0 - Fb);
den(abs(den) < eps) = NaN;

dff = (roi - F0) ./ den; % [time x roi]
end

function [dff_sorted_T, sortOrder] = peak_sorted_norm(dff)
% Per-ROI normalize to [0,1], then sort by peak time (ascending).
% Return dff_sorted_T as [ROI x time] for direct imagesc.
nTime = size(dff,1);
nROI  = size(dff,2);

dff_norm = zeros(size(dff));
for j = 1:nROI
    t = dff(:,j);
    tMin = min(t, [], 'omitnan');
    tMax = max(t, [], 'omitnan');
    if isfinite(tMin) && isfinite(tMax) && (tMax > tMin)
        dff_norm(:,j) = (t - tMin) ./ (tMax - tMin);
    else
        dff_norm(:,j) = zeros(nTime,1);
    end
end

peakIdx = zeros(1,nROI);
for j = 1:nROI
    [~, idx] = max(dff_norm(:,j));
    peakIdx(j) = idx;
end

[~, sortOrder] = sort(peakIdx, 'ascend');
dff_sorted = dff_norm(:, sortOrder); % [time x ROI]
dff_sorted_T = dff_sorted';          % [ROI x time]
end

function plot_peak_sorted_heatmap_big(dff, figTitle, outpath, DPI)
[dff_sorted_T, ~] = peak_sorted_norm(dff);

fig = figure('Visible','off');
% make bigger canvas
set(fig, 'Units','pixels', 'Position', [100 100 1600 900]);

imagesc(dff_sorted_T);
colormap(parula); colorbar;
set(gca, 'YDir','normal'); % ROI 1 at top by default in imagesc? we keep normal then flip by axis if needed
set(gca, 'YDir','reverse'); % ensure ROI 1 shown top
xlabel('Time (frame index)');
ylabel('ROI (sorted by peak time)');
title(strrep(figTitle,'_','\_'), 'Interpreter','tex');

axis tight;
set(gca,'FontSize',12);

save_figure(fig, outpath, DPI);
close(fig);
end

function plot_corr_matrix_big(C, figTitle, outpath, DPI)
fig = figure('Visible','off');
set(fig, 'Units','pixels', 'Position', [100 100 1100 950]);

imagesc(C, [-1 1]);
colormap(parula); colorbar;
axis image tight;
xlabel('ROI'); ylabel('ROI');
title(strrep(figTitle,'_','\_'), 'Interpreter','tex');
set(gca,'FontSize',12);

save_figure(fig, outpath, DPI);
close(fig);
end

function save_figure(fig, outpath, DPI)
[~,~,ext] = fileparts(outpath);
if strcmpi(ext,'.fig')
    savefig(fig, outpath);
else
    if isempty(ext)
        outpath = [outpath '.png'];
    end
    set(fig,'PaperPositionMode','auto');
    print(fig, outpath, '-dpng', sprintf('-r%d', DPI));
end
end

function [meanR, neuronsGtN, pairsGtN] = corr_metrics(C, thr)
% meanR: mean of upper triangle (excluding diag), ignoring NaN
% neuronsGtN: #neurons that have ANY correlation > thr with another neuron
% pairsGtN: #pairs (upper triangle) with correlation > thr

if isempty(C) || size(C,1) < 2
    meanR = NaN; neuronsGtN = 0; pairsGtN = 0; return;
end

N = size(C,1);
C(1:N+1:end) = NaN; % remove diagonal

% meanR from upper triangle
UT = triu(true(N),1);
vals = C(UT);
meanR = mean(vals, 'omitnan');

% pairs R>thr
pairsGtN = sum(vals > thr, 'omitnan');

% neurons with ANY R>thr
mask = (C > thr);
neuronsGtN = sum(any(mask,2));
end

function make_4col_heatmap_summary_from_dff(rec, outDir, DPI)
ensure_dir(outDir);

% collect by group and stage
[wt_base, wt_ptz, hom_base, hom_ptz] = split_records_by_group(rec);

counts = [numel(wt_base), numel(wt_ptz), numel(hom_base), numel(hom_ptz)];
nRow = max(counts);
if nRow == 0, return; end

fig = figure('Visible','off');
% scale figure size with rows to avoid tiny plots
W = 2200;
H = max(900, 320*nRow);
set(fig, 'Units','pixels', 'Position', [50 50 W H]);

t = tiledlayout(nRow+1, 4, 'Padding','compact', 'TileSpacing','compact');

% header row
titles = {'WT baseline','WT PTZ','HOM baseline','HOM PTZ'};
for c = 1:4
    ax = nexttile(t, c);
    axis off;
    text(0, 0.5, titles{c}, 'FontWeight','bold', 'FontSize', 16);
end

% helper to draw one heatmap tile
drawTile = @(ax, dffPath, label) draw_heatmap_tile(ax, dffPath, label);

for r = 1:nRow
    % WT baseline
    ax = nexttile(t, r*4 + 1); axis off;
    if r <= numel(wt_base)
        drawTile(ax, wt_base(r).dff_base_path, short_label(wt_base(r)));
    else
        text(0.45,0.5,'-','FontSize',18);
    end

    % WT PTZ
    ax = nexttile(t, r*4 + 2); axis off;
    if r <= numel(wt_ptz)
        drawTile(ax, wt_ptz(r).dff_ptz_path, short_label(wt_ptz(r)));
    else
        text(0.45,0.5,'-','FontSize',18);
    end

    % HOM baseline
    ax = nexttile(t, r*4 + 3); axis off;
    if r <= numel(hom_base)
        drawTile(ax, hom_base(r).dff_base_path, short_label(hom_base(r)));
    else
        text(0.45,0.5,'-','FontSize',18);
    end

    % HOM PTZ
    ax = nexttile(t, r*4 + 4); axis off;
    if r <= numel(hom_ptz)
        drawTile(ax, hom_ptz(r).dff_ptz_path, short_label(hom_ptz(r)));
    else
        text(0.45,0.5,'-','FontSize',18);
    end
end

title(t, 'Heatmap summary (peak-time sorted, replotted)', 'FontSize', 18);

outPng = fullfile(outDir, 'Heatmap_Summary_4col_BIG.png');
save_figure(fig, outPng, DPI);
savefig(fig, fullfile(outDir, 'Heatmap_Summary_4col_BIG.fig'));
close(fig);
end

function draw_heatmap_tile(ax, dffPath, labelStr)
dff = readmatrix(dffPath, 'Sheet', 'dff');
[dff_sorted_T, ~] = peak_sorted_norm(dff);

axes(ax); %#ok<LAXES>
imagesc(dff_sorted_T);
colormap(ax, parula);
axis(ax, 'tight');
set(ax, 'YDir','reverse');
set(ax, 'XTick',[], 'YTick',[]);
title(ax, strrep(labelStr,'_','\_'), 'FontSize', 10, 'Interpreter','tex');
end

function make_4col_corr_summary_from_corr(rec, outDir, DPI)
ensure_dir(outDir);

[wt_base, wt_ptz, hom_base, hom_ptz] = split_records_by_group(rec);
counts = [numel(wt_base), numel(wt_ptz), numel(hom_base), numel(hom_ptz)];
nRow = max(counts);
if nRow == 0, return; end

fig = figure('Visible','off');
W = 2200;
H = max(900, 320*nRow);
set(fig, 'Units','pixels', 'Position', [50 50 W H]);

t = tiledlayout(nRow+1, 4, 'Padding','compact', 'TileSpacing','compact');

titles = {'WT baseline corr','WT PTZ corr','HOM baseline corr','HOM PTZ corr'};
for c = 1:4
    ax = nexttile(t, c);
    axis off;
    text(0, 0.5, titles{c}, 'FontWeight','bold', 'FontSize', 16);
end

for r = 1:nRow
    ax = nexttile(t, r*4 + 1); axis off;
    if r <= numel(wt_base)
        draw_corr_tile(ax, wt_base(r).corr_base_path, short_label(wt_base(r)));
    else, text(0.45,0.5,'-','FontSize',18); end

    ax = nexttile(t, r*4 + 2); axis off;
    if r <= numel(wt_ptz)
        draw_corr_tile(ax, wt_ptz(r).corr_ptz_path, short_label(wt_ptz(r)));
    else, text(0.45,0.5,'-','FontSize',18); end

    ax = nexttile(t, r*4 + 3); axis off;
    if r <= numel(hom_base)
        draw_corr_tile(ax, hom_base(r).corr_base_path, short_label(hom_base(r)));
    else, text(0.45,0.5,'-','FontSize',18); end

    ax = nexttile(t, r*4 + 4); axis off;
    if r <= numel(hom_ptz)
        draw_corr_tile(ax, hom_ptz(r).corr_ptz_path, short_label(hom_ptz(r)));
    else, text(0.45,0.5,'-','FontSize',18); end
end

title(t, 'Correlation summary (replotted)', 'FontSize', 18);

outPng = fullfile(outDir, 'Corr_Summary_4col_BIG.png');
save_figure(fig, outPng, DPI);
savefig(fig, fullfile(outDir, 'Corr_Summary_4col_BIG.fig'));
close(fig);
end

function draw_corr_tile(ax, corrPath, labelStr)
C = readmatrix(corrPath, 'Sheet', 'corr');

axes(ax); %#ok<LAXES>
imagesc(C, [-1 1]);
colormap(ax, parula);
axis(ax, 'image');
set(ax, 'XTick',[], 'YTick',[]);
title(ax, strrep(labelStr,'_','\_'), 'FontSize', 10, 'Interpreter','tex');
end

function [wt_base, wt_ptz, hom_base, hom_ptz] = split_records_by_group(rec)
% For alignment in summary plots, we keep the same record list for baseline and ptz
% but separate by group for columns.
wt = rec(strcmpi({rec.group}, 'wt'));
hom = rec(strcmpi({rec.group}, 'hom'));

% baseline lists
wt_base = wt;
hom_base = hom;

% ptz lists
wt_ptz = wt;
hom_ptz = hom;
end

function s = short_label(r)
% compact label used in summary tiles
% prefer date + animal if available; else file stem
stem = strip_ext(r.file);
if ~isempty(r.date) && ~isempty(r.animal)
    s = sprintf('%s_%s', r.date, r.animal);
elseif ~isempty(r.date)
    s = r.date;
else
    s = stem;
end
end

