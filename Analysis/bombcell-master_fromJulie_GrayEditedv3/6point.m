%% ptz_30min_4groups_6point_5minbins.m
% 30-min PTZ with 5-min analysis bins -> 6 timepoints: 5,10,15,20,25,30 min
% Four groups anchored to your 6x5-min counts + duration means.
% Figures 1–5 saved (PNG+PDF); per-bin Kruskal–Wallis p-values shown; CSV of stats written.

clear; clc; close all; rng(42);

% ---------- Colors ----------
colWT   = [  1,  99, 146]/255;  % WT  (#016392)
colHOM  = [199,  33,  41]/255;  % HOM (#C72129)
colCTRL = [122,  62, 157]/255;  % HOM+Control virus (purple)
colKV   = [ 46, 125,  50]/255;  % HOM+Kv1.1 virus (green)

test_overall = 'Kruskal-Wallis';
test_pair    = 'Wilcoxon rank-sum';

% ---------- Output ----------
outdir = fullfile(pwd,'ptz_30min_4groups_6point_5minbins');
if ~exist(outdir,'dir'); mkdir(outdir); end

% ---------- Cohort / Time ----------
nPerGroup = 6;                    % mice/group
T_total   = 30*60;                % 30 min (s)

% Generation bins (5-min) & analysis bins (also 5-min now)
gen_edges      = 0:300:T_total;   % for simulating events (5-min)
analysis_edges = 0:300:T_total;   % for plotting/stats (5-min)
nBins          = numel(analysis_edges)-1;           % = 6
xvals          = analysis_edges(2:end)/60;          % [5 10 15 20 25 30] (min)
labels_bins    = {'0-5','5-10','10-15','15-20','20-25','25-30'};
mins_per_bin   = (analysis_edges(2)-analysis_edges(1))/60;  % = 5

% Midpoints for each 5-min bin (sec) to anchor durations:
dur_times_sec6 = (2.5:5:27.5)*60;   % 2.5,7.5,12.5,17.5,22.5,27.5 min
dur_sigma = 0.40;                   % log-normal spread

% ===================== Your 6x5-min anchors =====================
% WT
WT_counts6   = [26, 33, 34, 35, 23, 17];
WT_dur_means = [0.04667, 0.04335, 0.04000, 0.03350, 0.02670, 0.02600];

% HOM
HOM_counts6   = [30, 45, 46, 31, 33, 33];
HOM_dur_means = [0.41700, 0.47500, 0.53300, 0.44950, 0.36600, 0.03600];

% HOM + Control virus
CTRL_counts6   = [33, 50, 47, 33, 30, 36];
CTRL_dur_means = [0.45400, 0.47100, 0.48800, 0.43400, 0.38000, 0.37000];

% HOM + Kv1.1 virus
KV_counts6   = [32, 29, 26, 14, 14, 17];
KV_dur_means = [0.17500, 0.21650, 0.25800, 0.21750, 0.17700, 0.17000];

% Convert counts to events/min per 5-min bin
rate_WT   = WT_counts6   / 5;
rate_HOM  = HOM_counts6  / 5;
rate_CTRL = CTRL_counts6 / 5;
rate_KV   = KV_counts6   / 5;

% Build ?(t) on a fine grid so E[dur]=given mean in each 5-min bin.
% For X~LogNormal(?,?): E[X]=exp(? + ?^2/2) -> ? = ln(mean) ? ?^2/2
tgrid = linspace(0, T_total, 601);
mu_WT   = interp1(dur_times_sec6,  log(WT_dur_means)   - 0.5*dur_sigma^2, tgrid, 'nearest', 'extrap');
mu_HOM  = interp1(dur_times_sec6,  log(HOM_dur_means)  - 0.5*dur_sigma^2, tgrid, 'nearest', 'extrap');
mu_CTRL = interp1(dur_times_sec6,  log(CTRL_dur_means) - 0.5*dur_sigma^2, tgrid, 'nearest', 'extrap');
mu_KV   = interp1(dur_times_sec6,  log(KV_dur_means)   - 0.5*dur_sigma^2, tgrid, 'nearest', 'extrap');

% ===================== Simulate events (per group) =====================
WT   = simulate_from_bins(nPerGroup, gen_edges, rate_WT,   tgrid, mu_WT,   dur_sigma); WT.group = repmat({'WT'},        height(WT),1);
HOM  = simulate_from_bins(nPerGroup, gen_edges, rate_HOM,  tgrid, mu_HOM,  dur_sigma); HOM.group = repmat({'HOM'},       height(HOM),1);
CTRL = simulate_from_bins(nPerGroup, gen_edges, rate_CTRL, tgrid, mu_CTRL, dur_sigma); CTRL.group = repmat({'HOM+Ctrl'},  height(CTRL),1);
KV   = simulate_from_bins(nPerGroup, gen_edges, rate_KV,   tgrid, mu_KV,   dur_sigma); KV.group = repmat({'HOM+Kv1.1'}, height(KV),1);

% ===================== Summaries in 5-min analysis bins (6 pts) =====================
S_WT   = per_mouse_binned(WT,   nPerGroup, analysis_edges); S_WT.group   = repmat("WT",        height(S_WT),1);
S_HOM  = per_mouse_binned(HOM,  nPerGroup, analysis_edges); S_HOM.group  = repmat("HOM",       height(S_HOM),1);
S_CTRL = per_mouse_binned(CTRL, nPerGroup, analysis_edges); S_CTRL.group = repmat("HOM+Ctrl",  height(S_CTRL),1);
S_KV   = per_mouse_binned(KV,   nPerGroup, analysis_edges); S_KV.group   = repmat("HOM+Kv1.1", height(S_KV),1);

% Add derived metrics (now per 5-min bin)
S_WT   = add_bin_metrics(S_WT,   analysis_edges);
S_HOM  = add_bin_metrics(S_HOM,  analysis_edges);
S_CTRL = add_bin_metrics(S_CTRL, analysis_edges);
S_KV   = add_bin_metrics(S_KV,   analysis_edges);

% AUC over 0–30 min (per mouse)
AUC_events = struct( ...
    'WT',   accumarray(S_WT.mouse,   S_WT.n_events), ...
    'HOM',  accumarray(S_HOM.mouse,  S_HOM.n_events), ...
    'CTRL', accumarray(S_CTRL.mouse, S_CTRL.n_events), ...
    'KV',   accumarray(S_KV.mouse,   S_KV.n_events)  );
AUC_burden = struct( ...
    'WT',   accumarray(S_WT.mouse,   S_WT.total_duration)/T_total*100, ...
    'HOM',  accumarray(S_HOM.mouse,  S_HOM.total_duration)/T_total*100, ...
    'CTRL', accumarray(S_CTRL.mouse, S_CTRL.total_duration)/T_total*100, ...
    'KV',   accumarray(S_KV.mouse,   S_KV.total_duration)/T_total*100 );

% ===================== Stats (overall + pairwise) =====================
p_kw_freq   = kw_per_bin({S_WT,S_HOM,S_CTRL,S_KV}, 'freq',          nBins);
p_kw_dur    = kw_per_bin({S_WT,S_HOM,S_CTRL,S_KV}, 'mean_duration', nBins);
p_kw_burden = kw_per_bin({S_WT,S_HOM,S_CTRL,S_KV}, 'burden_pct',    nBins);

p_kw_auc_events = kw_overall({AUC_events.WT, AUC_events.HOM, AUC_events.CTRL, AUC_events.KV});
p_kw_auc_burden = kw_overall({AUC_burden.WT, AUC_burden.HOM, AUC_burden.CTRL, AUC_burden.KV});

% Pairwise Wilcoxon rank-sum (saved to CSV)
groups_order = {'WT','HOM','HOM+Ctrl','HOM+Kv1.1'};
pairs = nchoosek(1:4,2);
rows = {};
for b = 1:nBins
    Vfreq = { S_WT.freq(S_WT.bin==b), S_HOM.freq(S_HOM.bin==b), S_CTRL.freq(S_CTRL.bin==b), S_KV.freq(S_KV.bin==b) };
    Vdur  = { S_WT.mean_duration(S_WT.bin==b), S_HOM.mean_duration(S_HOM.bin==b), S_CTRL.mean_duration(S_CTRL.bin==b), S_KV.mean_duration(S_KV.bin==b) };
    Vbur  = { S_WT.burden_pct(S_WT.bin==b), S_HOM.burden_pct(S_HOM.bin==b), S_CTRL.burden_pct(S_CTRL.bin==b), S_KV.burden_pct(S_KV.bin==b) };
    for pi = 1:size(pairs,1)
        i = pairs(pi,1); j = pairs(pi,2);
        rows(end+1,:) = {'frequency', labels_bins{b}, groups_order{i}, groups_order{j}, ranksum(Vfreq{i}, Vfreq{j}), test_pair}; %#ok<SAGROW>
        rows(end+1,:) = {'duration',  labels_bins{b}, groups_order{i}, groups_order{j}, ranksum(Vdur{i},  Vdur{j}),  test_pair}; %#ok<SAGROW>
        rows(end+1,:) = {'burden',    labels_bins{b}, groups_order{i}, groups_order{j}, ranksum(Vbur{i},  Vbur{j}),  test_pair}; %#ok<SAGROW>
    end
end
VaucE = {AUC_events.WT, AUC_events.HOM, AUC_events.CTRL, AUC_events.KV};
VaucB = {AUC_burden.WT, AUC_burden.HOM, AUC_burden.CTRL, AUC_burden.KV};
for pi = 1:size(pairs,1)
    i = pairs(pi,1); j = pairs(pi,2);
    rows(end+1,:) = {'AUC_events','AUC', groups_order{i}, groups_order{j}, ranksum(VaucE{i}, VaucE{j}), test_pair}; %#ok<SAGROW>
    rows(end+1,:) = {'AUC_burden','AUC', groups_order{i}, groups_order{j}, ranksum(VaucB{i}, VaucB{j}), test_pair}; %#ok<SAGROW>
end

% Overall + pairwise table (matching columns)
T_overall = table( ...
    [repmat({'frequency'},nBins,1); repmat({'duration'},nBins,1); repmat({'burden'},nBins,1); {'AUC_events'}; {'AUC_burden'}], ...
    [labels_bins'; labels_bins'; labels_bins'; {'AUC'}; {'AUC'}], ...
    [p_kw_freq;   p_kw_dur;     p_kw_burden;   p_kw_auc_events; p_kw_auc_burden], ...
    repmat({test_overall}, nBins*3+2, 1), ...
    'VariableNames', {'metric','bin','p_value','test'});
T_overall.group1 = repmat({'ALL'}, height(T_overall), 1);
T_overall.group2 = repmat({'ALL'}, height(T_overall), 1);
T_overall = T_overall(:, {'metric','bin','group1','group2','p_value','test'});

T_pairwise = cell2table(rows, 'VariableNames', {'metric','bin','group1','group2','p_value','test'});
Tstats = [T_overall; T_pairwise];
writetable(Tstats, fullfile(outdir,'stats_pvalues.csv'));

% ===================== Figures (now 6 timepoints) =====================
% Fig 1: Frequency time-course
fig1 = figure('Color','w','Position',[120 120 980 540]); hold on
plot_timecourse_multi({S_WT,S_HOM,S_CTRL,S_KV}, {'WT','HOM','HOM+Ctrl','HOM+Kv1.1'}, ...
                      'freq', 'Sync events per min', 'Event frequency (5-min bins)', ...
                      [colWT; colHOM; colCTRL; colKV], xvals);
add_pbox_kw(gca, sprintf('%s p (frequency):', test_overall), labels_bins, p_kw_freq);
save_fig(fig1, fullfile(outdir,'1_frequency_timecourse'));

% Fig 2: Duration time-course
fig2 = figure('Color','w','Position',[120 120 980 540]); hold on
plot_timecourse_multi({S_WT,S_HOM,S_CTRL,S_KV}, {'WT','HOM','HOM+Ctrl','HOM+Kv1.1'}, ...
                      'mean_duration', 'Event duration (s)', 'Sync event duration (5-min bins)', ...
                      [colWT; colHOM; colCTRL; colKV], xvals);
add_pbox_kw(gca, sprintf('%s p (duration):', test_overall), labels_bins, p_kw_dur);
save_fig(fig2, fullfile(outdir,'2_duration_timecourse'));

% Fig 3: Burden time-course
fig3 = figure('Color','w','Position',[120 120 980 540]); hold on
plot_timecourse_multi({S_WT,S_HOM,S_CTRL,S_KV}, {'WT','HOM','HOM+Ctrl','HOM+Kv1.1'}, ...
                      'burden_pct', 'Time in sync (% per bin)', 'Synchronization burden (5-min bins)', ...
                      [colWT; colHOM; colCTRL; colKV], xvals);
add_pbox_kw(gca, sprintf('%s p (burden):', test_overall), labels_bins, p_kw_burden);
save_fig(fig3, fullfile(outdir,'3_burden_timecourse'));

% Fig 4: AUC panels (unchanged)
fig4 = figure('Color','w','Position',[120 120 1100 480]);
subplot(1,2,1);
bar_dots_groups({AUC_events.WT, AUC_events.HOM, AUC_events.CTRL, AUC_events.KV}, ...
                {'WT','HOM','HOM+Ctrl','HOM+Kv1.1'}, [colWT;colHOM;colCTRL;colKV], ...
                'Total events (0–30 min)', 'AUC: total events');
text(0.5,0.95,sprintf('%s p=%.3g', test_overall, p_kw_auc_events),'Units','normalized', ...
     'HorizontalAlignment','center','FontWeight','bold');
subplot(1,2,2);
bar_dots_groups({AUC_burden.WT, AUC_burden.HOM, AUC_burden.CTRL, AUC_burden.KV}, ...
                {'WT','HOM','HOM+Ctrl','HOM+Kv1.1'}, [colWT;colHOM;colCTRL;colKV], ...
                'Time in sync (% of 30 min)', 'AUC: burden');
text(0.5,0.95,sprintf('%s p=%.3g', test_overall, p_kw_auc_burden),'Units','normalized', ...
     'HorizontalAlignment','center','FontWeight','bold');
save_fig(fig4, fullfile(outdir,'4_auc_panels'));

% Fig 5: Cumulative events (means)
fig5 = figure('Color','w','Position',[120 120 980 540]); hold on
[t_wt, c_wt]   = cumulative_curve(WT,   nPerGroup, T_total);
[t_hom, c_hom] = cumulative_curve(HOM,  nPerGroup, T_total);
[t_c, c_c]     = cumulative_curve(CTRL, nPerGroup, T_total);
[t_k, c_k]     = cumulative_curve(KV,   nPerGroup, T_total);
plot(t_wt/60,  c_wt,  'LineWidth',2, 'Color',colWT);
plot(t_hom/60, c_hom, 'LineWidth',2, 'Color',colHOM);
plot(t_c/60,   c_c,   'LineWidth',2, 'Color',colCTRL);
plot(t_k/60,   c_k,   'LineWidth',2, 'Color',colKV);
xlabel('Time after onset (min)'); ylabel('Mean cumulative events');
title('Cumulative events (0–30 min)');
legend({'WT','HOM','HOM+Ctrl','HOM+Kv1.1'},'Location','northwest');
grid on; set(gca,'Box','off','LineWidth',1.2);
save_fig(fig5, fullfile(outdir,'5_cumulative_events'));

fprintf('\nFigures + stats written to: %s\n', outdir);

%% ===================== Helpers =====================

function T = simulate_from_bins(n, bin_edges, rate_min, tgrid, mu_t, sigma)
% Simulate Poisson counts per 5-min bin; durations ~ LogNormal(mu(t), sigma).
T = table([],[],[], 'VariableNames',{'mouse','start','duration'});
for m = 1:n
    for b = 1:numel(bin_edges)-1
        t1 = bin_edges(b); t2 = bin_edges(b+1);
        lam = rate_min(b) * (t2-t1)/60;
        k   = poissrnd(lam);
        starts = sort(t1 + rand(k,1)*(t2-t1));
        if ~isempty(starts)
            mu_local = interp1(tgrid, mu_t, starts, 'nearest', 'extrap');
            durs = lognrnd(mu_local, sigma, size(starts));
            T = [T; table(repmat(m,k,1), starts, durs, ...
                'VariableNames',{'mouse','start','duration'})]; %#ok<AGROW>
        end
    end
end
end

function S = per_mouse_binned(T, nM, edges)
nBins = numel(edges)-1; nrows = nM*nBins;
mouse = zeros(nrows,1); binIdx = zeros(nrows,1);
n_events = zeros(nrows,1); mean_duration = zeros(nrows,1); total_duration = zeros(nrows,1);
row = 0;
for m = 1:nM
    for b = 1:nBins
        row = row+1; t1=edges(b); t2=edges(b+1);
        if isempty(T), ii=false(0,1); else, ii=(T.mouse==m)&(T.start>=t1)&(T.start<t2); end
        ne = sum(ii);
        if ne>0, td=sum(T.duration(ii)); md=mean(T.duration(ii)); else, td=0; md=0; end
        mouse(row)=m; binIdx(row)=b; n_events(row)=ne; total_duration(row)=td; mean_duration(row)=md;
    end
end
S = table(mouse, binIdx, n_events, mean_duration, total_duration, ...
          'VariableNames', {'mouse','bin','n_events','mean_duration','total_duration'});
end

function S = add_bin_metrics(S, edges)
mins_per_bin = (edges(2)-edges(1))/60;          % = 5
S.freq       = S.n_events ./ mins_per_bin;      % events/min (per 5-min bin)
S.burden_pct = 100*S.total_duration ./ (edges(2)-edges(1));
end

function p = kw_per_bin(Slist, field, nBins)
p = zeros(nBins,1);
for b = 1:nBins
    y = []; g = [];
    for gi = 1:numel(Slist)
        vals = Slist{gi}.(field)(Slist{gi}.bin==b);
        y = [y; vals]; g = [g; gi*ones(size(vals))]; %#ok<AGROW>
    end
    if numel(unique(g))>1 && ~isempty(y), p(b)=kruskalwallis(y,g,'off'); else, p(b)=NaN; end
end
end

function p = kw_overall(cellvecs)
y = []; g = [];
for gi = 1:numel(cellvecs)
    v = cellvecs{gi}; y=[y; v(:)]; g=[g; gi*ones(numel(v),1)]; %#ok<AGROW>
end
if numel(unique(g))>1 && ~isempty(y), p=kruskalwallis(y,g,'off'); else, p=NaN; end
end

function plot_timecourse_multi(Slist, glabels, field, ylab, ttl, colors, xvals)
hold on
% spaghetti lines
for gi = 1:numel(Slist)
    col = colors(gi,:); faint = lighten(col,0.65); S=Slist{gi};
    for m = 1:max(S.mouse)
        y = S.(field)(S.mouse==m);
        plot(xvals, y, '-', 'Color', faint, 'LineWidth', 1);
    end
end
% means ± SEM
for gi = 1:numel(Slist)
    S=Slist{gi}; col=colors(gi,:);
    m = arrayfun(@(b) mean(S.(field)(S.bin==b)), 1:numel(xvals));
    s = arrayfun(@(b) sem(S.(field)(S.bin==b)),  1:numel(xvals));
    errorbar(xvals, m, s, '-o', 'Color', col, 'MarkerFaceColor', col, 'LineWidth', 2);
end
xlabel('Time after onset (min)'); ylabel(ylab); title(ttl);
legend(glabels, 'Location','best'); xlim([0 30]); grid on; set(gca,'Box','off','LineWidth',1.2);
end

function add_pbox_kw(ax, header, labels_bins, pvals)
lines = cell(numel(pvals)+1,1); lines{1}=header;
for i=1:numel(pvals)
    if isnan(pvals(i)), txt='p=NA'; else, txt=sprintf('p=%.3g', pvals(i)); end
    lines{1+i} = sprintf('%s: %s', labels_bins{i}, txt);
end
text(ax, 0.98, 0.98, strjoin(lines, '\n'), 'Units','normalized', ...
     'HorizontalAlignment','right','VerticalAlignment','top', ...
     'BackgroundColor','w','Margin',6,'FontSize',10);
end

function bar_dots_groups(values_cell, glabels, colors, ylab, ttl)
ng = numel(values_cell); M=zeros(1,ng); S=zeros(1,ng);
for gi=1:ng, v=values_cell{gi}; M(gi)=mean(v); S(gi)=sem(v); end
b = bar(1:ng, M, 'FaceColor','flat'); hold on
for gi=1:ng, b.CData(gi,:) = colors(gi,:); end
errorbar(1:ng,M,S,'k','LineStyle','none','CapSize',8,'LineWidth',1.2);
j=0.10;
for gi=1:ng
    v=values_cell{gi}; xj=gi+(rand(size(v))-0.5)*2*j;
    scatter(xj,v,36,'MarkerFaceColor',colors(gi,:),'MarkerEdgeColor','k');
end
set(gca,'XTick',1:ng,'XTickLabel',glabels); xlim([0.5 ng+0.5]);
ylabel(ylab); title(ttl); grid on; set(gca,'Box','off','LineWidth',1.2);
end

function c2 = lighten(c,f), c2 = 1 - (1 - c) * (1 - f); end
function s  = sem(x), x=x(:); s = std(x,'omitnan')/sqrt(sum(~isnan(x))); end

function save_fig(h, base)
[folder,~,~] = fileparts(base);
if ~isempty(folder) && ~exist(folder,'dir'); mkdir(folder); end
try
    exportgraphics(h,[base '.png'],'Resolution',300);
    exportgraphics(h,[base '.pdf']);
catch
    set(h,'PaperPositionMode','auto');
    try, print(h,[base '.png'],'-dpng','-r300'); catch, print(h,'-dpng','-r300',[base '.png']); end
    set(h,'Renderer','painters');
    try, print(h,[base '.pdf'],'-dpdf','-painters'); catch, print(h,'-dpdf','-painters',[base '.pdf']); end
end
end

function [t,cmean]=cumulative_curve(T,nM,Twin)
t=linspace(0,Twin,601); curves=zeros(nM,numel(t));
for m=1:nM, tt=sort(T.start(T.mouse==m)); curves(m,:)=arrayfun(@(x)sum(tt<=x),t); end
cmean=mean(curves,1);
end
