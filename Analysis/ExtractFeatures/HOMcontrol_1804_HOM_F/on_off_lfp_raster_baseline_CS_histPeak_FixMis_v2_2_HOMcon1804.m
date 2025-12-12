

%%   找id
% ---- Verify neuron_id vs clustertwo.xlsx col 2 (+1) ----
% Assumes you already built resultArray_wt, so:
% neuron_id = resultArray_wt(:,1);

excel_path = 'H:\npxlkv11_summer_2025\1804_HOM_F_con\2025-07-16_12-29-25\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\clustertwo.xlsx';
%excel_path = '/Volumes/GZ_NPXL_25/seizure/2024-01-02_WT_HOM-male-adult/2024-01-02_13-43-59/Record Node 101/experiment2/recording2/continuous/Neuropix-PXI-100.ProbeA-AP/clustertwo.xlsx'
% Read excel (robust to headers/mixed types)
try
    T = readtable(excel_path);                   % best if the sheet has headers
    % Get 2nd column numerically
    if width(T) >= 2
        col2 = T{:,2};
    else
        error('Excel file has fewer than 2 columns.');
    end
    if ~isnumeric(col2)
        col2 = str2double(string(col2));         % coerce text to numbers
    end
catch
    % Fallback if table read fails
    X = readmatrix(excel_path);
    if size(X,2) < 2
        error('Excel file has fewer than 2 columns.');
    end
    col2 = X(:,2);
end

excel_ids0 = col2(:);
excel_ids1 = excel_ids0 + 1;                     % Excel’s 0-based IDs -> +1

neuron_id = resultArray_wt(:,1);                 % from your pipeline
neuron_id = neuron_id(:);

fprintf('Counts -> neuron_id: %d, excel col2(+1): %d\n', numel(neuron_id), numel(excel_ids1));

% ----- Set-level comparison (ignores order/duplicates) -----
equal_sets = isequal(sort(neuron_id), sort(excel_ids1(~isnan(excel_ids1))));
if equal_sets
    fprintf('[OK] Same ID set after +1 conversion (order may differ).\n');
else
    only_in_neuron = setdiff(neuron_id, excel_ids1);
    only_in_excel  = setdiff(excel_ids1, neuron_id);
    fprintf('[WARN] ID sets differ.\n');
    if ~isempty(only_in_neuron)
        fprintf('  IDs only in neuron_id (first up to 10): %s\n', mat2str(only_in_neuron(1:min(10,end))'));
    end
    if ~isempty(only_in_excel)
        fprintf('  IDs only in excel(+1) (first up to 10): %s\n', mat2str(only_in_excel(1:min(10,end))'));
    end
end

% ----- Order-level comparison (do they line up row-by-row?) -----
nmin = min(numel(neuron_id), numel(excel_ids1));
rowwise_equal = isequal(neuron_id(1:nmin), excel_ids1(1:nmin));
if rowwise_equal && numel(neuron_id)==numel(excel_ids1)
    fprintf('[OK] Same order and same length.\n');
elseif rowwise_equal
    fprintf('[OK] First %d rows match in order, but lengths differ.\n', nmin);
else
    % Show a few row-wise mismatches
    mism = find(neuron_id(1:nmin) ~= excel_ids1(1:nmin));
    fprintf('[INFO] Row order differs. First up to 10 mismatches (row, neuron_id, excel+1):\n');
    for k = 1:min(10, numel(mism))
        r = mism(k);
        fprintf('  %d: %g vs %g\n', r, neuron_id(r), excel_ids1(r));
    end
end

% ----- Optional: map neuron_id -> row in Excel (if you need labels later) -----
[tf, loc] = ismember(neuron_id, excel_ids1);   % loc gives index in excel list
fprintf('[Map] %d/%d neuron_ids found in excel(+1).\n', sum(tf), numel(neuron_id));

%%
% --- Region labeling from Excel depth (col 4) and splitting variables ---

% Assumes you already have:
%   resultArray_wt           % rows = neurons; col 1 = neuron_id/template ID
%   neuron_id = resultArray_wt(:,1);
%   selected_neurons         % row indices of interest in resultArray_wt
%   excel_path set previously

excel_path = 'H:\npxlkv11_summer_2025\1804_HOM_F_con\2025-07-16_12-29-25\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\clustertwo.xlsx';
%excel_path = '/Volumes/GZ_NPXL_25/seizure/2024-01-02_WT_HOM-male-adult/2024-01-02_13-43-59/Record Node 101/experiment2/recording2/continuous/Neuropix-PXI-100.ProbeA-AP/clustertwo.xlsx'
% --- Read depth (Column 4) robustly
try
    T = readtable(excel_path);
    if width(T) < 4
        error('Excel file has fewer than 4 columns.');
    end
    depth_col = T{:,4};
    if ~isnumeric(depth_col)
        depth_col = str2double(string(depth_col));
    end
catch
    X = readmatrix(excel_path);
    if size(X,2) < 4
        error('Excel file has fewer than 4 columns.');
    end
    depth_col = X(:,4);
end

depth_um = depth_col(:);                         % ensure column vector
neuron_id = resultArray_wt(:,1);                 % your IDs

% --- Quick sanity checks (should already be true from your last step)
assert(numel(neuron_id)==numel(depth_um), 'neuron_id and depth length mismatch.');
% If you want to be extra strict about the ID alignment:
% excel_ids1 = (T{:,2} if numeric else readmatrix)(:,2) + 1; % already verified earlier
% assert(isequal(neuron_id, excel_ids1(:)), 'Row order changed since last check.');

% --- Region rule: depth < 2625 => striatum; otherwise cortex
is_STR = depth_um < 2625;
is_CTX = ~is_STR;

region = repmat("cortex", numel(depth_um), 1);
region(is_STR) = "striatum";

% --- Split neuron_id by region
neuron_id_STR = neuron_id(is_STR);
neuron_id_CTX = neuron_id(is_CTX);

% --- Split selected_neurons (these are ROW INDICES in resultArray_wt)
if ~exist('selected_neurons','var') || isempty(selected_neurons)
    warning('selected_neurons not found; creating empty region splits.');
    selected_neurons_STR = [];
    selected_neurons_CTX = [];
else
    selected_neurons = selected_neurons(:);
    selected_neurons_STR = selected_neurons(is_STR(selected_neurons));
    selected_neurons_CTX = selected_neurons(is_CTX(selected_neurons));
end

% --- (Optional) region table for convenient lookup
RegionTable = table( ...
    neuron_id(:), ...
    depth_um(:), ...
    categorical(region), ...
    'VariableNames', {'neuron_id','depth_um','region'} );

% --- Save to MAT for reuse
out_mat = 'neuron_region_labels.mat';
save(out_mat, 'is_STR','is_CTX','region','depth_um', ...
    'neuron_id','neuron_id_STR','neuron_id_CTX', ...
    'selected_neurons','selected_neurons_STR','selected_neurons_CTX', ...
    'RegionTable');

fprintf('[OK] Regions labeled. Saved outputs to %s\n', out_mat);

% --- Tiny printout
fprintf('Counts -> STR: %d, CTX: %d (total %d)\n', nnz(is_STR), nnz(is_CTX), numel(is_STR));
if exist('selected_neurons','var') && ~isempty(selected_neurons)
    fprintf('Selected -> STR: %d, CTX: %d (total %d)\n', ...
        numel(selected_neurons_STR), numel(selected_neurons_CTX), numel(selected_neurons));
end



%% ---- Build 1 s–extended SWD intervals (once) ----
% extend_sec = 0.0;                                   % <-- change this if you want
% fs0 = original_sampling_rate;                     % 2500 in your script
% N   = numel(corrected_baseline);
% extN = round(extend_sec * fs0);
% 
% % bounds for each event, then extend and clip to [1, N]
% evt_bounds = nan(numel(swd_events),2);
% for i = 1:numel(swd_events)
%     v = swd_events{i};
%     if ~isempty(v)
%         evt_bounds(i,:) = [v(1) v(end)];
%     end
% end
% evt_bounds = evt_bounds(~isnan(evt_bounds(:,1)),:);
% evt_ext = [ max(1, evt_bounds(:,1) - extN),  min(N, evt_bounds(:,2) + extN) ];
% 
% % merge overlapping extended intervals
% evt_ext = sortrows(evt_ext,1);
% merged_ext = [];
% for k = 1:size(evt_ext,1)
%     if isempty(merged_ext) || evt_ext(k,1) > merged_ext(end,2) + 1
%         merged_ext = [merged_ext; evt_ext(k,:)]; %#ok<AGROW>
%     else
%         merged_ext(end,2) = max(merged_ext(end,2), evt_ext(k,2));
%     end
% end
% swd_events_ext_1s = merged_ext;                   % [K x 2], sample index start/end
% 

extend_sec = 1;                                    % 延长秒数 (仅向后)
fs0 = original_sampling_rate;                      % 2500 Hz
N   = numel(corrected_baseline);
extN = round(extend_sec * fs0);

% 仅提取每个 event 的起点
evt_bounds = nan(numel(swd_events),2);
for i = 1:numel(swd_events)
    v = swd_events{i};
    if ~isempty(v)
        start_idx = v(1);                          % 起点
        evt_bounds(i,:) = [start_idx, min(N, start_idx + extN)]; % 起点 + 1 秒
    end
end
evt_bounds = evt_bounds(~isnan(evt_bounds(:,1)),:);

% 合并重叠区间
evt_bounds = sortrows(evt_bounds,1);
merged_ext = [];
for k = 1:size(evt_bounds,1)
    if isempty(merged_ext) || evt_bounds(k,1) > merged_ext(end,2) + 1
        merged_ext = [merged_ext; evt_bounds(k,:)]; %#ok<AGROW>
    else
        merged_ext(end,2) = max(merged_ext(end,2), evt_bounds(k,2));
    end
end

swd_events_ext_1s = merged_ext;                    % [K x 2], 每个事件持续 1 秒


%% ==================== DROP-IN: region-specific composites (2,3,5,6) ====================

% ---------- Resolve region selections ----------
% Expecting: is_STR (logical), neuron_id (from resultArray_wt(:,1))
%            selected_neurons (row indices, optional)
% Build selected_neurons_STR/CTX as row indices in resultArray_wt.
if ~exist('is_STR','var')
    error('is_STR not found. Run the depth-based labeling step first.');
end
if ~exist('neuron_id','var') || isempty(neuron_id)
    neuron_id = resultArray_wt(:,1);  % safety
end

% If selected_neurons not provided, default to "all rows" per region
if ~exist('selected_neurons','var') || isempty(selected_neurons)
    selected_neurons = (1:size(resultArray_wt,1)).';
end
selected_neurons = selected_neurons(:);
is_CTX = ~is_STR;

selected_neurons_STR = selected_neurons(is_STR(selected_neurons));
selected_neurons_CTX = selected_neurons(is_CTX(selected_neurons));

% ---------- X-limits ----------
if ~exist('x_start','var') || ~exist('x_end','var') || isempty(x_start) || isempty(x_end)
    x_start = 0;
    % fall back to max spike time if A/time not defined
    try
        x_end = max(length(A)/original_sampling_rate, max(resultArray_wt(:)));
    catch
        x_end = max(resultArray_wt(:));
    end
    if ~isfinite(x_end) || x_end <= x_start
        x_end = 1;  % minimal guard
    end
end

% ---------- Common subplot positions (matching your previous layout rows) ----------
pos_2 = [0.1, 0.70, 0.8, 0.10];  % (2) Filtered LFP
pos_3 = [0.1, 0.55, 0.8, 0.10];  % (3) SWD detection
pos_5 = [0.1, 0.25, 0.8, 0.13];  % (5) Raster (region-only)
pos_6 = [0.1, 0.10, 0.8, 0.07];  % (6) Population rate (region-only)

% ---------- Helpers for LFP ----------
if ~exist('fs','var') || isempty(fs), fs = 2500; end
[b_lfp, a_lfp] = butter(4, [5 60]/(fs/2), 'bandpass');
lfp_mean = mean(d,1);
lfp_filt = filtfilt(b_lfp, a_lfp, lfp_mean);
lfp_scale = 0.195;  % µV/bit (adjust if needed)
lfp_t = (1:numel(lfp_filt))/fs;

% ---------- Convenience handle for SWD trace ----------
swd_t = (1:length(corrected_baseline))/original_sampling_rate;

% ==================== Figure: STRIATUM ====================
figSTR = figure('Name','Region composite — Striatum (2,3,5,6)','Position',[120 60 1200 820]);
axSTR = gobjects(4,1);

% (2) Filtered LFP
axSTR(1) = subplot('Position', pos_2);
plot(lfp_t, lfp_filt*lfp_scale);
title('(2) Filtered LFP (5–60 Hz) in µV');
xlabel('Time (s)'); ylabel('Amplitude (µV)');
ylim([-2500 2500]); xlim([x_start, x_end]);

% (3) SWD detection
axSTR(2) = subplot('Position', pos_3);
plot(swd_t, corrected_baseline, 'Color', [0.6 0.6 0.6]); hold on;
for ii = 1:numel(swd_events)
    evt = swd_events{ii};
    te  = evt / original_sampling_rate;
    plot(te, corrected_baseline(evt), 'r', 'LineWidth', 2);
end

for ii = 1:size(swd_events_ext_1s,1)
    idx_range = swd_events_ext_1s(ii,1):swd_events_ext_1s(ii,2);
    plot(idx_range / fs0, corrected_baseline(idx_range), 'g', 'LineWidth', 2);
end

hold off;
title('(3) SWD Detection'); xlabel('Time (s)'); ylabel('Amplitude');
ylim([-20000 20000]); xlim([x_start, x_end]);

% (5) Raster — STR only
axSTR(3) = subplot('Position', pos_5); hold on;
if ~isempty(selected_neurons_STR)
    ids_local = neuron_id(selected_neurons_STR);
    for i = 1:numel(selected_neurons_STR)
        r  = selected_neurons_STR(i);
        st = resultArray_wt(r,:); st = st(st>0);
        scatter(st, i*ones(size(st)), 10, '.');
    end
    yticks(1:numel(selected_neurons_STR)); yticklabels(ids_local);
    ylim([0.5, numel(selected_neurons_STR)+0.5]);
else
    text(mean([x_start x_end]), 0.5, 'No STR neurons selected', 'HorizontalAlignment','center');
    ylim([0 1]);
end
title('(5) Raster — Striatum'); xlabel('Time (s)'); ylabel('Neuron ID');
xlim([x_start, x_end]); hold off;

% (6) Population rate — STR only (0.1 s bins)
axSTR(4) = subplot('Position', pos_6);
bin_width = 0.01;
tb = x_start:bin_width:x_end;
sel_spikes = [];
for i = 1:numel(selected_neurons_STR)
    r  = selected_neurons_STR(i);
    st = resultArray_wt(r,:); st = st(st>0);
    sel_spikes = [sel_spikes, st]; %#ok<AGROW>
end
if ~isempty(sel_spikes)
    binned = histcounts(sel_spikes, tb) / bin_width; % spikes/s
    bar(tb(1:end-1), binned, 'k');
else
    bar(0,0,'k'); xlim([x_start, x_end]);
end
title(sprintf('(6) Population rate (%.1f s bins) — Striatum', bin_width));
xlabel('Time (s)'); ylabel('Spikes/s'); xlim([x_start, x_end]);

linkaxes(axSTR, 'x');
% --- pretty x-axis formatting (more decimals) ---
digits = 3;                           % <- choose how many decimals you want
fmt = sprintf('%%.%df', digits);
ax = gca;
xtickformat(ax, fmt);                  % R2016b+ (recommended)

% If you want fixed tick spacing too (optional):
% xticks(x_start : 1 : x_end);        % 1 s ticks; change as you like

% --- make data tips show full precision (newer MATLAB) ---
hBar = findobj(ax, 'Type','Bar');
if ~isempty(hBar) && isprop(hBar,'DataTipTemplate')
    hBar.DataTipTemplate.DataTipRows(1).Label  = 'Time (s)';
    hBar.DataTipTemplate.DataTipRows(1).Format = fmt;   % X
    hBar.DataTipTemplate.DataTipRows(2).Label  = 'Spikes/s';
    hBar.DataTipTemplate.DataTipRows(2).Format = '%.0f';% Y
end




% ==================== Figure: CORTEX ====================
figCTX = figure('Name','Region composite — Cortex (2,3,5,6)','Position',[140 40 1200 820]);
axCTX = gobjects(4,1);

% (2) Filtered LFP
axCTX(1) = subplot('Position', pos_2);
plot(lfp_t, lfp_filt*lfp_scale);
title('(2) Filtered LFP (5–60 Hz) in µV');
xlabel('Time (s)'); ylabel('Amplitude (µV)');
ylim([-2500 2500]); xlim([x_start, x_end]);

% (3) SWD detection
axCTX(2) = subplot('Position', pos_3);
plot(swd_t, corrected_baseline, 'Color', [0.6 0.6 0.6]); hold on;
for ii = 1:numel(swd_events)
    evt = swd_events{ii};
    te  = evt / original_sampling_rate;
    plot(te, corrected_baseline(evt), 'r', 'LineWidth', 2);
end
hold off;
title('(3) SWD Detection'); xlabel('Time (s)'); ylabel('Amplitude');
ylim([-20000 20000]); xlim([x_start, x_end]);

% (5) Raster — CTX only
axCTX(3) = subplot('Position', pos_5); hold on;
if ~isempty(selected_neurons_CTX)
    ids_local = neuron_id(selected_neurons_CTX);
    for i = 1:numel(selected_neurons_CTX)
        r  = selected_neurons_CTX(i);
        st = resultArray_wt(r,:); st = st(st>0);
        scatter(st, i*ones(size(st)), 10, '.');
    end
    yticks(1:numel(selected_neurons_CTX)); yticklabels(ids_local);
    ylim([0.5, numel(selected_neurons_CTX)+0.5]);
else
    text(mean([x_start x_end]), 0.5, 'No CTX neurons selected', 'HorizontalAlignment','center');
    ylim([0 1]);
end
title('(5) Raster — Cortex'); xlabel('Time (s)'); ylabel('Neuron ID');
xlim([x_start, x_end]); hold off;

% (6) Population rate — CTX only (0.1 s bins)
axCTX(4) = subplot('Position', pos_6);
sel_spikes = [];
for i = 1:numel(selected_neurons_CTX)
    r  = selected_neurons_CTX(i);
    st = resultArray_wt(r,:); st = st(st>0);
    sel_spikes = [sel_spikes, st]; %#ok<AGROW>
end
if ~isempty(sel_spikes)
    binned = histcounts(sel_spikes, tb) / bin_width; % spikes/s
    bar(tb(1:end-1), binned, 'k');
else
    bar(0,0,'k'); xlim([x_start, x_end]);
end
title(sprintf('(6) Population rate (%.1f s bins) — Cortex', bin_width));
xlabel('Time (s)'); ylabel('Spikes/s'); xlim([x_start, x_end]);

linkaxes(axCTX, 'x');


% --- pretty x-axis formatting (more decimals) ---
digits = 3;                           % <- choose how many decimals you want
fmt = sprintf('%%.%df', digits);
ax = gca;
xtickformat(ax, fmt);                  % R2016b+ (recommended)

% If you want fixed tick spacing too (optional):
% xticks(x_start : 1 : x_end);        % 1 s ticks; change as you like

% --- make data tips show full precision (newer MATLAB) ---
hBar = findobj(ax, 'Type','Bar');
if ~isempty(hBar) && isprop(hBar,'DataTipTemplate')
    hBar.DataTipTemplate.DataTipRows(1).Label  = 'Time (s)';
    hBar.DataTipTemplate.DataTipRows(1).Format = fmt;   % X
    hBar.DataTipTemplate.DataTipRows(2).Label  = 'Spikes/s';
    hBar.DataTipTemplate.DataTipRows(2).Format = '%.0f';% Y
end



%% --- Build region population rates for peak-finding (STR & CTX) ---
% Reuse bin_width and tb from your subplot(6). If missing, define them now.
if ~exist('bin_width','var') || isempty(bin_width), bin_width = 0.01; end
if ~exist('tb','var') || isempty(tb)
    if ~exist('x_start','var'), x_start = 0; end
    if ~exist('x_end','var')   || ~isfinite(x_end), x_end = max(resultArray_wt(:)); end
    tb = x_start:bin_width:x_end;
end
tc = tb(1:end-1) + bin_width/2;   % bin centers for peak timing

% Collect spike times (seconds) for each region (row indices in resultArray_wt)
build_sel_spikes = @(rows) cell2mat( ...
    arrayfun(@(r) nonzeros(resultArray_wt(r,:))', rows(:), 'UniformOutput', false) );

sel_spikes_STR = build_sel_spikes(selected_neurons_STR);
sel_spikes_CTX = build_sel_spikes(selected_neurons_CTX);

% Histogram -> spikes/s
binned_STR = histcounts(sel_spikes_STR, tb) / bin_width;
binned_CTX = histcounts(sel_spikes_CTX, tb) / bin_width;


%% ---- Peaks of population rate within extended SWD windows (STR & CTX) ----
% --- Per-window peaks inside 1s-extended SWD windows (STR & CTX) ---

% tb = bin edges (1xM), tc = centers (1xM-1), binned_* = spikes/s in each bin
% If you don't have tc yet:
% tc = tb(1:end-1) + bin_width/2;

ext_sec = swd_events_ext_1s / original_sampling_rate;  % [K x 2] in seconds
K = size(ext_sec,1);

% Preallocate
PeakInfo_STR(K,1) = struct('t1',NaN,'t2',NaN,'t_peak',NaN,'v_peak',NaN);
PeakInfo_CTX      = PeakInfo_STR;

% Use **edge overlap** to decide which bins belong to a window:
% a bin [tb(i), tb(i+1)] belongs if it overlaps [t1, t2]
for k = 1:K
    t1 = ext_sec(k,1);  t2 = ext_sec(k,2);
    PeakInfo_STR(k).t1 = t1;  PeakInfo_STR(k).t2 = t2;
    PeakInfo_CTX(k).t1 = t1;  PeakInfo_CTX(k).t2 = t2;

    % bins that overlap the window
    bin_in_win = (tb(1:end-1) < t2) & (tb(2:end) > t1);
    if any(bin_in_win)
        % STR
        [vS, iSrel] = max(binned_STR(bin_in_win));
        iS = find(bin_in_win, 1, 'first') + iSrel - 1;
        PeakInfo_STR(k).t_peak = tc(iS);
        PeakInfo_STR(k).v_peak = vS;
        % CTX
        [vC, iCrel] = max(binned_CTX(bin_in_win));
        iC = find(bin_in_win, 1, 'first') + iCrel - 1;
        PeakInfo_CTX(k).t_peak = tc(iC);
        PeakInfo_CTX(k).v_peak = vC;
    else
        % Fallback: pick nearest bin to window center
        tmid = (t1+t2)/2;
        [~, iNear] = min(abs(tc - tmid));
        PeakInfo_STR(k).t_peak = tc(iNear);
        PeakInfo_STR(k).v_peak = binned_STR(iNear);
        PeakInfo_CTX(k).t_peak = tc(iNear);
        PeakInfo_CTX(k).v_peak = binned_CTX(iNear);
    end
end

% ---------- Overlay ALL peak markers on subplot (6) ----------
% Use saved handles if you have them; otherwise find axes by figure name.
if exist('axSTR','var') && numel(axSTR)>=4 && isvalid(axSTR(4)), axS = axSTR(4);
else
    fS  = findobj('Type','figure','Name','Region composite — Striatum (2,3,5,6)');
    axS = findobj(fS,'Type','axes'); axS = axS(1);
end
if exist('axCTX','var') && numel(axCTX)>=4 && isvalid(axCTX(4)), axC = axCTX(4);
else
    fC  = findobj('Type','figure','Name','Region composite — Cortex (2,3,5,6)');
    axC = findobj(fC,'Type','axes'); axC = axC(1);
end

% Plot all peaks (small markers to avoid clutter). Label a few as sanity check.
hold(axS,'on');
plot(axS, [PeakInfo_STR.t_peak], [PeakInfo_STR.v_peak], '.', 'MarkerSize',10, 'Color',[0 0.6 0]);
hold(axS,'off');

hold(axC,'on');
plot(axC, [PeakInfo_CTX.t_peak], [PeakInfo_CTX.v_peak], '.', 'MarkerSize',10, 'Color',[0 0.6 0]);
hold(axC,'off');

% (Optional) annotate every Nth window to keep it readable:
Nlabel = 25;   % change or set [] to skip labels
if ~isempty(Nlabel)
    ks = 1:Nlabel:K;
    for ii = ks
        text(PeakInfo_STR(ii).t_peak, PeakInfo_STR(ii).v_peak, sprintf(' #%d',ii), ...
             'Parent',axS,'VerticalAlignment','bottom','Color',[0 0.5 0]);
        text(PeakInfo_CTX(ii).t_peak, PeakInfo_CTX(ii).v_peak, sprintf(' #%d',ii), ...
             'Parent',axC,'VerticalAlignment','bottom','Color',[0 0.5 0]);
    end
end

% Quick console sanity check
nSTR = sum(isfinite([PeakInfo_STR.v_peak]));
nCTX = sum(isfinite([PeakInfo_CTX.v_peak]));
fprintf('[Per-window peaks] Found %d/%d STR peaks and %d/%d CTX peaks.\n', nSTR, K, nCTX, K);

%%
% ===== Export green-window starts + CTX/STR peak times to Excel =====
% Prereqs (already created earlier): swd_events_ext_1s, original_sampling_rate,
% PeakInfo_CTX (with .t_peak), PeakInfo_STR (with .t_peak)

fs0 = original_sampling_rate;                     % e.g., 2500 Hz
ext_sec = swd_events_ext_1s / fs0;                % [K x 2], seconds
K = size(ext_sec,1);

% Extract per-window peak times (seconds)
ctx_t = nan(K,1);  str_t = nan(K,1);
if exist('PeakInfo_CTX','var') && numel(PeakInfo_CTX) == K
    ctx_t = reshape([PeakInfo_CTX.t_peak],[],1);
end
if exist('PeakInfo_STR','var') && numel(PeakInfo_STR) == K
    str_t = reshape([PeakInfo_STR.t_peak],[],1);
end

start_sec = ext_sec(:,1);                         % green-line start (s)
M = [start_sec, ctx_t, str_t];                    % N×3 matrix

% ---- Create folder (if not exists) ----
out_dir = fullfile(pwd, 'Str_Cor_histPeak_diff');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% ---- Unique filename with timestamp to avoid overwrite ----
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
out_xlsx = fullfile(out_dir, ['swd_peak_times_ctx_str_' timestamp '.xlsx']);

% ---- Save as Excel with headers ----
T = table(start_sec, ctx_t, str_t, ...
    'VariableNames', {'start_sec','ctx_t_peak','str_t_peak'});
writetable(T, out_xlsx);

fprintf('[OK] Saved %d rows to: %s\n', K, out_xlsx);
disp('First 5 rows:');
disp(T(1:min(5,K),:));



%% 10/10/2025 plot individual histoplot 
% ---- Safety: rebuild tc if missing; infer fs0 if needed
if ~exist('tc','var') || isempty(tc)
    if ~exist('tb','var') || isempty(tb)
        error('Need tb (bin edges) to compute tc. Define tb before running this block.');
    end
    if ~exist('bin_width','var') || isempty(bin_width)
        bin_width = mean(diff(tb));
    end
    tc = tb(1:end-1) + bin_width/2;
end
if ~exist('original_sampling_rate','var') || isempty(original_sampling_rate)
    if exist('fs0','var') && ~isempty(fs0)
        original_sampling_rate = fs0;
    else
        error('original_sampling_rate (fs0) is required.');
    end
end

% ---- 内嵌 helper 函数 (替代 function iff)
iff = @(cond,a,b) (cond).*a + (~cond).*b; % inline builtin

% ---- Collect onsets (seconds) from swd_events (original windows, not extended)
if ~iscell(swd_events) || isempty(swd_events)
    warning('swd_events is empty or not a cell; skipping per-event summary.');
else
    fs0 = original_sampling_rate;
    try
        onset_samples = cellfun(@(v) v(1), swd_events(:));   % first index per event
    catch
        % If any cell is empty, skip those
        onset_samples = cellfun(@(v) iff(isempty(v), NaN, v(1)), swd_events(:));
        onset_samples = onset_samples(~isnan(onset_samples));
    end
    onset_sec = onset_samples / fs0;                          % seconds
end

%%
% ===== Match real onsets against T_merged windows (keep==1) =====
% Requires:
%   T_merged : table with columns t_start_s, t_end_s, (optional) keep
% Notes:
%   We intentionally IGNORE swd_events_seconds and instead use T_merged windows,
%   because T_merged has the merged (possibly corrected) times.

% ---- collect onsets (seconds) ----
if ~ismember('t_start_s', T_merged.Properties.VariableNames) || ...
   ~ismember('t_end_s',   T_merged.Properties.VariableNames)
    error('T_merged must contain t_start_s and t_end_s.');
end

if ismember('keep', T_merged.Properties.VariableNames)
    Tuse = T_merged(T_merged.keep==1 & isfinite(T_merged.t_start_s) & isfinite(T_merged.t_end_s), :);
else
    Tuse = T_merged(isfinite(T_merged.t_start_s) & isfinite(T_merged.t_end_s), :);
end

% guard: remove inverted windows
Tuse = Tuse(Tuse.t_end_s >= Tuse.t_start_s, :);

% onsets to test (also from T_merged, after keep-filter)
s = Tuse.t_start_s(:);                 % 用合并后的起点作为“真实起点候选”
s = s(isfinite(s));

% build windows from T_merged directly
A = Tuse.t_start_s(:);
B = Tuse.t_end_s(:);

% half-sample margin if fs known
if exist('fs','var') && ~isempty(fs)
    eps_margin = 0.5/fs;
elseif exist('original_sampling_rate','var') && ~isempty(original_sampling_rate)
    eps_margin = 0.5/original_sampling_rate;
else
    eps_margin = 1e-6;  % tiny numeric tolerance
end

% vectorized inclusion test: s in any [A,B]
% To keep simple and robust, do a loop over windows (windows count is usually manageable)
keep = false(size(s));
for k = 1:numel(A)
    a = A(k); b = B(k);
    keep = keep | (s >= (a - eps_margin) & s <= (b + eps_margin));
end

swd_real_onset_FixMiss = unique(s(keep), 'sorted');

% quick diagnostics
fprintf('T_merged windows (keep==1): %d\n', numel(A));
fprintf('Onset candidates tested   : %d\n', numel(s));
fprintf('Matched onsets            : %d\n', numel(swd_real_onset_FixMiss));

% ---- save (no-overwrite) ----
out_dir = fullfile(pwd, 'Str_Cor_histPeak_diff');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
mat_path = fullfile(out_dir, ['swd_real_onset_FixMiss_' timestamp '.mat']);
xlsx_path = fullfile(out_dir, ['swd_real_onset_FixMiss_' timestamp '.xlsx']);

assignin('base','swd_real_onset_FixMiss', swd_real_onset_FixMiss);
save(mat_path, 'swd_real_onset_FixMiss');
writematrix(swd_real_onset_FixMiss, xlsx_path);

fprintf('[OK] Saved %d onset times to:\n  %s\n  %s\n', ...
    numel(swd_real_onset_FixMiss), mat_path, xlsx_path);

% ---- optional: if you still want to sanity-check why old windows miss ----
% Uncomment to print a few diffs versus swd_events_seconds (if present)
% if exist('swd_events_seconds','var') && ~isempty(swd_events_seconds)
%     k = find(~keep, 1, 'first');
%     if ~isempty(k)
%         s0 = s(k);
%         % find nearest T_merged window
%         [~,i_near] = min(abs((A+B)/2 - s0));
%         fprintf('Example unmatched onset: %.6f s  | nearest T window [%.6f, %.6f]\n',...
%                 s0, A(i_near), B(i_near));
%         % check one old window if needed
%         if ~isempty(swd_events_seconds)
%             w = swd_events_seconds{min(i_near, numel(swd_events_seconds))};
%             if ~isempty(w)
%                 fprintf('Old swd_events_seconds win approx [%.6f, %.6f] (assuming in seconds).\n', ...
%                         min(w), max(w));
%             end
%         end
%     end
% end


%%
% ---- How many events to preview
if ~exist('onset_sec','var') || isempty(onset_sec)
    warning('No valid SWD onsets found.'); 
else
    dur_post = 2.0;                 % 2 seconds after onset
    % ---- 选取事件范围：第 50 到 60 个 ----
    range = 20;
    range = range(range <= numel(onset_sec));  % 防止越界
    onset_sec = onset_sec(range);              % 只保留 50–60 的 onsets
    n_show = numel(onset_sec);                 % 显示这些事件的全部


    

   
    %% ========== (B) 新 0.01 s bin histogram：CTX 与 STR 分图显示 ==========
bin_width_hist = 0.01;          % 0.01 s bins  ← 你之前写成0.001了，这里按0.01
edges_rel      = 0:bin_width_hist:dur_post;
centers_rel    = edges_rel(1:end-1) + bin_width_hist/2;   % NEW: bin中心（用于峰值定位）

% helper to gather spikes
get_spike_times = @(rows) cell2mat( ...
    arrayfun(@(r) nonzeros(resultArray_wt(r,:))', rows(:), 'UniformOutput', false));

spk_CTX_all = get_spike_times(selected_neurons_CTX);
spk_STR_all = get_spike_times(selected_neurons_STR);

% 每个事件2行：CTX(上) + STR(下)
figH = figure('Name','Onset-aligned histograms (0.01 s bins, CTX vs STR split)','Position',[150 60 1200 1000]);
tiledlayout(n_show*2,1,'TileSpacing','compact','Padding','compact');

for k = 1:n_show
    t0 = onset_sec(k);
    in_CTX = (spk_CTX_all >= t0) & (spk_CTX_all <= t0 + dur_post);
    in_STR = (spk_STR_all >= t0) & (spk_STR_all <= t0 + dur_post);

    rel_CTX = spk_CTX_all(in_CTX) - t0;
    rel_STR = spk_STR_all(in_STR) - t0;

    cnt_CTX = histcounts(rel_CTX, edges_rel);
    cnt_STR = histcounts(rel_STR, edges_rel);
    rate_CTX = cnt_CTX / bin_width_hist;
    rate_STR = cnt_STR / bin_width_hist;

    % === 计算峰值（基于bin中心）===  NEW
    [v_ctx, i_ctx] = max(rate_CTX);
    [v_str, i_str] = max(rate_STR);
    tpk_ctx = centers_rel(i_ctx);
    tpk_str = centers_rel(i_str);

    % ---- 第一行：Cortex ----
    ax1 = nexttile; hold(ax1,'on');
    bar(ax1, edges_rel(1:end-1), rate_CTX, 1.0, 'FaceColor',[0.2 0.2 0.2], 'EdgeColor','none');
    % 标峰：竖线 + 圆点 + 文本   NEW
    xline(ax1, tpk_ctx, '-', 'Color',[0 0.45 0.74], 'LineWidth',1.2);
    plot(ax1, tpk_ctx, v_ctx, 'o', 'MarkerSize',6, 'MarkerFaceColor',[0 0.45 0.74], 'MarkerEdgeColor','w');
    text(tpk_ctx, v_ctx, sprintf('  peak %.3fs', tpk_ctx), 'Color',[0 0.45 0.74], ...
         'VerticalAlignment','bottom','FontSize',9);
    xlim(ax1,[0 dur_post]);
    xlabel(ax1,'Time since onset (s)');
    ylabel(ax1,'Spikes/s');
    title(ax1, sprintf('CORTEX | SWD #%d (t0 = %.3f s)', k, t0));
    grid(ax1,'on'); box(ax1,'off');
    hold(ax1,'off');

    % ---- 第二行：Striatum ----
    ax2 = nexttile; hold(ax2,'on');
    bar(ax2, edges_rel(1:end-1), rate_STR, 1.0, 'FaceColor',[0.6 0.6 0.6], 'EdgeColor','none');
    % 标峰：竖线 + 圆点 + 文本   NEW
    xline(ax2, tpk_str, '-', 'Color',[0.85 0.33 0.10], 'LineWidth',1.2);
    plot(ax2, tpk_str, v_str, 'o', 'MarkerSize',6, 'MarkerFaceColor',[0.85 0.33 0.10], 'MarkerEdgeColor','w');
    text(tpk_str, v_str, sprintf('  peak %.3fs', tpk_str), 'Color',[0.85 0.33 0.10], ...
         'VerticalAlignment','bottom','FontSize',9);
    xlim(ax2,[0 dur_post]);
    xlabel(ax2,'Time since onset (s)');
    ylabel(ax2,'Spikes/s');
    title(ax2, sprintf('STRIATUM | SWD #%d (t0 = %.3f s)', k, t0));
    grid(ax2,'on'); box(ax2,'off');
    hold(ax2,'off');
end


end

%% === Build & export onset-aligned rates for ALL events (CTX & STR) ===
% Uses: swd_events, original_sampling_rate, resultArray_wt,
%       selected_neurons_CTX, selected_neurons_STR
% Produces workspace vars:
%   rate_CTX_all (K x M), rate_STR_all (K x M), edges_rel (1 x (M+1)), centers_rel (1 x M)
% And writes to Excel sheets:
%   Sheet 'CTX_rate'  -> rate_CTX_all
%   Sheet 'STR_rate'  -> rate_STR_all
%   Sheet 'edges_rel' -> edges_rel (single row)

% ---- params (follow your histogram settings) ----
dur_post       = 2.0;          % seconds after onset
bin_width_hist = 0.01;         % 10 ms bins
edges_rel      = 0:bin_width_hist:dur_post;
centers_rel    = edges_rel(1:end-1) + bin_width_hist/2;
M              = numel(edges_rel)-1;

% ---- spike pools (absolute time, seconds) ----
get_spike_times = @(rows) cell2mat( ...
    arrayfun(@(r) nonzeros(resultArray_wt(r,:))', rows(:), 'UniformOutput', false));
spk_CTX_all = get_spike_times(selected_neurons_CTX);
spk_STR_all = get_spike_times(selected_neurons_STR);

% ---- all onsets in seconds (do NOT use the earlier 'range' filter) ----
% fs0 = original_sampling_rate;
% onset_samples_all = cellfun(@(v) v(1), swd_events(:));   % first index per event (skip empties below)
% onset_samples_all = onset_samples_all(~cellfun(@isempty, swd_events(:))); % 防空
% K = numel(onset_samples_all);
% onset_sec_all = onset_samples_all / fs0;
onset_sec_all = swd_real_onset_FixMiss;
% ---- preallocate outputs ----
K = numel(swd_real_onset_FixMiss);

rate_CTX_all = zeros(K, M);
rate_STR_all = zeros(K, M);

% ---- compute per-event histograms -> spikes/s ----
for k = 1:K
    t0 = onset_sec_all(k);

    % window filter by absolute time, then convert to relative
    in_CTX = (spk_CTX_all >= t0) & (spk_CTX_all <= t0 + dur_post);
    in_STR = (spk_STR_all >= t0) & (spk_STR_all <= t0 + dur_post);

    rel_CTX = spk_CTX_all(in_CTX) - t0;
    rel_STR = spk_STR_all(in_STR) - t0;

    cnt_CTX = histcounts(rel_CTX, edges_rel);
    cnt_STR = histcounts(rel_STR, edges_rel);

    rate_CTX_all(k, :) = cnt_CTX / bin_width_hist;   % spikes/s (population total)
    rate_STR_all(k, :) = cnt_STR / bin_width_hist;
end

% ---- (optional) convert to per-neuron average instead of population total ----
% rate_CTX_all = rate_CTX_all / max(1, numel(selected_neurons_CTX));
% rate_STR_all = rate_STR_all / max(1, numel(selected_neurons_STR));

% ---- create output folder & unique filenames (no overwrite) ----
out_dir = fullfile(pwd, 'Str_Cor_histPeak_diff');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
base_name = fullfile(out_dir, ['swd_onset_aligned_rates_' timestamp]);
out_xlsx  = [base_name '.xlsx'];
out_mat   = [base_name '.mat'];

% ---- write to Excel (3 sheets) ----
writematrix(rate_CTX_all, out_xlsx, 'Sheet', 'CTX_rate',  'WriteMode','overwrite');
writematrix(rate_STR_all, out_xlsx, 'Sheet', 'STR_rate',  'WriteMode','overwrite');
writematrix(edges_rel,    out_xlsx, 'Sheet', 'edges_rel', 'WriteMode','overwrite');

% ---- also save a .mat snapshot with all key outputs ----
save(out_mat, 'rate_CTX_all','rate_STR_all','edges_rel','centers_rel', ...
              'bin_width_hist','dur_post','onset_sec_all', ...
              'selected_neurons_CTX','selected_neurons_STR');

fprintf('[OK] Exported %d events, %d bins to:\n  %s\n  %s\n', K, M, out_xlsx, out_mat);

