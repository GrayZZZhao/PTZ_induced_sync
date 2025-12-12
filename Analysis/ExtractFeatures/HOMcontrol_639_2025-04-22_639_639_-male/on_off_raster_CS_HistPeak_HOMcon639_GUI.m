

%%   找id
% ---- Verify neuron_id vs clustertwo.xlsx col 2 (+1) ----
% Assumes you already built resultArray_wt, so:
% neuron_id = resultArray_wt(:,1);

excel_path = 'D:\npxl_kv11\2025-04-22_639_639_HOM_kv11control-male-adult\2025-04-22_15-45-09\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\clustertwo.xlsx';
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

excel_path = 'D:\npxl_kv11\2025-04-22_639_639_HOM_kv11control-male-adult\2025-04-22_15-45-09\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\clustertwo.xlsx';
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
% extend_sec = 1;                                   % <-- change this if you want
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

extend_sec = 0.5;                                   % <-- change this if you want
fs0 = original_sampling_rate;                     % 2500 in your script
N   = numel(corrected_baseline);
extN = round(extend_sec * fs0);

% bounds for each event, then extend and clip to [1, N]
evt_bounds = nan(numel(swd_events),2);
for i = 1:numel(swd_events)
    v = swd_events{i};
    if ~isempty(v)
        evt_bounds(i,:) = [v(1) v(end)];
    end
end
evt_bounds = evt_bounds(~isnan(evt_bounds(:,1)),:);

% ✅ only extend the *end* of each event
evt_ext = [ evt_bounds(:,1),  min(N, evt_bounds(:,2) + extN) ];

% merge overlapping extended intervals
evt_ext = sortrows(evt_ext,1);
merged_ext = [];
for k = 1:size(evt_ext,1)
    if isempty(merged_ext) || evt_ext(k,1) > merged_ext(end,2) + 1
        merged_ext = [merged_ext; evt_ext(k,:)]; %#ok<AGROW>
    else
        merged_ext(end,2) = max(merged_ext(end,2), evt_ext(k,2));
    end
end

swd_events_ext_1s = merged_ext;                   % [K x 2], sample index start/end



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

%%

%% ===================== DROP-IN GUI: CTX/STR per-event viewer (smooth-based peak detection + 7 panels, resized + green labels) =====================
% ===================== DROP-IN GUI: CTX/STR per-event viewer (smooth-based peak detection + 7 panels, resized + green labels) =====================

smooth_win = 3;  % 平滑窗口大小（单位：bin 数）；设为 1 表示不平滑

% ---------- 1) Prompt: depth threshold for CTX vs STR ----------
if ~exist('depth_um','var') || ~exist('neuron_id','var')
    error('depth_um / neuron_id not found. Run the Excel depth-loading step first.');
end

answer = inputdlg( ...
    {'Depth threshold (µm) separating Striatum (< thr) and Cortex (>= thr):'}, ...
    'Region boundary', [1 60], {'2625'});
if isempty(answer)
    disp('User cancelled depth threshold dialog.'); 
    return;
end
depth_thr = str2double(answer{1});
if isnan(depth_thr)
    error('Invalid depth threshold.');
end
fprintf('Using depth threshold = %.1f µm (STR: < thr, CTX: >= thr).\n', depth_thr);

% ---------- 2) Region label & neuron split ----------
is_STR = depth_um < depth_thr;
is_CTX = ~is_STR;
region = repmat("cortex", numel(depth_um), 1);
region(is_STR) = "striatum";
neuron_id_STR = neuron_id(is_STR);
neuron_id_CTX = neuron_id(is_CTX);

if ~exist('selected_neurons','var') || isempty(selected_neurons)
    selected_neurons = (1:size(resultArray_wt,1)).';
else
    selected_neurons = selected_neurons(:);
end
selected_neurons_STR = selected_neurons(is_STR(selected_neurons));
selected_neurons_CTX = selected_neurons(is_CTX(selected_neurons));

RegionTable = table( ...
    neuron_id(:), depth_um(:), categorical(region), ...
    'VariableNames', {'neuron_id','depth_um','region'});
fprintf('Region counts -> STR: %d, CTX: %d (total %d)\n', ...
    nnz(is_STR), nnz(is_CTX), numel(is_STR));

% ---------- 3) Build population histograms & per-window peaks ----------
if ~exist('swd_events_ext_1s','var') || isempty(swd_events_ext_1s)
    error('swd_events_ext_1s not found. Make sure merged_ext has been built.');
end
if ~exist('original_sampling_rate','var') || isempty(original_sampling_rate)
    error('original_sampling_rate is required.');
end
fs0     = original_sampling_rate;
ext_sec = swd_events_ext_1s / fs0;   % [K x 2] s
K       = size(ext_sec,1);

% 全局时间范围
if exist('x_start','var') && exist('x_end','var') && ~isempty(x_start) && ~isempty(x_end)
    t_min = x_start; t_max = x_end;
else
    t_min = 0;
    t_max = max(resultArray_wt(:));
end

% 0.01 s bins
bin_width = 0.01;
tb = t_min:bin_width:t_max;
tc = tb(1:end-1) + bin_width/2;

% spike 提取 helper（整体 population 用，保留）
get_spike_times = @(rows) cell2mat( ...
    arrayfun(@(r) nonzeros(resultArray_wt(r,:))', rows(:), 'UniformOutput', false));

sel_spikes_STR = get_spike_times(selected_neurons_STR);
sel_spikes_CTX = get_spike_times(selected_neurons_CTX);

% 原始 population rate
binned_STR_raw = histcounts(sel_spikes_STR, tb) / bin_width;
binned_CTX_raw = histcounts(sel_spikes_CTX, tb) / bin_width;

% ✅ 平滑后的 population rate（用于 peak 计算和绘图）
if smooth_win > 1
    binned_STR = smooth(binned_STR_raw, smooth_win);
    binned_CTX = smooth(binned_CTX_raw, smooth_win);
else
    binned_STR = binned_STR_raw;
    binned_CTX = binned_CTX_raw;
end

% --- 初始化结构体 ---
PeakInfo_STR = repmat(struct('t1',NaN,'t2',NaN,'t_peak',NaN,'v_peak',NaN), K, 1);
PeakInfo_CTX = PeakInfo_STR;

% --- 每个 SWD 事件求基于平滑的自动 peak ---
for k = 1:K
    t1 = ext_sec(k,1); t2 = ext_sec(k,2);
    PeakInfo_STR(k).t1 = t1; PeakInfo_STR(k).t2 = t2;
    PeakInfo_CTX(k).t1 = t1; PeakInfo_CTX(k).t2 = t2;
    bin_in_win = (tb(1:end-1) < t2) & (tb(2:end) > t1);
    if any(bin_in_win)
        % STR
        [vS,iSrel] = max(binned_STR(bin_in_win));
        iS = find(bin_in_win,1,'first') + iSrel - 1;
        PeakInfo_STR(k).t_peak = tc(iS); 
        PeakInfo_STR(k).v_peak = binned_STR(iS);
        % CTX
        [vC,iCrel] = max(binned_CTX(bin_in_win));
        iC = find(bin_in_win,1,'first') + iCrel - 1;
        PeakInfo_CTX(k).t_peak = tc(iC); 
        PeakInfo_CTX(k).v_peak = binned_CTX(iC);
    else
        % fallback: 最接近事件中心的 bin
        tmid = (t1+t2)/2; 
        [~,iNear] = min(abs(tc-tmid));
        PeakInfo_STR(k).t_peak = tc(iNear); 
        PeakInfo_STR(k).v_peak = binned_STR(iNear);
        PeakInfo_CTX(k).t_peak = tc(iNear); 
        PeakInfo_CTX(k).v_peak = binned_CTX(iNear);
    end
end
fprintf('[Per-window peaks after smoothing] STR peaks: %d, CTX peaks: %d (K=%d events)\n', ...
    sum(isfinite([PeakInfo_STR.v_peak])), sum(isfinite([PeakInfo_CTX.v_peak])), K);

% 准备自动 peak 向量 & 手动编辑记录
auto_ctx_t = reshape([PeakInfo_CTX.t_peak],[],1);
auto_ctx_v = reshape([PeakInfo_CTX.v_peak],[],1);
auto_str_t = reshape([PeakInfo_STR.t_peak],[],1);
auto_str_v = reshape([PeakInfo_STR.v_peak],[],1);

edit_ctx_flag = false(K,1);
edit_ctx_t    = nan(K,1);
edit_ctx_v    = nan(K,1);

edit_str_flag = false(K,1);
edit_str_t    = nan(K,1);
edit_str_v    = nan(K,1);

% ---------- 3b) 预计算每个 neuron 的 spike times（用于 raster，加速） ----------
% CTX
spike_times_CTX = cell(numel(selected_neurons_CTX),1);
for ii = 1:numel(selected_neurons_CTX)
    r = selected_neurons_CTX(ii);
    spike_times_CTX{ii} = nonzeros(resultArray_wt(r,:))';  % 只做一次
end
% STR
spike_times_STR = cell(numel(selected_neurons_STR),1);
for ii = 1:numel(selected_neurons_STR)
    r = selected_neurons_STR(ii);
    spike_times_STR{ii} = nonzeros(resultArray_wt(r,:))';
end

% ---------- 4) Build GUI ----------
swd_t = (1:length(corrected_baseline)) / fs0;
t_min = tb(1); t_max = tb(end);

hFig = figure('Name','SWD event viewer — CTX & STR (smooth-based peaks + 7 panels)', ...
              'Position',[150 40 1300 1100], ...
              'CloseRequestFcn', @on_close);  % <- 关闭时保存一次 quality

% ==== 自定义 7 个 panel 的高度（1–4 缩小，5–6 放大） ====
h = [0.07 0.07 0.08 0.08 0.14 0.14 0.18];
top_margin = 0.04;
gap = 0.01;
left = 0.08;
width = 0.9;

y = zeros(1,7);
curr_top = 1 - top_margin;
for i = 1:7
    y(i) = curr_top - h(i);
    curr_top = y(i) - gap;
end

ax = gobjects(7,1);
for i = 1:7
    ax(i) = axes('Parent',hFig, ...
                 'Position',[left, y(i), width, h(i)]);
end

% 底部控件：slider + 按钮
uicontrol('Style','text','Units','normalized','Position',[0.02 0.01 0.12 0.03], ...
    'String','Event index:','HorizontalAlignment','left');
hSlider = uicontrol('Style','slider','Units','normalized','Position',[0.14 0.01 0.50 0.03], ...
    'Min',1,'Max',K,'Value',1,'SliderStep',[1/max(1,K-1) 5/max(1,K-1)]);
hPrev   = uicontrol('Style','pushbutton','Units','normalized','Position',[0.66 0.01 0.07 0.03],'String','Prev');
hNext   = uicontrol('Style','pushbutton','Units','normalized','Position',[0.74 0.01 0.07 0.03],'String','Next');
hTxt    = uicontrol('Style','text','Units','normalized','Position',[0.82 0.01 0.16 0.03],'HorizontalAlignment','left');

% 上方按钮
hSetCTX = uicontrol('Style','pushbutton','Units','normalized', ...
    'Position',[0.02 0.055 0.10 0.03],'String','Set CTX peak');
hSetSTR = uicontrol('Style','pushbutton','Units','normalized', ...
    'Position',[0.14 0.055 0.10 0.03],'String','Set STR peak');
hExport = uicontrol('Style','pushbutton','Units','normalized', ...
    'Position',[0.26 0.055 0.14 0.03],'String','Export quality');

% 跳转控件
uicontrol('Style','text','Units','normalized', ...
    'Position',[0.42 0.055 0.09 0.03],'String','Go to t (s):','HorizontalAlignment','left');
hJumpEdit = uicontrol('Style','edit','Units','normalized', ...
    'Position',[0.51 0.055 0.08 0.03], 'String','0');
hJumpBtn  = uicontrol('Style','pushbutton','Units','normalized', ...
    'Position',[0.60 0.055 0.08 0.03],'String','Go');

% 打包结构
S = struct();
S.ext_sec = ext_sec; 
S.K = K; 
S.ax = ax; 
S.slider = hSlider; 
S.text = hTxt;
S.selected_neurons_CTX = selected_neurons_CTX; 
S.selected_neurons_STR = selected_neurons_STR;
S.neuron_id_CTX = neuron_id_CTX; 
S.neuron_id_STR = neuron_id_STR;
S.resultArray_wt = resultArray_wt; 
S.corrected_baseline = corrected_baseline;
S.swd_t = swd_t; 
S.tb = tb; 
S.tc = tc; 
S.binned_CTX = binned_CTX; 
S.binned_STR = binned_STR;
S.binned_CTX_raw = binned_CTX_raw;
S.binned_STR_raw = binned_STR_raw;
S.PeakInfo_CTX = PeakInfo_CTX; 
S.PeakInfo_STR = PeakInfo_STR;
S.auto_ctx_t = auto_ctx_t; 
S.auto_ctx_v = auto_ctx_v; 
S.auto_str_t = auto_str_t; 
S.auto_str_v = auto_str_v;
S.edit_ctx_flag = edit_ctx_flag; 
S.edit_ctx_t = edit_ctx_t; 
S.edit_ctx_v = edit_ctx_v;
S.edit_str_flag = edit_str_flag; 
S.edit_str_t = edit_str_t; 
S.edit_str_v = edit_str_v;
S.curr_idx = 1; 
S.fs0 = fs0; 
S.t_min = t_min; 
S.t_max = t_max; 
S.hJumpEdit = hJumpEdit;
% 预计算的 spike cell，供 raster 使用
S.spike_times_CTX = spike_times_CTX;
S.spike_times_STR = spike_times_STR;

setappdata(hFig,'SWD_GUI_DATA',S);

% 设置回调
hSlider.Callback = @(src,~) local_change_event(hFig,round(get(src,'Value')));
hPrev.Callback   = @(~,~)  local_change_event(hFig,get_current_idx(hFig)-1);
hNext.Callback   = @(~,~)  local_change_event(hFig,get_current_idx(hFig)+1);
hSetCTX.Callback = @(~,~)  local_set_peak_async(hFig,'CTX');
hSetSTR.Callback = @(~,~)  local_set_peak_async(hFig,'STR');
hExport.Callback = @(~,~)  local_export_quality(hFig);
hJumpBtn.Callback= @(~,~)  local_jump_to_time(hFig);

% 初始绘制
local_change_event(hFig,1);

%% ===================== Helper: 读当前 idx =====================
function idx = get_current_idx(hFig)
S = getappdata(hFig,'SWD_GUI_DATA'); 
idx = S.curr_idx;
end

%% ===================== Helper: CloseRequestFcn 自动保存 =====================
function on_close(hFig,~)
try
    % 关闭 GUI 时统一导出一次 qualityMatrix（.mat + .xlsx）
    local_export_quality(hFig);
catch ME
    warning('Auto-save on close failed: %s',ME.message);
end
delete(hFig);
end

%% ===================== Helper: 切换/刷新事件 =====================
function local_change_event(hFig,idx)
S = getappdata(hFig,'SWD_GUI_DATA'); 
K = S.K; 
idx = max(1,min(K,round(idx)));
S.curr_idx = idx; 
setappdata(hFig,'SWD_GUI_DATA',S);
set(S.slider,'Value',idx);

t0 = S.ext_sec(idx,1);     % event 起点 (s)
W  = 8;                    % 总显示窗口长度 (s)
t1 = max(S.t_min, t0 - W/2);
t2 = min(S.t_max, t0 + W/2);

set(S.text,'String',sprintf('Event %d/%d (center %.3f s, window [%.3f,%.3f] s)', ...
    idx,K,t0,t1,t2));

axCWT   = S.ax(1);
axSWD   = S.ax(2);
axCTX   = S.ax(3);
axSTR   = S.ax(4);
axCTXr  = S.ax(5);
axSTRr  = S.ax(6);
axCombo = S.ax(7);   % combined histogram 轴

%% --- (1) CWT ---
axes(axCWT); cla(axCWT);

fs = S.fs0;
x  = S.corrected_baseline(:);

bp = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',1,'HalfPowerFrequency2',60, ...
    'SampleRate',fs);
x_f = filtfilt(bp, x);
x_f = detrend(x_f, 'linear');

ix0 = max(1, floor(t1*fs)+1);
ix1 = min(numel(x_f), ceil(t2*fs));
sig_win = x_f(ix0:ix1);
t_win   = ((ix0:ix1)-1)/fs;

fb = cwtfilterbank('SignalLength',numel(sig_win), 'SamplingFrequency',fs, ...
    'Wavelet','morse','VoicesPerOctave',24,'TimeBandwidth',60, ...
    'FrequencyLimits',[0.5 60]);
[cfs, f] = wt(fb, sig_win);

P    = abs(cfs).^2;
Zlog = log10(P + eps);
mu   = mean(Zlog, 2);
sd   = std(Zlog, 0, 2) + eps;
Zz   = (Zlog - mu) ./ sd;

clim = [-2 4];
imagesc(axCWT, t_win, f, Zz); axis(axCWT,'xy');
xlim(axCWT,[t1 t2]); ylim(axCWT,[0.5 60]);
xlabel(axCWT,'Time (s)'); ylabel(axCWT,'Frequency (Hz)');
title(axCWT,sprintf('CWT (Morse, z-log power) — Event %d',idx));
colormap(axCWT, 'jet'); caxis(axCWT, clim);

%% --- (2) SWD detection ---
axes(axSWD); cla(axSWD);
plot(axSWD,S.swd_t,S.corrected_baseline,'Color',[0.6 0.6 0.6]); hold(axSWD,'on');
e1 = S.ext_sec(idx,1); 
e2 = S.ext_sec(idx,2);
ymin = min(S.corrected_baseline); 
ymax = max(S.corrected_baseline);

patch(axSWD,[e1 e2 e2 e1],[ymin ymin ymax ymax],[0.6 0.9 0.6], ...
      'FaceAlpha',0.4,'EdgeColor','none');

plot(axSWD,S.swd_t,S.corrected_baseline,'k');
xlim(axSWD,[t1 t2]); xlabel(axSWD,'Time (s)'); ylabel(axSWD,'Amplitude');
title(axSWD,sprintf('SWD detection (Event %d)',idx)); 
hold(axSWD,'off');

%% --- (3–4) Raster helpers（使用预计算 spike cell，加速） ---
    function draw_raster(ax, spike_cell, neuron_ids, label_str)
        axes(ax); cla(ax); hold(ax,'on'); 
        nN = numel(spike_cell);
        if nN>0
            yl1 = 0.5; yl2 = nN+0.5;
        else
            yl1 = 0; yl2 = 1;
        end
        gg1 = max(t1,e1); gg2 = min(t2,e2);
        if gg1 < gg2
            patch(ax,[gg1 gg2 gg2 gg1],[yl1 yl1 yl2 yl2],[0.6 0.9 0.6], ...
                  'FaceAlpha',0.25,'EdgeColor','none');
        end

        % 将所有 spike 合并成一个向量做一次 scatter（比每 neuron 一个 scatter 快）
        all_t = [];
        all_y = [];
        for ii = 1:nN
            st = spike_cell{ii};
            if isempty(st), continue; end
            st_win = st(st>=t1 & st<=t2);
            if isempty(st_win), continue; end
            all_t = [all_t, st_win]; %#ok<AGROW>
            all_y = [all_y, ii*ones(1,numel(st_win))]; %#ok<AGROW>
        end
        if ~isempty(all_t)
            scatter(ax, all_t, all_y, 8, '.', ...
                'MarkerEdgeColor',[0 0 0], 'HitTest','off');
        else
            text(mean([t1 t2]),0.5,'No units','Parent',ax,'HorizontalAlignment','center');
        end

        if nN>0
            ylim(ax,[0.5 nN+0.5]);
            % 为了提速，大量 neuron 时不画全部 yticklabel（可按需求调整阈值）
            if nN <= 80
                yticks(ax,1:nN); 
                yticklabels(ax,neuron_ids(:)); 
            else
                yticks(ax,[]);  % 只画栅格，不画标签
            end
        else
            ylim(ax,[0 1]);
        end
        xlim(ax,[t1 t2]); 
        xlabel(ax,'Time (s)'); 
        ylabel(ax,'Neuron ID');
        title(ax,label_str); 
        hold(ax,'off');
    end

draw_raster(axCTX, S.spike_times_CTX, S.neuron_id_CTX, ...
            sprintf('Raster — Cortex (Event %d)',idx));
draw_raster(axSTR, S.spike_times_STR, S.neuron_id_STR, ...
            sprintf('Raster — Striatum (Event %d)',idx));

%% --- (5–6) Population rate: raw + smooth ---
axes(axCTXr); cla(axCTXr); hold(axCTXr,'on');
axes(axSTRr); cla(axSTRr); hold(axSTRr,'on');

bin_in_win = (S.tb(1:end-1)<t2) & (S.tb(2:end)>t1);
t_bins     = S.tc(bin_in_win);
y_ctx_raw  = S.binned_CTX_raw(bin_in_win);
y_str_raw  = S.binned_STR_raw(bin_in_win);
y_ctx_s    = S.binned_CTX(bin_in_win);
y_str_s    = S.binned_STR(bin_in_win);

% Cortex panel
if ~isempty(t_bins)
    bar(axCTXr,t_bins,y_ctx_raw,1.0,'FaceColor',[0.7 0.7 0.7],'EdgeColor','none'); 
    plot(axCTXr,t_bins,y_ctx_s,'-','Color',[1.0 0.4 0.0],'LineWidth',1.5);
end
xlim(axCTXr,[t1 t2]); xlabel(axCTXr,'Time (s)'); ylabel(axCTXr,'Spikes/s');
title(axCTXr,sprintf('Population rate — Cortex (Event %d)',idx));
grid(axCTXr,'on'); box(axCTXr,'off');

% Striatum panel
if ~isempty(t_bins)
    bar(axSTRr,t_bins,y_str_raw,1.0,'FaceColor',[0.7 0.7 0.7],'EdgeColor','none');
    plot(axSTRr,t_bins,y_str_s,'-','Color',[0.8 0.0 0.8],'LineWidth',1.5);
end
xlim(axSTRr,[t1 t2]); xlabel(axSTRr,'Time (s)'); ylabel(axSTRr,'Spikes/s');
title(axSTRr,sprintf('Population rate — Striatum (Event %d)',idx));
grid(axSTRr,'on'); box(axSTRr,'off');

% --- 绿色自动 peak + 文字 label（基于 smooth） ---
pC_t = S.auto_ctx_t(idx); pC_v = S.auto_ctx_v(idx);
pS_t = S.auto_str_t(idx); pS_v = S.auto_str_v(idx);

if isfinite(pC_t) && isfinite(pC_v)
    plot(axCTXr,pC_t,pC_v,'o','MarkerSize',8, ...
        'MarkerFaceColor',[0 0.7 0],'MarkerEdgeColor','k');
    % 绿色文字显示 X 值
    text(axCTXr,pC_t,pC_v, sprintf('%.3f',pC_t), ...
        'Color',[0 0.7 0], 'FontSize',8, ...
        'VerticalAlignment','bottom', 'HorizontalAlignment','center');
end
if isfinite(pS_t) && isfinite(pS_v)
    plot(axSTRr,pS_t,pS_v,'o','MarkerSize',8, ...
        'MarkerFaceColor',[0 0.7 0],'MarkerEdgeColor','k');
    text(axSTRr,pS_t,pS_v, sprintf('%.3f',pS_t), ...
        'Color',[0 0.7 0], 'FontSize',8, ...
        'VerticalAlignment','bottom', 'HorizontalAlignment','center');
end

% 手动编辑 peak（红色）
if S.edit_ctx_flag(idx) && isfinite(S.edit_ctx_t(idx)) && isfinite(S.edit_ctx_v(idx))
    plot(axCTXr,S.edit_ctx_t(idx),S.edit_ctx_v(idx),'o','MarkerSize',8, ...
        'MarkerFaceColor',[1 0 0],'MarkerEdgeColor','k');
end
if S.edit_str_flag(idx) && isfinite(S.edit_str_t(idx)) && isfinite(S.edit_str_v(idx))
    plot(axSTRr,S.edit_str_t(idx),S.edit_str_v(idx),'o','MarkerSize',8, ...
        'MarkerFaceColor',[1 0 0],'MarkerEdgeColor','k');
end

hold(axCTXr,'off'); 
hold(axSTRr,'off');

%% --- (7) Combined panel: raw CTX & STR + smooth ---
axes(axCombo); cla(axCombo); hold(axCombo,'on');

if ~isempty(t_bins)
    area(axCombo, t_bins, y_str_raw, ...
        'FaceColor',[0.9 0.8 0.95], ...
        'EdgeColor','none', ...
        'FaceAlpha',0.8);

    area(axCombo, t_bins, y_ctx_raw, ...
        'FaceColor',[1.0 0.9 0.7], ...
        'EdgeColor','none', ...
        'FaceAlpha',0.6);

    plot(axCombo, t_bins, y_ctx_s, '-', ...
        'Color',[1.0 0.4 0.0], 'LineWidth',1.5);

    plot(axCombo, t_bins, y_str_s, '-', ...
        'Color',[0.8 0.0 0.8], 'LineWidth',1.5);
end

xlim(axCombo,[t1 t2]);
xlabel(axCombo,'Time (s)');
ylabel(axCombo,'Spikes/s');
title(axCombo,sprintf('Population rate (Combined) — CTX & STR, Event %d',idx));
grid(axCombo,'on'); box(axCombo,'off');
legend(axCombo,{'STR raw','CTX raw','CTX smooth','STR smooth'},'Location','best');

hold(axCombo,'off');
end  % local_change_event

%% ===================== Helper: 手动点击 peak =====================
function local_set_peak_async(hFig,regionTag)
S = getappdata(hFig,'SWD_GUI_DATA');
idx = S.curr_idx;

switch upper(regionTag)
    case 'CTX'
        ax = S.ax(5);
    case 'STR'
        ax = S.ax(6);
    otherwise
        return;
end

if ~ishandle(ax), return; end

set(hFig,'Pointer','crosshair');
set(hFig,'WindowButtonDownFcn',@(src,~)click_callback(src,ax,regionTag));

    function click_callback(src,ax_local,tag)
        if ~ishandle(ax_local), return; end
        cp = get(ax_local,'CurrentPoint'); 
        x = cp(1,1); 
        y = cp(1,2);
        S = getappdata(src,'SWD_GUI_DATA');
        idx = S.curr_idx;
        [~,iNear] = min(abs(S.tc - x));
        x = S.tc(iNear);
        switch upper(tag)
            case 'CTX'
                S.edit_ctx_flag(idx) = true;
                S.edit_ctx_t(idx)    = x;
                S.edit_ctx_v(idx)    = y;
            case 'STR'
                S.edit_str_flag(idx) = true;
                S.edit_str_t(idx)    = x;
                S.edit_str_v(idx)    = y;
        end
        setappdata(src,'SWD_GUI_DATA',S);
        set(src,'Pointer','arrow','WindowButtonDownFcn','');
        local_change_event(src,idx);
        % 不再 autosave
    end
end

%% ===================== Helper: 按时间跳转 =====================
function local_jump_to_time(hFig)
S = getappdata(hFig,'SWD_GUI_DATA');
if ~isfield(S,'hJumpEdit') || ~ishandle(S.hJumpEdit)
    return;
end
str = get(S.hJumpEdit,'String');
t_query = str2double(str);
if isnan(t_query)
    warndlg('请输入合法的时间（秒）','Jump to time');
    return;
end

[~,idx] = min(abs(S.ext_sec(:,1) - t_query));
fprintf('Jump to time %.3f s -> nearest event #%d (start = %.3f s)\n', ...
    t_query, idx, S.ext_sec(idx,1));

local_change_event(hFig,idx);
end

%% ===================== Helper: 导出 qualityMatrix =====================
function local_export_quality(hFig)
S = getappdata(hFig,'SWD_GUI_DATA');
K = S.K;

qualityMatrix = nan(K,11);
qualityMatrix(:,1)  = (1:K).';
qualityMatrix(:,2)  = S.auto_ctx_t;
qualityMatrix(:,3)  = S.auto_ctx_v;
qualityMatrix(:,4)  = double(S.edit_ctx_flag);
qualityMatrix(:,5)  = S.edit_ctx_t;
qualityMatrix(:,6)  = S.edit_ctx_v;
qualityMatrix(:,7)  = S.auto_str_t;
qualityMatrix(:,8)  = S.auto_str_v;
qualityMatrix(:,9)  = double(S.edit_str_flag);
qualityMatrix(:,10) = S.edit_str_t;
qualityMatrix(:,11) = S.edit_str_v;

assignin('base','qualityMatrix',qualityMatrix);

out_dir = fullfile(pwd,'Str_Cor_histPeak_diff');
if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

timestamp = datestr(now,'yyyymmdd_HHMMSS');
base_name = fullfile(out_dir, ['qualityMatrix_' timestamp]);
mat_path  = [base_name '.mat'];
xlsx_path = [base_name '.xlsx'];

save(mat_path,'qualityMatrix','-v7.3');

qualityTable = table( ...
    (1:K).', ...
    S.auto_ctx_t, S.auto_ctx_v, S.edit_ctx_flag, S.edit_ctx_t, S.edit_ctx_v, ...
    S.auto_str_t, S.auto_str_v, S.edit_str_flag, S.edit_str_t, S.edit_str_v, ...
    'VariableNames', { ...
        'event_idx', ...
        'ctx_auto_t','ctx_auto_v','ctx_edited','ctx_edit_t','ctx_edit_v', ...
        'str_auto_t','str_auto_v','str_edited','str_edit_t','str_edit_v'});

writetable(qualityTable, xlsx_path);
assignin('base','qualityTable',qualityTable);

fprintf('[Quality] Saved to:\n  %s\n  %s\n', mat_path, xlsx_path);
end

%% ===================== Helper: Autosave wrapper =====================
function local_autosave(hFig)
% 不再在点击 peak 时自动保存；
% 保留这个函数占位（避免其它地方引用时报错），目前不做任何事。
% 如需恢复 autosave，可在此处调用 local_export_quality(hFig)。
% local_export_quality(hFig);
end



