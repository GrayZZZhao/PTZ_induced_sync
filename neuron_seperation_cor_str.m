%% Customized for PTZ IID recording: detection on mean(d) instead of frequency-specific SWD
% Prereqs assumed available in your path: 
% loading_lfp.m, baseline_correction.m, plot_swd_spectrogram_0_60.m, 
% plot_filtered_lfp.m (returns filtered trace), detect_swd.m (sets swd_events)

%% Load and preprocess LFP
loading_lfp
corrected_baseline = baseline_correction(mean(d), 2500);
plot_swd_spectrogram_0_60(corrected_baseline, 2500);
A = plot_filtered_lfp(mean(d), 2500);

% Detect SWD/IID events at 2500 Hz on the baseline-corrected mean-LFP
detect_swd(corrected_baseline);
swd_events_2500Hz = swd_events;    % keep a copy

% (Optional) avoid overwriting swd_events by repeated calls:
% detect_swd(A);
% detect_swd(mean(d));

%% Resample event indices from 2.5 kHz to 30 kHz (index scaling)
original_sampling_rate = 2500;     % Hz
new_sampling_rate      = 30000;    % Hz
scaling_factor         = new_sampling_rate / original_sampling_rate;

swd_events_30000Hz = cell(size(swd_events));  % preallocate
for i = 1:numel(swd_events)
    swd_events_30000Hz{i} = swd_events{i} * scaling_factor; % indices scaled
end
disp('SWD events converted to 30000Hz (indices):');
disp(swd_events_30000Hz);

%% Convert event indices (2.5 kHz) to seconds
swd_events_seconds = cell(size(swd_events));
for i = 1:numel(swd_events)
    swd_events_seconds{i} = swd_events{i} / original_sampling_rate;
end
disp('SWD events in seconds:');
disp(swd_events_seconds);

%% Build logical burst array at 30 kHz using the 30 kHz-scaled event index windows
% If you want to cover a larger 30 kHz span than A provides, adjust size_ap.
size_ap = 12 * size(A, 2);               % heuristic span (samples at 30 kHz)
logic_array = false(1, size_ap);

for i = 1:numel(swd_events_30000Hz)
    current_event = swd_events_30000Hz{i};
    if ~isempty(current_event)
        min_index = min(current_event);
        max_index = max(current_event);
        if min_index >= 1 && max_index <= numel(logic_array)
            logic_array(min_index:max_index) = true;
        end
    end
end
disp('Logical array populated successfully.');

figure;
hold on;
plot(logic_array);
ylim([0, 2]);
hold off;

%% Load spike raster arrays (WT example) and build (templateID, spikeTimes) table
spike_time_wt = readNPY('D:\Ylabrecording2025\2024-12-24_1347_WT-male-adult\2024-12-24_16-00-36\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\spike_times.npy');
spike_time_full_wt = double(spike_time_wt) / 30000;

spike_templates_wt = readNPY('D:\Ylabrecording2025\2024-12-24_1347_WT-male-adult\2024-12-24_16-00-36\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\spike_templates.npy');
spike_templates_full_wt = double(spike_templates_wt) + 1;   % 1-based IDs

spike_channel_wt = [spike_time_full_wt, spike_templates_full_wt];

[~, sortIdx_wt] = sort(spike_channel_wt(:, 2));
sortedArray2D_wt = spike_channel_wt(sortIdx_wt, :);

uniqueValues_wt = unique(sortedArray2D_wt(:, 2));
groupedRows_wt  = accumarray(sortedArray2D_wt(:, 2), sortedArray2D_wt(:, 1), [], @(x) {x'});

maxGroupSize_wt = max(cellfun(@length, groupedRows_wt));
resultArray_wt  = nan(length(uniqueValues_wt), maxGroupSize_wt + 1);
for i = 1:length(uniqueValues_wt)
    resultArray_wt(i, 1) = uniqueValues_wt(i);
    resultArray_wt(i, 2:length(groupedRows_wt{i}) + 1) = groupedRows_wt{i};
end

timepoint_array = resultArray_wt(:, 2:end);
neuron_id       = resultArray_wt(:, 1);

%% Burst vs non-burst spike rates (using logic_array @ 30 kHz)
sampling_rate          = 30000;          % Hz
min_burst_duration     = 1;              % seconds (threshold)
min_burst_samples      = min_burst_duration * sampling_rate;
bin_size               = 0.1;            % seconds
bin_samples            = bin_size * sampling_rate; %#ok<NASGU>

% Identify contiguous true segments
burst_diff    = diff([0, logic_array, 0]);
burst_starts  = find(burst_diff == 1);
burst_ends    = find(burst_diff == -1) - 1;
burst_durations = burst_ends - burst_starts + 1;

% Keep bursts >= threshold
valid_bursts = find(burst_durations >= min_burst_samples);
burst_starts = burst_starts(valid_bursts);
burst_ends   = burst_ends(valid_bursts);

num_neurons            = size(resultArray_wt, 1);
burst_spike_rate       = zeros(num_neurons, 1);
non_burst_spike_rate   = zeros(num_neurons, 1);

total_burst_time     = sum((burst_ends - burst_starts + 1)) / sampling_rate;
total_non_burst_time = numel(logic_array) / sampling_rate - total_burst_time;

% Precompute all burst ranges for non-burst counting
if ~isempty(burst_starts)
    all_burst_ranges = cell2mat(arrayfun(@(s, e) s:e, burst_starts, burst_ends, 'UniformOutput', false));
else
    all_burst_ranges = [];
end

for neuron = 1:num_neurons
    spike_times = resultArray_wt(neuron, :);
    spike_times = spike_times(spike_times > 0);           % seconds
    spike_samples = round(spike_times * sampling_rate);   % samples

    % spikes in bursts
    burst_spike_count = 0;
    for b = 1:length(burst_starts)
        burst_range = burst_starts(b):burst_ends(b);
        burst_spike_count = burst_spike_count + sum(ismember(spike_samples, burst_range));
    end

    % spikes outside bursts
    if ~isempty(all_burst_ranges)
        non_burst_spike_count = sum(~ismember(spike_samples, all_burst_ranges));
    else
        non_burst_spike_count = numel(spike_samples);
    end

    % spike rates (spikes/s)
    burst_spike_rate(neuron)     = burst_spike_count     / max(total_burst_time,     eps);
    non_burst_spike_rate(neuron) = non_burst_spike_count / max(total_non_burst_time, eps);
end

burst_to_non_burst_rate_ratio = burst_spike_rate ./ (non_burst_spike_rate + 1e-6);

% Select neurons with >2x burst rate
selected_neurons = find(burst_to_non_burst_rate_ratio > 2);

% X-axis range (seconds)
x_start = 0;
x_end   = max(resultArray_wt(:));

%% Multi-panel visualization
figure;
subplot_positions = [ ...
    0.1, 0.85, 0.8, 0.10;  ... % (1) Spectrogram
    0.1, 0.70, 0.8, 0.10;  ... % (2) Filtered LFP
    0.1, 0.55, 0.8, 0.10;  ... % (3) SWD detection
    0.1, 0.40, 0.8, 0.10;  ... % (4) Burst detection
    0.1, 0.25, 0.8, 0.13;  ... % (5) Raster
    0.1, 0.10, 0.8, 0.07];     % (6) Binned spike rates

ax = gobjects(6, 1);

% (1) Spectrogram 0–60 Hz of corrected_baseline at 2.5 kHz
ax(1) = subplot('Position', subplot_positions(1, :));
window_size = 1 * 2500;
noverlap    = 0.5 * window_size;
nfft        = 2^nextpow2(window_size);
[S, F, T, P] = spectrogram(corrected_baseline, window_size, noverlap, nfft, 2500);
PdB = 10 * log10(P);
freq_limit = F <= 60;
S_limited  = PdB(freq_limit, :);
F_limited  = F(freq_limit);
imagesc(T, F_limited, S_limited);
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram (0–60 Hz)');
xlim([x_start, x_end]);
colormap(jet);

% (2) Filtered LFP (5–60 Hz) in uV
ax(2) = subplot('Position', subplot_positions(2, :));
hold on;
low_cutoff = 5; 
high_cutoff = 60;
[b, a] = butter(4, [low_cutoff, high_cutoff] / (2500 / 2), 'bandpass');
filtered_5_60_lfp = filtfilt(b, a, mean(d));
scale_factor = 0.195;  % uV/bit
filtered_lfp_microV = filtered_5_60_lfp * scale_factor;
time_vector = (1:length(filtered_5_60_lfp)) / 2500;
plot(time_vector, filtered_lfp_microV, 'b');
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
title('Filtered LFP (5–60 Hz) in \muV');
ylim([-2500, 2500]);
xlim([x_start, x_end]);
hold off;

% (3) SWD detection overlay on mean(d)
ax(3) = subplot('Position', subplot_positions(3, :));
time_in_seconds = (1:length(mean(d))) / original_sampling_rate;
plot(time_in_seconds, mean(d), 'Color', [0.6, 0.6, 0.6]);
hold on;
dmean = mean(d);
for i = 1:numel(swd_events)
    event = swd_events{i};
    event_time = event / original_sampling_rate;
    plot(event_time, dmean(event), 'r', 'LineWidth', 2);
end
xlabel('Time (s)');
ylabel('Amplitude (a.u.)');
title('SWD Detection in LFP Signal');
xlim([x_start, x_end]);
ylim([-20000, 20000]);
hold off;

% (4) Burst detection (>=1 s) shaded
ax(4) = subplot('Position', subplot_positions(4, :));
hold on;
for bidx = 1:length(burst_starts)
    start_time = burst_starts(bidx) / sampling_rate;
    end_time   = burst_ends(bidx)   / sampling_rate;
    fill([start_time, end_time, end_time, start_time], [0, 0, 1, 1], 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
end
title('Burst Detection (1 s or more)');
ylabel('Burst');
ylim([0, 1]);
xlim([x_start, x_end]);
hold off;

% (5) Raster plot for selected neurons
ax(5) = subplot('Position', subplot_positions(5, :));
hold on;
for i = 1:length(selected_neurons)
    r = selected_neurons(i);
    spike_times = resultArray_wt(r, :);
    spike_times = spike_times(spike_times > 0);
    scatter(spike_times, ones(size(spike_times)) * r, 10, '.');
end
xlabel('Time (s)');
ylabel('Neuron Index');
title('Raster Plot (Selected Neurons)');
xlim([x_start, x_end]);
hold off;

% (6) Binned spike rates for selected neurons (0.1 s bins)
ax(6) = subplot('Position', subplot_positions(6, :));
bin_width = 0.1;
time_bins = x_start:bin_width:x_end;
selected_spike_times = [];
for i = 1:length(selected_neurons)
    r = selected_neurons(i);
    st = resultArray_wt(r, :);
    st = st(st > 0);
    selected_spike_times = [selected_spike_times, st]; %#ok<AGROW>
end
binned_spikes = histcounts(selected_spike_times, time_bins) / bin_width;
bar(time_bins(1:end-1), binned_spikes, 'k');
xlabel('Time (s)');
ylabel('Spike Rate (spikes/s)');
title('Binned Spike Rates (Selected Neurons)');
xlim([x_start, x_end]);

linkaxes(ax, 'x');
set(gcf, 'Position', [100, 100, 800, 900]);

%% ---- Verify neuron_id vs clustertwo.xlsx col 2 (+1) ----
excel_path = 'D:\Ylabrecording2025\2024-12-24_1347_WT-male-adult\2024-12-24_16-00-36\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\clustertwo.xlsx';

try
    T = readtable(excel_path);
    if width(T) >= 2
        col2 = T{:, 2};
    else
        error('Excel file has fewer than 2 columns.');
    end
    if ~isnumeric(col2)
        col2 = str2double(string(col2));
    end
catch
    X = readmatrix(excel_path);
    if size(X, 2) < 2
        error('Excel file has fewer than 2 columns.');
    end
    col2 = X(:, 2);
end

excel_ids0 = col2(:);
excel_ids1 = excel_ids0 + 1;   % 0-based -> 1-based

neuron_id = resultArray_wt(:, 1);
neuron_id = neuron_id(:);

fprintf('Counts -> neuron_id: %d, excel col2(+1): %d\n', numel(neuron_id), numel(excel_ids1));

equal_sets = isequal(sort(neuron_id), sort(excel_ids1(~isnan(excel_ids1))));
if equal_sets
    fprintf('[OK] Same ID set after +1 conversion (order may differ).\n');
else
    only_in_neuron = setdiff(neuron_id, excel_ids1);
    only_in_excel  = setdiff(excel_ids1, neuron_id);
    fprintf('[WARN] ID sets differ.\n');
    if ~isempty(only_in_neuron)
        fprintf('  IDs only in neuron_id (first up to 10): %s\n', mat2str(only_in_neuron(1:min(10, end))'));
    end
    if ~isempty(only_in_excel)
        fprintf('  IDs only in excel(+1) (first up to 10): %s\n', mat2str(only_in_excel(1:min(10, end))'));
    end
end

nmin = min(numel(neuron_id), numel(excel_ids1));
rowwise_equal = isequal(neuron_id(1:nmin), excel_ids1(1:nmin));
if rowwise_equal && numel(neuron_id) == numel(excel_ids1)
    fprintf('[OK] Same order and same length.\n');
elseif rowwise_equal
    fprintf('[OK] First %d rows match in order, but lengths differ.\n', nmin);
else
    mism = find(neuron_id(1:nmin) ~= excel_ids1(1:nmin));
    fprintf('[INFO] Row order differs. First up to 10 mismatches (row, neuron_id, excel+1):\n');
    for k = 1:min(10, numel(mism))
        r = mism(k);
        fprintf('  %d: %g vs %g\n', r, neuron_id(r), excel_ids1(r));
    end
end

[tf, loc] = ismember(neuron_id, excel_ids1); %#ok<ASGLU>
fprintf('[Map] %d/%d neuron_ids found in excel(+1).\n', sum(tf), numel(neuron_id));

%% --- Region labeling from Excel depth (col 4) and splitting variables ---
excel_path = 'D:\Ylabrecording2025\2024-12-24_1347_WT-male-adult\2024-12-24_16-00-36\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\clustertwo.xlsx';

try
    T = readtable(excel_path);
    if width(T) < 4
        error('Excel file has fewer than 4 columns.');
    end
    depth_col = T{:, 4};
    if ~isnumeric(depth_col)
        depth_col = str2double(string(depth_col));
    end
catch
    X = readmatrix(excel_path);
    if size(X, 2) < 4
        error('Excel file has fewer than 4 columns.');
    end
    depth_col = X(:, 4);
end

depth_um = depth_col(:);
neuron_id = resultArray_wt(:, 1);

assert(numel(neuron_id) == numel(depth_um), 'neuron_id and depth length mismatch.');

is_STR = depth_um < 2625;
is_CTX = ~is_STR;

region = repmat("cortex", numel(depth_um), 1);
region(is_STR) = "striatum";

neuron_id_STR = neuron_id(is_STR);
neuron_id_CTX = neuron_id(is_CTX);

if ~exist('selected_neurons', 'var') || isempty(selected_neurons)
    warning('selected_neurons not found; creating empty region splits.');
    selected_neurons_STR = [];
    selected_neurons_CTX = [];
else
    selected_neurons = selected_neurons(:);
    selected_neurons_STR = selected_neurons(is_STR(selected_neurons));
    selected_neurons_CTX = selected_neurons(is_CTX(selected_neurons));
end

RegionTable = table( ...
    neuron_id(:), ...
    depth_um(:), ...
    categorical(region), ...
    'VariableNames', {'neuron_id', 'depth_um', 'region'} ...
);

out_mat = 'neuron_region_labels.mat';
save(out_mat, 'is_STR', 'is_CTX', 'region', 'depth_um', ...
    'neuron_id', 'neuron_id_STR', 'neuron_id_CTX', ...
    'selected_neurons', 'selected_neurons_STR', 'selected_neurons_CTX', ...
    'RegionTable');
fprintf('[OK] Regions labeled. Saved outputs to %s\n', out_mat);
fprintf('Counts -> STR: %d, CTX: %d (total %d)\n', nnz(is_STR), nnz(is_CTX), numel(is_STR));
if exist('selected_neurons', 'var') && ~isempty(selected_neurons)
    fprintf('Selected -> STR: %d, CTX: %d (total %d)\n', numel(selected_neurons_STR), numel(selected_neurons_CTX), numel(selected_neurons));
end

%% ---- Build 1 s–extended SWD intervals (2.5 kHz sample space) ----
extend_sec = 1;
fs0 = original_sampling_rate;  % 2500
N   = numel(A);
extN = round(extend_sec * fs0);

evt_bounds = nan(numel(swd_events), 2);
for i = 1:numel(swd_events)
    v = swd_events{i};
    if ~isempty(v)
        evt_bounds(i, :) = [v(1), v(end)];
    end
end
evt_bounds = evt_bounds(~isnan(evt_bounds(:, 1)), :);

evt_ext = [max(1, evt_bounds(:, 1) - extN), min(N, evt_bounds(:, 2) + extN)];
evt_ext = sortrows(evt_ext, 1);

merged_ext = [];
for k = 1:size(evt_ext, 1)
    if isempty(merged_ext) || evt_ext(k, 1) > merged_ext(end, 2) + 1
        merged_ext = [merged_ext; evt_ext(k, :)]; %#ok<AGROW>
    else
        merged_ext(end, 2) = max(merged_ext(end, 2), evt_ext(k, 2));
    end
end
swd_events_ext_1s = merged_ext;   % [K x 2] start/end indices @ 2.5 kHz

%% ==================== Region composites (2,3,5,6) ====================
if ~exist('is_STR', 'var')
    error('is_STR not found. Run the depth-based labeling step first.');
end
if ~exist('neuron_id', 'var') || isempty(neuron_id)
    neuron_id = resultArray_wt(:, 1);
end
if ~exist('selected_neurons', 'var') || isempty(selected_neurons)
    selected_neurons = (1:size(resultArray_wt, 1)).';
end
selected_neurons = selected_neurons(:);
is_CTX = ~is_STR;

selected_neurons_STR = selected_neurons(is_STR(selected_neurons));
selected_neurons_CTX = selected_neurons(is_CTX(selected_neurons));

if ~exist('x_start', 'var') || ~exist('x_end', 'var') || isempty(x_start) || isempty(x_end)
    x_start = 0;
    try
        x_end = max(length(A) / original_sampling_rate, max(resultArray_wt(:)));
    catch
        x_end = max(resultArray_wt(:));
    end
    if ~isfinite(x_end) || x_end <= x_start
        x_end = 1;
    end
end

pos_2 = [0.1, 0.70, 0.8, 0.10];
pos_3 = [0.1, 0.55, 0.8, 0.10];
pos_5 = [0.1, 0.25, 0.8, 0.13];
pos_6 = [0.1, 0.10, 0.8, 0.07];

fs = 2500;
[b_lfp, a_lfp] = butter(4, [5, 60] / (fs / 2), 'bandpass');
lfp_mean = mean(d, 1);
lfp_filt = filtfilt(b_lfp, a_lfp, lfp_mean);
lfp_scale = 0.195;
lfp_t = (1:numel(lfp_filt)) / fs;

swd_t = (1:length(A)) / original_sampling_rate;

% ----- Striatum figure -----
figSTR = figure('Name', 'Region composite — Striatum (2,3,5,6)', 'Position', [120, 60, 1200, 820]);
axSTR = gobjects(4, 1);

axSTR(1) = subplot('Position', pos_2);
plot(lfp_t, lfp_filt * lfp_scale);
title('(2) Filtered LFP (5–60 Hz) in \muV');
xlabel('Time (s)'); ylabel('Amplitude (\muV)');
ylim([-2500, 2500]); xlim([x_start, x_end]);

axSTR(2) = subplot('Position', pos_3);
plot(swd_t, A, 'Color', [0.6, 0.6, 0.6]); hold on;
for ii = 1:numel(swd_events)
    evt = swd_events{ii};
    te  = evt / original_sampling_rate;
    plot(te, A(evt), 'r', 'LineWidth', 2);
end
for ii = 1:size(swd_events_ext_1s, 1)
    idx_range = swd_events_ext_1s(ii, 1):swd_events_ext_1s(ii, 2);
    plot(idx_range / fs0, A(idx_range), 'g', 'LineWidth', 2);
end
hold off;
title('(3) SWD Detection'); xlabel('Time (s)'); ylabel('Amplitude (a.u.)');
ylim([-20000, 20000]); xlim([x_start, x_end]);

axSTR(3) = subplot('Position', pos_5); hold on;
if ~isempty(selected_neurons_STR)
    ids_local = neuron_id(selected_neurons_STR);
    for i = 1:numel(selected_neurons_STR)
        r  = selected_neurons_STR(i);
        st = resultArray_wt(r, :); st = st(st > 0);
        scatter(st, i * ones(size(st)), 10, '.');
    end
    yticks(1:numel(selected_neurons_STR)); yticklabels(ids_local);
    ylim([0.5, numel(selected_neurons_STR) + 0.5]);
else
    text(mean([x_start, x_end]), 0.5, 'No STR neurons selected', 'HorizontalAlignment', 'center');
    ylim([0, 1]);
end
title('(5) Raster — Striatum'); xlabel('Time (s)'); ylabel('Neuron ID');
xlim([x_start, x_end]); hold off;

axSTR(4) = subplot('Position', pos_6);
bin_width = 0.01;
tb = x_start:bin_width:x_end;
sel_spikes = [];
for i = 1:numel(selected_neurons_STR)
    r  = selected_neurons_STR(i);
    st = resultArray_wt(r, :); st = st(st > 0);
    sel_spikes = [sel_spikes, st]; %#ok<AGROW>
end
if ~isempty(sel_spikes)
    binned = histcounts(sel_spikes, tb) / bin_width;
    bar(tb(1:end-1), binned, 'k');
else
    bar(0, 0, 'k'); xlim([x_start, x_end]);
end
title(sprintf('(6) Population rate (%.2f s bins) — Striatum', bin_width));
xlabel('Time (s)'); ylabel('Spikes/s'); xlim([x_start, x_end]);
linkaxes(axSTR, 'x');

% ----- Cortex figure -----
figCTX = figure('Name', 'Region composite — Cortex (2,3,5,6)', 'Position', [140, 40, 1200, 820]);
axCTX = gobjects(4, 1);

axCTX(1) = subplot('Position', pos_2);
plot(lfp_t, lfp_filt * lfp_scale);
title('(2) Filtered LFP (5–60 Hz) in \muV');
xlabel('Time (s)'); ylabel('Amplitude (\muV)');
ylim([-2500, 2500]); xlim([x_start, x_end]);

axCTX(2) = subplot('Position', pos_3);
plot(swd_t, A, 'Color', [0.6, 0.6, 0.6]); hold on;
for ii = 1:numel(swd_events)
    evt = swd_events{ii};
    te  = evt / original_sampling_rate;
    plot(te, A(evt), 'r', 'LineWidth', 2);
end
hold off;
title('(3) SWD Detection'); xlabel('Time (s)'); ylabel('Amplitude (a.u.)');
ylim([-20000, 20000]); xlim([x_start, x_end]);

axCTX(3) = subplot('Position', pos_5); hold on;
if ~isempty(selected_neurons_CTX)
    ids_local = neuron_id(selected_neurons_CTX);
    for i = 1:numel(selected_neurons_CTX)
        r  = selected_neurons_CTX(i);
        st = resultArray_wt(r, :); st = st(st > 0);
        scatter(st, i * ones(size(st)), 10, '.');
    end
    yticks(1:numel(selected_neurons_CTX)); yticklabels(ids_local);
    ylim([0.5, numel(selected_neurons_CTX) + 0.5]);
else
    text(mean([x_start, x_end]), 0.5, 'No CTX neurons selected', 'HorizontalAlignment', 'center');
    ylim([0, 1]);
end
title('(5) Raster — Cortex'); xlabel('Time (s)'); ylabel('Neuron ID');
xlim([x_start, x_end]); hold off;

axCTX(4) = subplot('Position', pos_6);
sel_spikes = [];
for i = 1:numel(selected_neurons_CTX)
    r  = selected_neurons_CTX(i);
    st = resultArray_wt(r, :); st = st(st > 0);
    sel_spikes = [sel_spikes, st]; %#ok<AGROW>
end
if ~isempty(sel_spikes)
    binned = histcounts(sel_spikes, tb) / bin_width;
    bar(tb(1:end-1), binned, 'k');
else
    bar(0, 0, 'k'); xlim([x_start, x_end]);
end
title(sprintf('(6) Population rate (%.2f s bins) — Cortex', bin_width));
xlabel('Time (s)'); ylabel('Spikes/s'); xlim([x_start, x_end]);
linkaxes(axCTX, 'x');

%% --- Population rate peaks in 1 s–extended SWD windows (STR & CTX) ---
if ~exist('tc', 'var') || isempty(tc)
    tc = tb(1:end-1) + bin_width / 2;   % bin centers
end

build_sel_spikes = @(rows) cell2mat( ...
    arrayfun(@(r) nonzeros(resultArray_wt(r, :))', rows(:), 'UniformOutput', false) );

sel_spikes_STR = build_sel_spikes(selected_neurons_STR);
sel_spikes_CTX = build_sel_spikes(selected_neurons_CTX);

binned_STR = histcounts(sel_spikes_STR, tb) / bin_width;
binned_CTX = histcounts(sel_spikes_CTX, tb) / bin_width;

ext_sec = swd_events_ext_1s / original_sampling_rate;  % [K x 2] seconds
K = size(ext_sec, 1);

PeakInfo_STR(K, 1) = struct('t1', NaN, 't2', NaN, 't_peak', NaN, 'v_peak', NaN);
PeakInfo_CTX      = PeakInfo_STR;

for k = 1:K
    t1 = ext_sec(k, 1);  t2 = ext_sec(k, 2);
    PeakInfo_STR(k).t1 = t1;  PeakInfo_STR(k).t2 = t2;
    PeakInfo_CTX(k).t1 = t1;  PeakInfo_CTX(k).t2 = t2;

    bin_in_win = (tb(1:end-1) < t2) & (tb(2:end) > t1);
    if any(bin_in_win)
        [vS, iSrel] = max(binned_STR(bin_in_win));
        iS = find(bin_in_win, 1, 'first') + iSrel - 1;
        PeakInfo_STR(k).t_peak = tc(iS);
        PeakInfo_STR(k).v_peak = vS;

        [vC, iCrel] = max(binned_CTX(bin_in_win));
        iC = find(bin_in_win, 1, 'first') + iCrel - 1;
        PeakInfo_CTX(k).t_peak = tc(iC);
        PeakInfo_CTX(k).v_peak = vC;
    else
        tmid = (t1 + t2) / 2;
        [~, iNear] = min(abs(tc - tmid));
        PeakInfo_STR(k).t_peak = tc(iNear);
        PeakInfo_STR(k).v_peak = binned_STR(iNear);
        PeakInfo_CTX(k).t_peak = tc(iNear);
        PeakInfo_CTX(k).v_peak = binned_CTX(iNear);
    end
end

% Overlay peak markers on rate subplots
if exist('axSTR', 'var') && numel(axSTR) >= 4 && isgraphics(axSTR(4))
    axS = axSTR(4);
else
    fS  = findobj('Type', 'figure', 'Name', 'Region composite — Striatum (2,3,5,6)');
    axList = findobj(fS, 'Type', 'axes');
    axS = axList(1);
end
if exist('axCTX', 'var') && numel(axCTX) >= 4 && isgraphics(axCTX(4))
    axC = axCTX(4);
else
    fC  = findobj('Type', 'figure', 'Name', 'Region composite — Cortex (2,3,5,6)');
    axList = findobj(fC, 'Type', 'axes');
    axC = axList(1);
end

hold(axS, 'on');
plot(axS, [PeakInfo_STR.t_peak], [PeakInfo_STR.v_peak], '.', 'MarkerSize', 10, 'Color', [0, 0.6, 0]);
hold(axS, 'off');

hold(axC, 'on');
plot(axC, [PeakInfo_CTX.t_peak], [PeakInfo_CTX.v_peak], '.', 'MarkerSize', 10, 'Color', [0, 0.6, 0]);
hold(axC, 'off');

Nlabel = 25;   % annotate every 25th window (optional)
if ~isempty(Nlabel)
    ks = 1:Nlabel:K;
    for ii = ks
        text(PeakInfo_STR(ii).t_peak, PeakInfo_STR(ii).v_peak, sprintf(' #%d', ii), ...
             'Parent', axS, 'VerticalAlignment', 'bottom', 'Color', [0, 0.5, 0]);
        text(PeakInfo_CTX(ii).t_peak, PeakInfo_CTX(ii).v_peak, sprintf(' #%d', ii), ...
             'Parent', axC, 'VerticalAlignment', 'bottom', 'Color', [0, 0.5, 0]);
    end
end

nSTR = sum(isfinite([PeakInfo_STR.v_peak]));
nCTX = sum(isfinite([PeakInfo_CTX.v_peak]));
fprintf('[Per-window peaks] Found %d/%d STR peaks and %d/%d CTX peaks.\n', nSTR, K, nCTX, K);

%% Export extended-window starts + CTX/STR peak times to Excel
fs0 = original_sampling_rate;                    % 2500 Hz
ext_sec = swd_events_ext_1s / fs0;               % [K x 2] seconds
K = size(ext_sec, 1);

ctx_t = nan(K, 1);
str_t = nan(K, 1);
if exist('PeakInfo_CTX', 'var') && numel(PeakInfo_CTX) == K
    ctx_t = reshape([PeakInfo_CTX.t_peak], [], 1);
end
if exist('PeakInfo_STR', 'var') && numel(PeakInfo_STR) == K
    str_t = reshape([PeakInfo_STR.t_peak], [], 1);
end
start_sec = ext_sec(:, 1);

T = table(start_sec, ctx_t, str_t, 'VariableNames', {'start_sec', 'ctx_t_peak', 'str_t_peak'});
out_xlsx = fullfile(pwd, 'swd_peak_times_ctx_str.xlsx');
writetable(T, out_xlsx);
fprintf('[OK] Saved %d rows to: %s\n', K, out_xlsx);
disp('First 5 rows:');
disp(T(1:min(5, K), :));
