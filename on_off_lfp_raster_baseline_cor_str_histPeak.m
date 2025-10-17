%% cusimized for PTZ IID recording : detection on mean(d) instead of frequncy specific swd 

loading_lfp
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
% Convert time points
swd_events_30000Hz = cell(size(swd_events))% Replace this with your actual variable

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




%%
% Initialize the logical array
%logic_array = false(1, 90793140); % Logical array with all values initialized to 0 (false)
size_ap = 12*size(A,2);
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
spike_time_wt = readNPY('D:\Ylabrecording2025\2024-12-24_1347_WT-male-adult\2024-12-24_16-00-36\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\spike_times.npy');
spike_time_full_wt = double(spike_time_wt) / 30000;
spike_templates_wt = readNPY('D:\Ylabrecording2025\2024-12-24_1347_WT-male-adult\2024-12-24_16-00-36\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\spike_templates.npy');
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

%%
% Load variables (ensure logical_array and resultArray_wt are in your workspace)
% logical_array: 1x90793140 logical array (30000 Hz)
% resultArray_wt: 337x112319 double array (spike times in seconds)

% Parameters
sampling_rate = 30000; % Sampling rate in Hz
min_burst_duration = 1; % Minimum burst duration in seconds %%%%%%%%%%%%%%%
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
selected_neurons = find(burst_to_non_burst_rate_ratio >2);  %(<0.5 meaning off,    >2 meaing 0n )

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
                     0.1, 0.1, 0.8, f7]; % Position for sixth subplot

ax = zeros(6,1);

% Plot in the first subplot (Spectrogram)
ax(1)=subplot('Position', subplot_positions(1, :));
%function plot_swd_spectrogram_0_60(filtered_lfp, fs)
    % Spectrogram of the filtered LFP data, limited to 0-60 Hz
    window_size = 1 * 2500;  % 1-second window
    noverlap = 0.5 * window_size;  % 50% overlap
    nfft = 2^nextpow2(window_size);  % FFT length

    % Compute the spectrogram
    [S, F, T, P] = spectrogram(corrected_baseline, window_size, noverlap, nfft, 2500);

    % Convert power to dB
    PdB = 10 * log10(P);

    % Limit the frequency range to 0-60 Hz
    freq_limit = F <= 60;
    S_limited = PdB(freq_limit, :);
    F_limited = F(freq_limit);

    % Plot the spectrogram
    hold on
    imagesc(T, F_limited, S_limited);
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Spectrogram of Filtered LFP Signal (0-60 Hz)');
    xlim([x_start, x_end])
    
    % Colormap: dark blue to red
    colormap(jet);
    %colorbar;
    hold off
%end




% Plot in the second subplot (Filtered LFP signal)
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
    
plot(time_in_seconds, mean(d), 'Color', [0.6 0.6 0.6]); % Plot the LFP signal in a lighter color
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

%%   找id
% ---- Verify neuron_id vs clustertwo.xlsx col 2 (+1) ----
% Assumes you already built resultArray_wt, so:
% neuron_id = resultArray_wt(:,1);

excel_path = 'D:\Ylabrecording2025\2024-12-24_1347_WT-male-adult\2024-12-24_16-00-36\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\clustertwo.xlsx';

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

excel_path = 'D:\Ylabrecording2025\2024-12-24_1347_WT-male-adult\2024-12-24_16-00-36\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\clustertwo.xlsx';

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
extend_sec = 1;                                   % <-- change this if you want
fs0 = original_sampling_rate;                     % 2500 in your script
N   = numel(A);
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
evt_ext = [ max(1, evt_bounds(:,1) - extN),  min(N, evt_bounds(:,2) + extN) ];

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
swd_t = (1:length(A))/original_sampling_rate;

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
plot(swd_t, A, 'Color', [0.6 0.6 0.6]); hold on;
for ii = 1:numel(swd_events)
    evt = swd_events{ii};
    te  = evt / original_sampling_rate;
    plot(te, A(evt), 'r', 'LineWidth', 2);
end

for ii = 1:size(swd_events_ext_1s,1)
    idx_range = swd_events_ext_1s(ii,1):swd_events_ext_1s(ii,2);
    plot(idx_range / fs0, A(idx_range), 'g', 'LineWidth', 2);
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
plot(swd_t, A, 'Color', [0.6 0.6 0.6]); hold on;
for ii = 1:numel(swd_events)
    evt = swd_events{ii};
    te  = evt / original_sampling_rate;
    plot(te, A(evt), 'r', 'LineWidth', 2);
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

% Extract per-window peak times (seconds). If not computed yet, run the
% "Per-window peaks" block we wrote earlier.
ctx_t = nan(K,1);  str_t = nan(K,1);
if exist('PeakInfo_CTX','var') && numel(PeakInfo_CTX) == K
    ctx_t = reshape([PeakInfo_CTX.t_peak],[],1);
end
if exist('PeakInfo_STR','var') && numel(PeakInfo_STR) == K
    str_t = reshape([PeakInfo_STR.t_peak],[],1);
end

start_sec = ext_sec(:,1);                         % green-line start (s)
M = [start_sec, ctx_t, str_t];                    % N×3 as requested

% Save as Excel with headers
T = table(start_sec, ctx_t, str_t, ...
    'VariableNames', {'start_sec','ctx_t_peak','str_t_peak'});

out_xlsx = fullfile(pwd, 'swd_peak_times_ctx_str.xlsx');  % change path if needed
writetable(T, out_xlsx);

fprintf('[OK] Saved %d rows to: %s\n', K, out_xlsx);
disp('First 5 rows:'); disp(T(1:min(5,K),:));


