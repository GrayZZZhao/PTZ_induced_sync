%% cusimized for PTZ IID recording : detection on mean(d) instead of frequncy specific swd 

loading_lfp_v2
corrected_baseline = baseline_correction_v2(mean(d),2500);
%plot_swd_spectrogram_0_60_v2(corrected_baseline,2500);
%A = plot_filtered_lfp_v2(mean(d), 2500);
detect_swd_v2(corrected_baseline);
%detect_swd_v2(A);
%detect_swd_v2(mean(d));

%% 删除假阳性
% % 假设你已经有 corrected_baseline (或 lfp_data)、fs、swd_events
% [swd_events_filtered, keep_idx] = curate_swd_events(corrected_baseline, 2500, swd_events);
% 
% 
 save('swd_events_FixMissing.mat','swd_events_filtered','keep_idx');   % 如变量较大可加 '-v7.3'


%%
% Given variables
%swd_events = swd_events_filtered;
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
size_ap = 12*size(corrected_baseline,2);
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
spike_time_wt = readNPY('E:\npxlkv11_summer_2025\1822_hom_F_kv1.1\2025-07-20_19-43-58\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\spike_times.npy');
spike_time_full_wt = double(spike_time_wt) / 30000;
spike_templates_wt = readNPY('E:\npxlkv11_summer_2025\1822_hom_F_kv1.1\2025-07-20_19-43-58\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\spike_templates.npy');
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


%% GUI

[swd_events_filtered, keep_idx] = curate_swd_events(corrected_baseline, 2500, swd_events, 10, timepoint_array, neuron_id, struct('max_neurons',400));


%%
save('swd_events_FixMissing.mat','swd_events_filtered','keep_idx');   % 如变量较大可加 '-v7.3'



%% recalcuate 2500hz and 30000hz time point 

%swd_events = swd_events_filtered; 

swd_events_2500Hz = swd_events;

swd_events_30000Hz = cell(size(swd_events));% Replace this with your actual variable

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




%% repopulte logic array 
%logic_array = false(1, 90793140); % Logical array with all values initialized to 0 (false)
size_ap = 12*size(corrected_baseline,2);
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

figure()
hold on 
plot(logic_array)
ylim([0,2])
hold off


%%
% Load variables (ensure logical_array and resultArray_wt are in your workspace)
% logical_array: 1x90793140 logical array (30000 Hz)
% resultArray_wt: 337x112319 double array (spike times in seconds)

% Parameters262
sampling_rate = 30000; % Sampling rate in Hz
min_burst_duration = 0.1; % Minimum burst duration in seconds %%%%%%%%%%%%%%%
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
                     0.1, 0.1, 0.8, 0.07]; % Position for sixth subplot

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


%%


%% === DROP-IN: keep only neurons synchronized with detected events ===
% Uses the already-computed burst_starts/burst_ends (in samples at 30 kHz)
% and resultArray_wt (rows=neurons, cols=spike times in seconds).
% A neuron is called "synchronized" if it has at least `min_spikes_in_bursts`
% spikes inside the detected SWD/burst windows. Tweak as you like.

min_spikes_in_bursts = 5;      % <-- adjust threshold (>=5 spikes inside bursts)
x_start = 0;                   
x_end   = max(resultArray_wt(:));   % plot span based on spike data (sec)

% Make [start end] intervals (samples) for convenience
burst_intervals = [burst_starts(:), burst_ends(:)];   % samples at 30k
num_neurons = size(resultArray_wt,1);

sync_mask = false(num_neurons,1);

for n = 1:num_neurons
    st = resultArray_wt(n,:);              % seconds
    st = st(st > 0);                        % remove padding
    if isempty(st), continue; end
    ss = round(st * sampling_rate);         % to samples (30 kHz)

    % Count spikes that fall inside any burst interval
    % (efficient enough for typical sizes; vectorized per-interval)
    if ~isempty(burst_intervals)
        spikes_in_bursts = arrayfun(@(i) sum(ss >= burst_intervals(i,1) & ss <= burst_intervals(i,2)), ...
                                    1:size(burst_intervals,1));
        total_in_bursts = sum(spikes_in_bursts);
    else
        total_in_bursts = 0;
    end
    sync_mask(n) = total_in_bursts >= min_spikes_in_bursts;
end

sync_neurons = find(sync_mask);

fprintf('Synchronized neurons found: %d of %d\n', numel(sync_neurons), num_neurons);

% === Replot ONLY synchronized neurons in a new figure (raster) ===
figure('Name','Synchronized Neurons - Raster','Position',[100 100 800 350]); 
hold on;

% light red bands for detected burst intervals (optional, comment out if not needed)
for b = 1:size(burst_intervals,1)
    start_time = burst_intervals(b,1) / sampling_rate;
    end_time   = burst_intervals(b,2) / sampling_rate;
    patch([start_time end_time end_time start_time], [0 0 1 1]*max(1,numel(sync_neurons))+1, ...
          [1 0 0], 'FaceAlpha', 0.08, 'EdgeColor', 'none');
end

ytick_labels = [];

for i = 1:numel(sync_neurons)
    rn = sync_neurons(i);
    spike_times = resultArray_wt(rn,:);
    spike_times = spike_times(spike_times > 0);          % seconds
    % plot this neuron's spikes on row i
    scatter(spike_times, ones(size(spike_times))*i, 8, 'k', '.');
    ytick_labels(end+1,1) = rn;                           %#ok<SAGROW> keep original row ID as label
end

xlabel('Time (s)');
ylabel('Neuron ID (row in resultArray\_wt)');
title(sprintf('Raster of Synchronized Neurons (>= %d spikes inside bursts)', min_spikes_in_bursts));
ylim([0.5, max(1,numel(sync_neurons)) + 0.5]);
yticks(1:numel(sync_neurons));
if ~isempty(sync_neurons)
    yticklabels(string(ytick_labels));
else
    yticklabels([]);
end
xlim([x_start, x_end]);
box on; grid on;
hold off;



%% === DROP-IN: plot first 50 synchronized neurons (ID ascending) in two windows ===
% Prereqs in workspace:
%   resultArray_wt  (rows = neurons, cols = spike times in seconds)
%   sync_neurons    (vector of neuron row indices judged synchronized)
% Optional (for shaded windows): 
%   burst_starts, burst_ends (in samples at sampling_rate = 30000)

if ~exist('sync_neurons','var') || isempty(sync_neurons)
    warning('sync_neurons not found or empty. Using all neurons instead.');
    sync_neurons = (1:size(resultArray_wt,1))';
end

% sort by neuron ID ascending
neurons_sorted_asc = sort(sync_neurons, 'ascend');

% EXCLUDE specific neuron IDs first (edit this list as needed)
exclude_ids = [22];
neurons_sorted_asc = neurons_sorted_asc(~ismember(neurons_sorted_asc, exclude_ids));

% keep first 50 after exclusion
K = min(50, numel(neurons_sorted_asc));
neurons_to_plot = neurons_sorted_asc(1:K);

fprintf('Selected %d neurons after excluding [%s] (ascending IDs).\n', K, num2str(exclude_ids));

% two windows to show (edit as needed)
winA = [186 286];   % seconds
winB = [860 960];   % seconds

% helper to draw burst bands if available
draw_bands = exist('burst_starts','var') && exist('burst_ends','var') ...
             && ~isempty(burst_starts) && ~isempty(burst_ends) ...
             && exist('sampling_rate','var');

figure('Name','First-50 synchronized neurons (ascending IDs)','Position',[100 100 1100 450]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

for w = 1:2
    if w==1, win = winA; else, win = winB; end

    nexttile; hold on;

    % optional: light red bands for detected bursts within the window
    if draw_bands
        % convert to seconds & keep those overlapping the window
        starts_s = burst_starts ./ sampling_rate;
        ends_s   = burst_ends   ./ sampling_rate;
        keep = ends_s >= win(1) & starts_s <= win(2);
        for b = find(keep(:))'
            x1 = max(starts_s(b), win(1));
            x2 = min(ends_s(b),   win(2));
            patch([x1 x2 x2 x1], [0 0 1 1]*max(1,numel(neurons_to_plot))+1, ...
                  [1 0 0], 'FaceAlpha', 0.08, 'EdgeColor', 'none');
        end
    end

    % plot rasters
    ytick_labels = zeros(numel(neurons_to_plot),1);
    for i = 1:numel(neurons_to_plot)
        rn = neurons_to_plot(i);                       % row index / neuron ID
        st = resultArray_wt(rn,:);                     % seconds
        st = st(st>0 & st>=win(1) & st<=win(2));       % keep spikes in window
        if ~isempty(st)
            scatter(st, ones(size(st))*i, 8, 'k', '.');
        end
        ytick_labels(i) = rn;                          % keep original ID label
    end

    xlim(win);
    ylim([0.5, max(1,numel(neurons_to_plot)) + 0.5]);
    yticks(1:numel(neurons_to_plot));
    if ~isempty(neurons_to_plot)
        yticklabels(string(ytick_labels));
    else
        yticklabels([]);
    end
    box on; grid on;
    xlabel('Time (s)');
    ylabel('Neuron ID (row in resultArray\_wt)');
    title(sprintf('First %d neurons (ID asc)  |  [%g, %g] s', ...
        numel(neurons_to_plot), win(1), win(2)));

    hold off;
end


%%
%% === DROP-IN: Replot panels 2 & 3 for the same two time windows ===
% Prereqs:
%   d                         -> raw LFP array (channels x samples at 2500 Hz)
%   swd_events (cell)         -> SWD sample indices at 2500 Hz
%   original_sampling_rate    -> 2500
% Optional:
%   winA, winB                -> time windows (sec) defined earlier

if ~exist('winA','var'), winA = [0 100]; end
if ~exist('winB','var'), winB = [350 450]; end

fs = original_sampling_rate;     % 2500 Hz
lfp_mean = mean(d, 1);           % average LFP across channels

% --- Recompute filtered LFP (5–60 Hz) in µV (matches your original panel 2) ---
[bb, aa] = butter(4, [5 60] / (fs/2), 'bandpass');
filtered_5_60 = filtfilt(bb, aa, lfp_mean);
scale_factor_uV = 0.195;                       % your conversion factor
filtered_uV = filtered_5_60 * scale_factor_uV; % µV
t = (1:numel(filtered_uV)) / fs;               % seconds

% Helper to plot SWD overlays (matches your original panel 3)
plot_swd_overlays = @(win) ...
    arrayfun(@(k) ...
        ( ...
            plot((swd_events{k}/fs), filtered_5_60(swd_events{k}), 'r', 'LineWidth', 1.5) ...
        ), ...
        find(~cellfun(@isempty, swd_events)) ...
    );

% ------------ Figure: 2 rows (panel 2 then 3) × 2 columns (winA, winB) ------------
figure('Name','Panels 2 & 3 with matching timelines','Position',[80 80 1200 500]);
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

wins = [winA; winB];
titlestr = {@() sprintf('Filtered LFP (5–60 Hz)  [%g, %g] s', winA(1), winA(2)), ...
            @() sprintf('Filtered LFP (5–60 Hz)  [%g, %g] s', winB(1), winB(2)); ...
            @() sprintf(' Detection  [%g, %g] s', winA(1), winA(2)), ...
            @() sprintf(' Detection  [%g, %g] s', winB(1), winB(2))};

for col = 1:2
    win = wins(col, :);

    % ----- Panel 2 (Filtered LFP) -----
    nexttile( (col==1)*1 + (col==2)*2 );  % tiles 1,2
    plot(t, filtered_uV, 'b');
    xlim(win); ylim([-2500 2500]);  % same y-lims you used
    xlabel('Time (s)'); ylabel('Amplitude (\muV)');
    title(titlestr{1, col}()); grid on; box on;

    % ----- Panel 3 (SWD Detection) -----
    nexttile( (col==1)*3 + (col==2)*4 );  % tiles 3,4
    plot(t, filtered_5_60, 'Color', [0.6 0.6 0.6]); hold on;
    plot_swd_overlays(win);  % draw red segments at SWD sample indices
    hold off;
    xlim(win); ylim([-2e4 2e4]);          % your original limits
    xlabel('Time (s)'); ylabel('Amplitude');
    title(titlestr{2, col}()); grid on; box on;
end





















%% === DROP-IN: downsample raster only within [379 387] and replot two windows ===
%% === FIXED DROP-IN: downsample (x5) & randomize in multiple windows, then replot ===
% Needs: resultArray_wt, and either neurons_to_plot or sync_neurons

ds_factor = 5;   % keep ~1/5 of spikes in each window
ds_windows = [ ...
    864   876  ;
    878   883;
    885   893  ;
    896   911  ;
    914   920  ;
    923   929  ;
    931   942   ;
    944   948  ;
    951   957];

% plotting windows
if ~exist('winA','var'), winA = [0 100]; end
if ~exist('winB','var'), winB = [350 450]; end

rng(1); % reproducible

% pick neurons to plot
if ~exist('neurons_to_plot','var') || isempty(neurons_to_plot)
    if exist('sync_neurons','var') && ~isempty(sync_neurons)
        neurons_to_plot = sync_neurons(:);
    else
        neurons_to_plot = (1:size(resultArray_wt,1))';
    end
end

% build modified spike trains (fixed logic)
mod_spikes = cell(numel(neurons_to_plot),1);

for i = 1:numel(neurons_to_plot)
    rn = neurons_to_plot(i);
    st = resultArray_wt(rn,:);           % seconds
    st = st(st > 0);                      % remove padding
    if isempty(st), mod_spikes{i} = []; continue; end

    % accumulate which spikes fall in ANY ds window
    mask_all = false(size(st));
    kept_rand = [];

    for w = 1:size(ds_windows,1)
        w1 = ds_windows(w,1); w2 = ds_windows(w,2);

        mask_w = (st >= w1) & (st <= w2);  % mask in THIS window (on original st)
        nin = sum(mask_w);
        if nin > 0
            nkeep = max(0, round(nin/ds_factor));  % ~1/ds_factor kept
            if nkeep > 0
                % randomize the kept spikes uniformly inside THIS window
                kept_rand = [kept_rand, w1 + (w2 - w1) * rand(1, nkeep)]; %#ok<AGROW>
            end
        end

        % grow the union mask (remove once after loop)
        mask_all = mask_all | mask_w;
    end

    % remove ALL in-window spikes once, then add randomized kept ones
    st_out = st(~mask_all);
    st_mod = sort([st_out(:); kept_rand(:)]);
    mod_spikes{i} = st_mod(:).';
end

% optional quick sanity print for the first neuron shown
if ~isempty(neurons_to_plot)
    rn1 = neurons_to_plot(1);
    st1  = resultArray_wt(rn1,:); st1  = st1(st1>0);
    st1m = mod_spikes{1};
    before = arrayfun(@(k) sum(st1  >= ds_windows(k,1) & st1  <= ds_windows(k,2)), 1:size(ds_windows,1));
    after  = arrayfun(@(k) sum(st1m >= ds_windows(k,1) & st1m <= ds_windows(k,2)), 1:size(ds_windows,1));
    fprintf('Neuron %d: spikes per window BEFORE: [%s]\n', rn1, num2str(before));
    fprintf('Neuron %d: spikes per window AFTER : [%s]\n', rn1, num2str(after));
end

% ---------- replot rasters ----------
figure('Name','Raster with downsampled & randomized windows (fixed)','Position',[120 120 1100 450]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
wins = [winA; winB];

for w = 1:2
    win = wins(w,:);
    nexttile; hold on;

    for i = 1:numel(neurons_to_plot)
        st_mod = mod_spikes{i};
        if isempty(st_mod), continue; end
        st_win = st_mod(st_mod >= win(1) & st_mod <= win(2));
        if ~isempty(st_win)
            y_i = ones(size(st_win))*i;
            scatter(st_win, y_i, 8, 'k', '.');

            % save the exact dots used
            dotX{w,i} = st_win;
            dotY{w,i} = y_i;
        end
    end

    xlim(win);
    ylim([0.5, max(1,numel(neurons_to_plot)) + 0.5]);
    yticks(1:numel(neurons_to_plot));
    yticklabels(string(neurons_to_plot(:)));
    box on; grid on;
    xlabel('Time (s)');
    ylabel('Neuron ID (row in resultArray\_wt)');
    title(sprintf('Raster [%g, %g] s | %d windows @ x%d downsample', win(1), win(2), size(ds_windows,1), ds_factor));
    hold off;
end


for w = 1:2
    allX{w} = [dotX{w,:}]; %#ok<AGROW>
    allY{w} = [dotY{w,:}]; %#ok<AGROW>
end

%%
% allX{w} = 1×N vector of spike times (sec)
% allY{w} = 1×N vector of row indices (1..numNeuronsShown)

figure('Name','Replot from allX/allY');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

for w = 1:numel(allX)
    X = allX{w};
    Y = allY{w};

    nexttile; hold on;
    scatter(X, Y, 8, 'k', '.');   % <— the dots

    % Axes limits using ONLY allX/allY
    if ~isempty(X)
        xlim([min(X) max(X)]);
    end
    if ~isempty(Y)
        ylim([0.5, max(Y)+0.5]);
    else
        ylim([0.5, 1.5]);
    end

    grid on; box on;
    xlabel('Time (s)'); ylabel('Neuron row');
    title(sprintf('Window %d', w));
    hold off;
end

%%

%% === DROP-IN (same Y axes across windows) ===

%% === DROP-IN (same Y axes across windows) ===
%% === DROP-IN: Raster (top) + Population histogram (bottom) from allX/allY ===
% Needs: allX{w} (times, s) and allY{w} (row indices). Optional: wins(w,:) = [tmin tmax].

if ~exist('allX','var') || ~exist('allY','var')
    error('allX/allY not found. Build them first.');
end

bin_summary = 0.1;        % <-- histogram bin (seconds). Try 0.02–0.1 as you like.
use_population_rate = false;  % false: raw counts; true: convert to spikes/s

figure('Name','Raster + Population Summary','Position',[80 80 1200 520]);
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

hist_axes = gobjects(1, numel(allX));
hist_ylim_max = 0;

for w = 1:numel(allX)
    X = allX{w};
    Y = allY{w};

    % ----- decide time window -----
    if exist('wins','var') && size(wins,1) >= w
        win = wins(w,:);
    else
        win = [min(X) max(X)];
    end

    % ================= Top: replot raster quick (for context) =================
    nexttile(w); hold on;
    if ~isempty(X)
        scatter(X, Y, 4, 'k', '.');   % small black dots
        xlim(win);
        ylim([0.5, max(1,max(Y))+0.5]);
        yticks(0:5:max(Y));  % lighter tick density
    else
        plot(0,0,'.'); xlim([0 1]); ylim([0.5 1.5]);
    end
    box on; grid on;
    xlabel('Time (s)'); ylabel('Neuron row');
    title(sprintf('Window %d', w));
    hold off;

    % ================= Bottom: population histogram =================
    edges = win(1):bin_summary:win(2);
    if edges(end) < win(2), edges = [edges win(2)]; end
    centers = (edges(1:end-1) + edges(2:end))/2;

    counts = histcounts(X, edges);          % all neurons combined
    if use_population_rate
        yvals = counts / bin_summary;       % spikes/s (population)
        ylab  = 'Spikes/s (population)';
    else
        yvals = counts;                     % raw counts
        ylab  = 'Spikes (population)';
    end

    hist_axes(w) = nexttile(w+2); hold on;
    bar(centers, yvals, 1.0, 'FaceColor','k', 'EdgeColor','none'); % width=1 fills bin
    xlim(win);
    grid on; box on;
    xlabel('Time (s)'); ylabel(ylab);
    title(sprintf('Window %d: population histogram (bin=%.2fs)', w, bin_summary));
    hold off;

    hist_ylim_max = max(hist_ylim_max, max(yvals,[],'omitnan'));
end

% --- unify histogram y-limits across windows ---
if hist_ylim_max <= 0, hist_ylim_max = 1; end
for w = 1:numel(allX)
    ylim(hist_axes(w), [0, hist_ylim_max*1.10]);
end


%%
function [swd_events_filtered, keep_idx] = curate_swd_events( ...
    lfp_data, fs, swd_events, win_sec, timepoint_array, neuron_id, opts)

    if nargin < 4 || isempty(win_sec), win_sec = 10; end
    if nargin < 5, timepoint_array = []; end
    if nargin < 6, neuron_id = []; end
    if nargin < 7 || isempty(opts), opts = struct(); end

    % -------- Options --------
    opt.max_neurons   = ifdef(opts,'max_neurons', 200);
    opt.marker        = ifdef(opts,'marker','.');
    opt.markersize    = ifdef(opts,'markersize',6);
    opt.show_raster   = ifdef(opts,'show_raster', true);
    opt.bin_w         = ifdef(opts,'bin_w', 0.05);     % <-- summation bin width (s)

    % -------- Input reshape / guards --------
    lfp_data = lfp_data(:);
    N = numel(lfp_data);
    M = numel(swd_events);
    if M == 0
        warning('swd_events 为空，直接返回空结果。');
        swd_events_filtered = {};
        keep_idx = false(0,1);
        return;
    end

    % ---- 事件头尾与长度 ----
    starts = zeros(M,1); ends = zeros(M,1); lens = zeros(M,1);
    for i = 1:M
        ev = swd_events{i};
        starts(i) = max(1, ev(1));
        ends(i)   = min(N, ev(end));
        lens(i)   = ends(i) - starts(i) + 1;
    end

    % ---- 栅格数据预处理 ----
    haveRaster = ~isempty(timepoint_array);
    if haveRaster
        nNeurons = size(timepoint_array,1);
        spikeCell = cell(nNeurons,1);
        for r = 1:nNeurons
            v = timepoint_array(r,:);
            v = v(~isnan(v));
            spikeCell{r} = v(:);    % 这里假设单位是“秒”
        end
        if isempty(neuron_id), neuron_id = (1:nNeurons).'; end
        plot_rows = min(opt.max_neurons, numel(spikeCell));
        row_idx   = (1:plot_rows).';
    else
        nNeurons = 0; row_idx = []; spikeCell = {};
    end

    % ====== 状态 ======
    S.idx         = 1;
    S.keep        = true(M,1);
    S.view_global = false;
    S.show_raster = opt.show_raster;
    S.win_half    = max(0.5, win_sec/2);
    S.fs          = fs;
    S.lfp         = lfp_data;
    S.starts      = starts; S.ends = ends; S.lens = lens;
    S.M = M; S.N = N;
    S.swd_events  = swd_events;
    S.ylim_fix    = [-2e4 2e4];

    S.haveRaster  = haveRaster;
    S.spikeCell   = spikeCell;
    S.row_idx     = row_idx;
    S.neuron_id   = neuron_id;

    % Summation binning
    S.bin_w       = opt.bin_w;

    % ====== 播放相关（数值输入：Hz）======
    S.play        = false;
    S.play_hz     = 1.0;     % 默认 1 Hz（每秒前进 1 个事件）
    S.play_min_hz = 0.1;
    S.play_max_hz = 20.0;
    S.timer       = [];
    S.min_dt      = 0.05;    % 定时器最小周期(秒)

    % ====== Figure / GUI ======
    S.fig = figure('Name','SWD curation','Color','w', ...
        'NumberTitle','off','Units','normalized','Position',[0.08 0.08 0.84 0.78], ...
        'WindowKeyPressFcn', @(~,e) keyHandler(e), ...
        'CloseRequestFcn',   @(~,~) onClose(true));

    % 上：LFP；中：Raster；下：Summation（每 bin 的总 spike 数）
    S.ax1 = axes('Parent',S.fig,'Position',[0.07 0.44 0.90 0.47]);
    S.ax2 = axes('Parent',S.fig,'Position',[0.07 0.27 0.90 0.13]);
    S.ax3 = axes('Parent',S.fig,'Position',[0.07 0.18 0.90 0.07]); % summation
    if ~S.haveRaster
        set(S.ax2,'Visible','off');
        set(S.ax3,'Visible','off');
    end

    % ---- 底部按钮（两行）----
    mk = @(str,pos,cb) uicontrol(S.fig,'Style','pushbutton','String',str, ...
        'Units','normalized','Position',pos,'Callback',cb, ...
        'KeyPressFcn',@(h,e) keyHandler(e),'FontSize',10);

    % 第一行
    y1 = 0.065; h1 = 0.07; w = 0.11; gap = 0.025; x = 0.07;
    mk('<< 上一个 (A)', [x         y1 w h1], @(~,~) prevEv());     x = x + w + gap;
    mk('下一个 (D) >>', [x         y1 w h1], @(~,~) nextEv());     x = x + w + gap;
    mk('保留 (K/Y)',    [x         y1 w h1], @(~,~) keepThis(true));x = x + w + gap;
    mk('删除 (Del/X)',  [x         y1 w h1], @(~,~) keepThis(false));x = x + w + gap;
    mk('切换 (T)',      [x         y1 0.09 h1], @(~,~) toggleKeep()); x = x + 0.09 + gap;
    mk('宽窗 +1s (W)',  [x         y1 0.12 h1], @(~,~) widenWin());   x = x + 0.12 + gap;
    mk('窄窗 -1s (S)',  [x         y1 0.12 h1], @(~,~) narrowWin());  x = x + 0.12 + gap;
    mk('全局/局部 (V)', [x         y1 0.12 h1], @(~,~) toggleGlobal());

    % 第二行（播放控制）
    y2 = 0.005; h2 = 0.05;
    S.btnPlay = uicontrol(S.fig,'Style','togglebutton','String','▶ 播放 (Space)', ...
        'Units','normalized','Position',[0.07 y2 0.16 h2], ...
        'Callback',@(h,~) togglePlay(), 'KeyPressFcn',@(h,e) keyHandler(e), ...
        'FontSize',10);

    uicontrol(S.fig,'Style','text','String','速度 (Hz)：', ...
        'Units','normalized','Position',[0.25 y2 0.09 h2], ...
        'BackgroundColor','w','HorizontalAlignment','left','FontSize',10);

    S.edtHz = uicontrol(S.fig,'Style','edit','String',num2str(S.play_hz,'%.3g'), ...
        'Units','normalized','Position',[0.34 y2 0.10 h2], ...
        'Callback',@(h,~) setSpeedFromEdit(), 'KeyPressFcn',@(h,e) keyHandler(e), ...
        'FontSize',10);

    S.btnApplyHz = uicontrol(S.fig,'Style','pushbutton','String','应用', ...
        'Units','normalized','Position',[0.45 y2 0.08 h2], ...
        'Callback',@(h,~) setSpeedFromEdit(), 'KeyPressFcn',@(h,e) keyHandler(e), ...
        'FontSize',10);

    if S.haveRaster
        mk('显/隐 Raster (H)', [0.87 y2 0.12 h2], @(~,~) toggleRaster());
    end

    S.txt = uicontrol(S.fig,'Style','text','Units','normalized',...
        'Position',[0.07 0.93 0.90 0.05],'BackgroundColor','w',...
        'HorizontalAlignment','left','FontSize',11,'String','');

    % 初始绘制 + 等待
    drawEvent();
    uiwait(S.fig);

    % 输出
    if isvalidStruct(S) && isfield(S,'keep')
        keep_idx = S.keep(:);
    else
        keep_idx = true(M,1);
    end
    swd_events_filtered = swd_events(keep_idx);
    try
        assignin('base','swd_events_filtered', swd_events_filtered);
        assignin('base','swd_keep_idx', keep_idx);
    catch
    end

    % ========= 内部函数 =========
    function drawEvent()
        if ~ishandle(S.ax1), return; end
        i = S.idx; t = (1:S.N)/S.fs;

        % ---- LFP 轴 ----
        cla(S.ax1); hold(S.ax1,'on');
        if S.view_global
            plot(S.ax1, t, S.lfp, 'Color', [0.7 0.7 0.7]);
            xs = [S.starts(i) S.ends(i)]/S.fs;
            patch(S.ax1,[xs(1) xs(2) xs(2) xs(1)], ...
                         [S.ylim_fix(1) S.ylim_fix(1) S.ylim_fix(2) S.ylim_fix(2)], ...
                         [1 0.8 0.8],'EdgeColor','none','FaceAlpha',0.35);
            xlim(S.ax1, [t(1) t(end)]);
            ylim(S.ax1, S.ylim_fix);
        else
            center = round((S.starts(i)+S.ends(i))/2);
            halfN  = round(S.win_half * S.fs);
            lo = max(1, center - halfN);
            hi = min(S.N, center + halfN);

            plot(S.ax1, t(lo:hi), S.lfp(lo:hi), 'Color', [0.6 0.6 0.6]);
            ev_lo = max(S.starts(i), lo); ev_hi = min(S.ends(i), hi);
            if ev_hi >= ev_lo
                idx = ev_lo:ev_hi;
                plot(S.ax1, t(idx), S.lfp(idx), 'r','LineWidth',1.5);
            end
            xlim(S.ax1, [t(lo) t(hi)]);
            ylim(S.ax1, S.ylim_fix);
        end
        ylabel(S.ax1,'Amplitude'); title(S.ax1,'SWD Curation'); grid(S.ax1,'on');

        % ---- Raster 轴 ----
        if S.haveRaster
            cla(S.ax2); hold(S.ax2,'on');
            if S.view_global
                text(0.02,0.5,'全局视图下隐藏 raster（按 V 切回局部）',...
                    'Units','normalized','Parent',S.ax2);
                set(S.ax2,'YTick',[]); xlim(S.ax2, [t(1) t(end)]);
                cla(S.ax3); set(S.ax3,'Visible','off');  % summation 也隐藏
            else
                set(S.ax3,'Visible','on');
                xl = xlim(S.ax1);
                tlo = xl(1); thi = xl(2);

                % 绘制 raster
                for r = 1:numel(S.row_idx)
                    rr = S.row_idx(r);
                    v = S.spikeCell{rr};
                    if isempty(v), continue; end
                    mask = (v >= tlo) & (v <= thi);
                    if any(mask)
                        x = v(mask); y = r * ones(sum(mask),1);
                        plot(S.ax2, x, y, opt.marker, 'MarkerSize', opt.markersize, 'Color', [0.1 0.1 0.1]);
                    end
                end
                xlim(S.ax2, [tlo thi]);
                ylim(S.ax2, [0 max(1,numel(S.row_idx))+1]);
                set(S.ax2,'YDir','normal'); grid(S.ax2,'on');
                ylabel(S.ax2, sprintf('Neurons (1..%d)', numel(S.row_idx)));
                set(S.ax2,'XTickLabel',[]); % X 轴刻度留给下方 summation

                % ---- Summation 轴（0.05 s bin 默认）----
                cla(S.ax3); hold(S.ax3,'on');
                edges = tlo:S.bin_w:thi;
                if numel(edges) < 2
                    edges = linspace(tlo, thi, max(2, ceil((thi-tlo)/S.bin_w)));
                end
                counts = zeros(1, numel(edges)-1);
                for r = 1:numel(S.row_idx)
                    rr = S.row_idx(r);
                    v = S.spikeCell{rr};
                    if isempty(v), continue; end
                    mask = (v >= tlo) & (v <= thi);
                    if any(mask)
                        counts = counts + histcounts(v(mask), edges);
                    end
                end
                ctrs = (edges(1:end-1) + edges(2:end))/2;
                stairs(S.ax3, ctrs, counts, 'LineWidth',1.2);
                xlim(S.ax3, [tlo thi]);
                grid(S.ax3,'on');
                ylabel(S.ax3, 'Spikes/bin');
                xlabel(S.ax3, sprintf('Time (s)  |  bin=%.3g s', S.bin_w));
            end

            if ~S.show_raster
                set(S.ax2,'Visible','off');
                set(S.ax3,'Visible','off');
            else
                set(S.ax2,'Visible','on');
                if ~S.view_global, set(S.ax3,'Visible','on'); end
            end
        end

        dur_s = S.lens(i)/S.fs;
        set(S.txt,'String',sprintf(['Event %d/%d | start=%.3fs, end=%.3fs, dur=%.3fs ',...
            '| keep=%d | win_total=%.2fs | 全局=%d | raster=%d (绘制 %d/%d) | bin=%.3gs | 速度=%.3g Hz | 播放=%d'],...
            i, S.M, S.starts(i)/S.fs, S.ends(i)/S.fs, dur_s, S.keep(i), ...
            2*S.win_half, S.view_global, S.show_raster, numel(S.row_idx), nNeurons, ...
            S.bin_w, S.play_hz, S.play));

        set([S.ax1 S.ax2 S.ax3],'HitTest','off','PickableParts','none');
    end

    % ---- 基本操作 ----
    function nextEv()
        S.idx = min(S.M, S.idx+1);
        drawEvent();
        if S.idx >= S.M && S.play
            S.play = false; updatePlayButton(); stopTimer();
        end
    end
    function prevEv(),     S.idx = max(1, S.idx-1);   drawEvent(); end
    function keepThis(tf), S.keep(S.idx) = tf;        drawEvent(); end
    function toggleKeep(), S.keep(S.idx) = ~S.keep(S.idx); drawEvent(); end
    function widenWin(),   S.win_half = min(30, S.win_half + 0.5); drawEvent(); end
    function narrowWin(),  S.win_half = max(0.5, S.win_half - 0.5); drawEvent(); end
    function toggleGlobal(), S.view_global = ~S.view_global; drawEvent(); end
    function toggleRaster(), S.show_raster = ~S.show_raster; drawEvent(); end

    % ---- 播放控制（数值输入框）----
    function togglePlay()
        S.play = ~S.play;
        updatePlayButton();
        if S.play, startTimer(); else, stopTimer(); end
    end

    function updatePlayButton()
        if S.play
            set(S.btnPlay,'Value',1,'String','⏸ 暂停 (Space)');
        else
            set(S.btnPlay,'Value',0,'String','▶ 播放 (Space)');
        end
    end

    function setSpeedFromEdit()
        str = strtrim(get(S.edtHz,'String'));
        v = str2double(str);
        if isnan(v) || ~isfinite(v)
            v = S.play_hz; % 保留原值
        end
        v = max(S.play_min_hz, min(S.play_max_hz, v));
        S.play_hz = v;
        set(S.edtHz,'String',num2str(S.play_hz,'%.3g'));
        if S.play
            startTimer(); % 运行中动态更新周期
        end
        drawEvent();
    end

    function startTimer()
        stopTimer();
        dt = max(S.min_dt, 1.0 / S.play_hz);    % 周期 = 1 / Hz
        S.timer = timer('ExecutionMode','fixedSpacing','Period',dt, ...
                        'TimerFcn',@(~,~) nextEv(), ...
                        'StartDelay',dt,'Tag','SWD_Autoplay_Hz');
        try, start(S.timer); catch, end
    end

    function stopTimer()
        if ~isempty(S.timer) && isvalid(S.timer)
            try, stop(S.timer); delete(S.timer); catch, end
        end
        S.timer = [];
    end

    % ---- 键盘 ----
    function keyHandler(e)
        if isempty(e) || ~isfield(e,'Key'), return; end
        switch lower(e.Key)
            case {'rightarrow','d','n'}, nextEv();
            case {'leftarrow','a','p'},  prevEv();
            case {'k','y','accept'},     keepThis(true);
            case {'x','r','delete','backspace'}, keepThis(false);   % 删除事件
            case {'t'},                  toggleKeep();
            case {'w'},                  widenWin();
            case {'s'},                  narrowWin();
            case {'v'},                  toggleGlobal();
            case {'h'},                  toggleRaster();
            case {'space'}               % 播放/暂停
                togglePlay();
            case {'g'}
                answ = inputdlg({'跳到事件 #（1~M）:'},'Go to',1,{num2str(S.idx)});
                if ~isempty(answ)
                    j = str2double(answ{1});
                    if ~isnan(j) && j>=1 && j<=S.M
                        S.idx = round(j); drawEvent();
                    end
                end
            case {'return','enter'}
                onClose(true); % 确认退出
            case {'escape','q'}
                choice = questdlg('放弃更改并退出？（将保留全部事件）','确认','是','否','否');
                if strcmp(choice,'是')
                    S.keep(:) = true;
                    onClose(true);
                end
        end
    end

    % ---- 关闭清理 ----
    function onClose(confirmDone)
        stopTimer();
        if ishandle(S.fig)
            try, uiresume(S.fig); catch, end
            delete(S.fig);
        end
    end
end

% ===== Helpers =====
function v = ifdef(s, f, d)
    if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end

function tf = isvalidStruct(S)
    tf = ~isempty(S) && isstruct(S) && isfield(S,'keep');
end
