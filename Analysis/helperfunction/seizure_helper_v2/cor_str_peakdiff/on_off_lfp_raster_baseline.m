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
spike_time_wt = readNPY('D:\npxl_kv11\2025-04-21_1750_594_HOM_kv11-female-adult\2025-04-21_13-18-58\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\spike_times.npy');
spike_time_full_wt = double(spike_time_wt) / 30000;
spike_templates_wt = readNPY('D:\npxl_kv11\2025-04-21_1750_594_HOM_kv11-female-adult\2025-04-21_13-18-58\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\spike_templates.npy');
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

