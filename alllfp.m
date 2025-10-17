%% Filter Settings (Same as Before)
fs = 2500;
low_cutoff = 5;
high_cutoff = 60;
[b, a] = butter(4, [low_cutoff, high_cutoff] / (fs / 2), 'bandpass');

offset_value = 10000;  % Offset for visualization
time_vector = (0:size(d, 2)-1) / fs;

%% Pre-allocate for Storing Filtered LFPs
filtered_lfps = zeros(size(d));  % 384 x n matrix

%% Filter All Channels
for channel_id = 1:384
    lfp_data = d(channel_id, :);
    filtered_lfps(channel_id, :) = filtfilt(b, a, lfp_data);
end

%% Plot All Filtered LFP Traces with Offset
figure;
ax1 = axes;
hold on;
for channel_id = 1:384
    plot(time_vector, filtered_lfps(channel_id, :) + (channel_id - 1) * offset_value, 'b');
end
xlabel('Time (s)');
ylabel('Electrode (Offset)');
title('Filtered LFP Traces (5–60 Hz) Across All Electrodes');
xlim([min(time_vector), max(time_vector)]);
hold off;

%% Detect Outlier Channels (based on variance of filtered LFP)
channel_variances = var(filtered_lfps, 0, 2);  % Variance of each channel's signal
mean_variance = mean(channel_variances);
std_variance = std(channel_variances);

% Threshold: Z-score > 2 as outliers
z_scores = (channel_variances - mean_variance) / std_variance;
outlier_channels = find(abs(z_scores) > 2);  % Outlier detection based on variance

%% Display Detected Outlier Channels
disp('Detected Outlier Channels (by variance z-score > 2):');
disp(outlier_channels);

%% Plot Outliers Separately (Optional for Checking)
figure;
hold on;
for idx = 1:length(outlier_channels)
    channel_id = outlier_channels(idx);
    plot(time_vector, filtered_lfps(channel_id, :) + (idx - 1) * offset_value, 'r');
end
xlabel('Time (s)');
ylabel('Outlier Electrode (Offset)');
title('Detected Outlier LFP Traces (5–60 Hz)');
xlim([min(time_vector), max(time_vector)]);
hold off;


