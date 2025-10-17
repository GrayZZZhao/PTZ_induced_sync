%% Load Your LFP Data
% 'd' is 384 x n, already loaded
fs = 2500;  % Sampling rate

%% Define Electrodes to Plot (Example: Channels 1 to 50)
channels_to_plot = 1:50;  % Adjust if you want a different set of channels
num_channels = length(channels_to_plot);

%% Prepare for Plotting
offset_value = 10000;  % Vertical offset to separate traces
time_vector = (0:size(d, 2)-1) / fs;

%% Plot Raw LFP Signals with Offset
figure;
hold on;
for idx = 1:num_channels
    channel_id = channels_to_plot(idx);
    lfp_data = d(channel_id, :);  % Raw LFP signal (no filtering)
    
    % Plot with vertical offset
    plot(time_vector, lfp_data + (idx - 1) * offset_value, 'k');  % 'k' = black lines
end
xlabel('Time (s)');
ylabel('Electrode (Offset)');
title(['Raw LFP Traces - Electrodes ', num2str(channels_to_plot(1)), ' to ', num2str(channels_to_plot(end))]);
xlim([min(time_vector), max(time_vector)]);
hold off;

% Adjust figure size (optional)
set(gcf, 'Position', [100, 100, 800, 800]);
