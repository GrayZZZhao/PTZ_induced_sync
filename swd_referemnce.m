%% Load Your LFP Data
% 'd' should already be loaded: 384 x n matrix
fs = 2500;  % Sampling rate

%% Subtract First Electrode as Reference
% Generate new D: referenced LFP data
D = d - d(1, :);  % Subtract first electrode from all electrodes

%% Define Channels to Plot (Channels 2 to 50)
channels_to_plot = 320:340;
num_channels = length(channels_to_plot);

%% Prepare for Plotting
offset_value = 10000;  % Vertical spacing between traces
time_vector = (0:size(D, 2)-1) / fs;

%% Plot Referenced LFP Signals with Offset
figure;
hold on;
for idx = 1:num_channels
    channel_id = channels_to_plot(idx);
    lfp_data = D(channel_id, :);  % Use referenced data
    
    % Plot with vertical offset
    plot(time_vector, lfp_data + (idx - 1) * offset_value, 'k');  % Black lines
end
xlabel('Time (s)');
ylabel('Electrode (Offset)');
title('Referenced LFP Traces (Electrodes 320-340, First Electrode Subtracted)');
xlim([min(time_vector), max(time_vector)]);
hold off;

% Adjust figure size (optional)
set(gcf, 'Position', [100, 100, 800, 800]);
