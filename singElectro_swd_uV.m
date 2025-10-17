%% Load Your LFP Data
% 'd' should already be loaded: 384 x n matrix
d = d - d(364, :);
fs = 2500;  % Sampling rate

%% Define Electrodes to Plot
%channels_to_plot = [ 50, 100, 150, 200, 250, 300, 350, 70, 140];  
channels_to_plot = 10:20;  
num_channels = length(channels_to_plot);

%% Filter Settings
low_cutoff = 5;
high_cutoff = 60;
[b, a] = butter(4, [low_cutoff, high_cutoff] / (fs / 2), 'bandpass');

%% Prepare for Plotting
offset_value = 500;  % Vertical spacing between traces
scale_factor = 0.194;  % Convert to microvolts (µV)
time_vector = (0:size(d, 2)-1) / fs;

%% Plot Multiple Electrodes on the Same Subplot with Offset (µV)
figure;
ax1 = subplot(2,1,1); % First subplot for LFP traces

hold on;
for idx = 1:num_channels
    channel_id = channels_to_plot(idx);
    lfp_data = d(channel_id, :);
    filtered_lfp = filtfilt(b, a, lfp_data);
    
    % Apply vertical offset and scale to µV
    plot(time_vector, (filtered_lfp * scale_factor) + (idx - 1) * offset_value, 'b');
end
xlabel('Time (s)');
ylabel('Amplitude (\muV) + Offset');
title(['Filtered LFP Traces (5–60 Hz, Scaled to µV) - Electrodes: ', num2str(channels_to_plot)]);
xlim([min(time_vector), max(time_vector)]);
hold off;

%% Spectrogram of One Representative Channel (e.g., Channel 10)
channel_for_spectrogram = channels_to_plot(1);  % First channel in your list
lfp_data_spec = d(channel_for_spectrogram, :);
filtered_lfp_spec = filtfilt(b, a, lfp_data_spec);

window_size = 1 * fs;          
noverlap = 0.5 * window_size;  
nfft = 2^nextpow2(window_size);

[S, F, T, P] = spectrogram(filtered_lfp_spec, window_size, noverlap, nfft, fs);
PdB = 10 * log10(P);
freq_limit = F <= 60;
F_limited = F(freq_limit);
S_limited = PdB(freq_limit, :);

%% Plot Spectrogram (Second Subplot)
ax2 = subplot(2,1,2);
imagesc(T, F_limited, S_limited);
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title(['Spectrogram (5–60 Hz) - Channel ', num2str(channel_for_spectrogram)]);
colormap(jet);
colorbar;
xlim([min(time_vector), max(time_vector)]);

%% Link X-axes
linkaxes([ax1, ax2], 'x');

% Adjust figure size
set(gcf, 'Position', [100, 100, 800, 600]);
