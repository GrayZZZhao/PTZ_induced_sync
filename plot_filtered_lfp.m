function [A] = plot_filtered_lfp(lfp_data, fs)
    % Bandpass Filter (5 to 60 Hz)
    low_cutoff = 5;   % Set a small non-zero value for the lower cutoff
    high_cutoff = 60;
    [b, a] = butter(4, [low_cutoff high_cutoff] / (fs / 2), 'bandpass');
    filtered_lfp = filtfilt(b, a, lfp_data);

    % Apply the scale factor to convert to microvolts (µV)
    scale_factor = 0.195;  % Conversion scale factor
    filtered_lfp_microV = filtered_lfp * scale_factor;
    
    A = filtered_lfp_microV;

    % Create a time vector
    time_vector = (1:length(filtered_lfp)) / fs;

    % Plot the filtered LFP signal in microvolts
    figure;
    plot(time_vector, filtered_lfp_microV, 'b');
    xlabel('Time (s)');
    ylabel('Amplitude (µV)');
    title('Filtered LFP Signal (5-60 Hz) in µV');
    ylim([-2500 2500]);
    %grid on;
end
