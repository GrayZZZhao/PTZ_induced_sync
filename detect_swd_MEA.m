function swd_events = detect_swd_MEA(lfp_data, fs)
    % Input:
    % lfp_data - LFP signal data (1D array)
    % fs - Sampling frequency (in Hz)

    % Output:
    % swd_events - Detected SWD events (start and end times in seconds)

    % 1. Bandpass Filter (5 to 40 Hz) to isolate the SWD frequency range
    low_cutoff = 5;
    high_cutoff = 40;
    [b, a] = butter(4, [low_cutoff high_cutoff] / (fs / 2), 'bandpass');
    filtered_lfp = filtfilt(b, a, lfp_data);

    % 2. Compute the signal envelope using Hilbert Transform
    analytic_signal = hilbert(filtered_lfp);
    amplitude_envelope = abs(analytic_signal);

    % 3. Define amplitude threshold for SWD detection
    threshold = mean(amplitude_envelope) + 3 * std(amplitude_envelope);  % Threshold based on 3 standard deviations

    % 4. Find points where the envelope exceeds the threshold
    swd_indices = find(amplitude_envelope > threshold);

    % 5. Group contiguous points into SWD events
    swd_events = {};
    current_event = [];

    for i = 1:length(swd_indices)
        if isempty(current_event) || swd_indices(i) == current_event(end) + 1
            current_event = [current_event, swd_indices(i)];
        else
            % Store the current event
            swd_events{end+1} = current_event;
            current_event = swd_indices(i);
        end
    end
    if ~isempty(current_event)
        swd_events{end+1} = current_event;
    end

    % 6. Convert indices to time (in seconds)
    swd_events_time = cellfun(@(event) [event(1)/fs, event(end)/fs], swd_events, 'UniformOutput', false);

    % 7. Filter out short events (e.g., less than 0.3 seconds)
    min_duration = 0.3; % Minimum event duration in seconds
    swd_events = swd_events_time(cellfun(@(event) diff(event) > min_duration, swd_events_time));

    % 8. Plot the LFP signal with detected SWD events
    plot_swd_with_events(lfp_data, swd_events, fs);

    % 9. Spectrogram visualization
    plot_swd_spectrogram(filtered_lfp, fs);

end

function plot_swd_with_events(lfp_data, swd_events, fs)
    % Plot LFP signal with detected SWD events
    time_vector = (1:length(lfp_data)) / fs;
    
    figure;
    plot(time_vector, lfp_data, 'Color', [0.6 0.6 0.6]); % LFP signal in gray
    hold on;

    % Highlight detected SWD events
    for i = 1:length(swd_events)
        event = swd_events{i};
        event_indices = round(event * fs);  % Convert times to indices
        plot(time_vector(event_indices(1):event_indices(2)), lfp_data(event_indices(1):event_indices(2)), 'r', 'LineWidth', 2);
    end
    
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Detected SWD Events in LFP Signal');
    hold off;
end

function plot_swd_spectrogram(filtered_lfp, fs)
    % Spectrogram of the filtered LFP data
    window_size = 1 * fs;  % 1-second window
    noverlap = 0.5 * window_size;  % 50% overlap
    nfft = 2^nextpow2(window_size);  % FFT length

    % Compute and plot the spectrogram
    [S, F, T, P] = spectrogram(filtered_lfp, window_size, noverlap, nfft, fs);

    % Convert power to dB
    PdB = 10 * log10(P);

    figure;
    imagesc(T, F, PdB);
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Spectrogram of Filtered LFP Signal');
    
    % Colormap: dark blue to red
    colormap(jet);
    colorbar;
end
