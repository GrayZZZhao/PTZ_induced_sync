function corrected_lfp = baseline_correction(lfp_data, fs)
    % High-pass filter to remove baseline drift
    low_cutoff = 0.1;  % Set the low cutoff frequency for baseline correction (e.g., 0.1 Hz)
    [b, a] = butter(2, low_cutoff / (fs / 2), 'high'); % Create high-pass filter
    corrected_lfp = filtfilt(b, a, lfp_data); % Apply the filter
end
