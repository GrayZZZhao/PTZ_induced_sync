function plot_swd_spectrogram(lfp_data, fs)
    % Bandpass filter (5 to 40 Hz)
    low_cutoff = 5;
    high_cutoff = 40;
    [b, a] = butter(4, [low_cutoff high_cutoff] / (fs / 2), 'bandpass');
    filtered_lfp = filtfilt(b, a, lfp_data);
    
    % Parameters for the spectrogram
    window_size = 1 * fs; % 1-second window
    noverlap = 0.5 * window_size; % 50% overlap
    nfft = 2^nextpow2(window_size); % FFT length

    % Compute and plot the spectrogram
    [S, F, T, P] = spectrogram(filtered_lfp, window_size, noverlap, nfft, fs);
    
    % Convert power to dB
    PdB = 10 * log10(P);

    % Plot the spectrogram
    figure;
    imagesc(T, F, PdB);
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Spectrogram of SWD Events');
    
    colormap(jet)
  
    colorbar;
end


