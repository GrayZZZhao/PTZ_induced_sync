function plot_swd_spectrogram_0_60(filtered_lfp, fs)
    % Spectrogram of the filtered LFP data, limited to 0-60 Hz
    window_size = 1 * fs;  % 1-second window
    noverlap = 0.5 * window_size;  % 50% overlap
    nfft = 2^nextpow2(window_size);  % FFT length

    % Compute the spectrogram
    [S, F, T, P] = spectrogram(filtered_lfp, window_size, noverlap, nfft, fs);

    % Convert power to dB
    PdB = 10 * log10(P);

    % Limit the frequency range to 0-60 Hz
    freq_limit = F <= 60;
    S_limited = PdB(freq_limit, :);
    F_limited = F(freq_limit);

    % Plot the spectrogram
    figure;
    imagesc(T, F_limited, S_limited);
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Spectrogram of Filtered LFP Signal (0-60 Hz)');
    
    % Colormap: dark blue to red
    colormap(jet);
    colorbar;
end
