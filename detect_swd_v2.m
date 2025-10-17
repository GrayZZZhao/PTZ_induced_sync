%GZ swd detection 20240321
function swd_events = detect_swd_v2(lfp_data)
    % Load LFP data
    %lfp_data = load(file_path);
    
    % Assume the data is stored in a variable called 'lfp_data'
    if isstruct(lfp_data)
        field_names = fieldnames(lfp_data);
        lfp_data = lfp_data.(field_names{1});
    end
    
    % Sampling frequency (in Hz)
    fs = 2500; % Adjust as per your data
    
    % Preprocess LFP data
    filtered_lfp = preprocess_lfp(lfp_data, fs);
    
    % Define amplitude threshold for SWD detection
    threshold = std(filtered_lfp) * 1.5; %  adjust 1 was 5
    
    % Detect SWD events
    swd_events = detect_swd_events(filtered_lfp, fs, threshold);
    
    % Plot the LFP signal with detected SWD events
    plot_swd_events(lfp_data, swd_events, fs);

    assignin('base', 'swd_events', swd_events);

end

function filtered_lfp = preprocess_lfp(lfp_data, fs)
    % Bandpass filter the LFP signal (1-100 Hz)
    low_cutoff = 6;
    high_cutoff = 12;
    [b, a] = butter(2, [low_cutoff high_cutoff] / (fs / 2), 'bandpass');%% 
    filtered_lfp = filtfilt(b, a, lfp_data);
end
% Wn = [low_cutoff high_cutoff] / (fs / 2) 表示保留频段，然后转化成0-1 喂送给 butter 
% changed from [1 100]



function swd_events = detect_swd_events(filtered_lfp, fs, threshold)
    % Compute the envelope of the signal to detect high-amplitude events
    analytic_signal = hilbert(filtered_lfp);
    %amplitude_envelope = abs(analytic_signal);
    amplitude_env     = abs(analytic_signal);
    % Define SWD based on amplitude threshold
    env_s             = movmedian(amplitude_env, max(1, round(0.03*fs)));  % 30 ms
    swd_indices = find(env_s > threshold);
    
    % Group contiguous indices into individual SWD events
    swd_events = {};
    current_event = [];
    
    for i = 1:length(swd_indices)
        if isempty(current_event) || swd_indices(i) == current_event(end) + 1
            current_event = [current_event, swd_indices(i)];
        else
            swd_events{end+1} = current_event;
            current_event = swd_indices(i);
        end
    end
    
    if ~isempty(current_event)
        swd_events{end+1} = current_event;
    end
    
    % Filter out short events (e.g., less than 0.5 seconds)
    min_duration =0.2 * fs;  % adjust 3 
    swd_events = swd_events(cellfun(@(event) length(event) > min_duration, swd_events));
end


% function plot_swd_events(lfp_data, swd_events, fs)
%     figure;
%     plot(lfp_data, 'Color', [0.6 0.6 0.6]); % Plot the LFP signal in a lighter color
%     hold on;
%     
%     for i = 1:length(swd_events)
%         event = swd_events{i};
%         plot(event, lfp_data(event), 'r', 'LineWidth', 2); % Plot SWD events in red with thicker lines
%     end
%     
%     xlabel('Time (samples)');
%     ylabel('Amplitude');
%     title('SWD Detection in LFP Signal');
%     ylim([-20000 20000]); % Set y-axis limits
%     hold off;
% end

function plot_swd_events(lfp_data, swd_events, fs)
    % Create time vector in seconds
    time_in_seconds = (1:length(lfp_data)) / fs;
    
    figure;
    plot(time_in_seconds, lfp_data, 'Color', [0.6 0.6 0.6]); % Plot the LFP signal in a lighter color
    hold on;
    
    for i = 1:length(swd_events)
        event = swd_events{i};
        event_time = event / fs; % Convert event indices to time in seconds
        plot(event_time, lfp_data(event), 'r', 'LineWidth', 2); % Plot SWD events in red with thicker lines
    end
    
    xlabel('Time (seconds)'); % Update x-axis label
    ylabel('Amplitude');
    title('SWD Detection in LFP Signal');
    ylim([-20000 20000]); % Set y-axis limits
    hold off;
end


