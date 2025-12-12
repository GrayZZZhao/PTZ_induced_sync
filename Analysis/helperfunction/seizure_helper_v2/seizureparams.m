%loading
loading_lfp;

%detection 

detect_swd(mean(d));


% Initialize arrays to store the onset times and lengths of each seizure event
onset_times = zeros(1, length(swd_events));
event_lengths = zeros(1, length(swd_events));

% Extract the onset times (first time point) and lengths of each seizure event
for i = 1:length(swd_events)
    onset_times(i) = swd_events{i}(1);
    event_lengths(i) = length(swd_events{i});
end

% Calculate the time differences between consecutive seizure event onsets
time_between_onsets = diff(onset_times);

% Display the results
disp('Time between consecutive seizure event onsets:');
disp(time_between_onsets);

disp('Length of each seizure event:');
disp(event_lengths);

%% Snip segments from mean_lfp

% Define the length of the time frame to snip (7500 frames)
time_frame_length = 7500;

% Initialize a cell array to store the snipped segments from mean_lfp
snipped_segments_mean_lfp = cell(1, length(swd_events));

mean_lfp = mean(d);  % Calculate the mean LFP across channels

% Extract the onset times and snip the segments from mean_lfp
for i = 1:length(swd_events)
    onset_time = swd_events{i}(1);
    
    % Ensure the snip does not exceed the length of mean_lfp
    if onset_time + time_frame_length - 1 <= length(mean_lfp)
        snipped_segments_mean_lfp{i} = mean_lfp(onset_time:(onset_time + time_frame_length - 1));
    else
        % If the segment would exceed the data length, snip only until the end of the data
        snipped_segments_mean_lfp{i} = mean_lfp(onset_time:end);
    end
end

% Display a message confirming the snipping process is complete
disp('Snipping of segments from mean_lfp is complete.');

% % Plot each snipped segment from mean_lfp
for i = 1:length(snipped_segments_mean_lfp)
    figure;  % Create a new figure window
    plot(snipped_segments_mean_lfp{i});  % Plot the current snipped segment
    xlabel('Time (frames)');
    ylabel('Amplitude');
    title(['Snipped Segment from mean\_lfp ', num2str(i)]);
    
    % Pause and wait for a key press to move to the next figure
    disp('Press any key to continue to the next figure...');
    pause;
    
    % Close the current figure after moving to the next
    close(gcf);
end

disp('All mean_lfp segments have been displayed.');

%% Snip segments from all channels in d

% Initialize a cell array to store the snipped segments across all channels
snipped_segments_all_channels = cell(1, length(swd_events));
event_timeframe = cell(1, length(swd_events));

% Extract the onset times and snip the segments from d for all channels
for i = 1:length(swd_events)
    onset_time = swd_events{i}(1);
    
    % Ensure the snip does not exceed the length of d
    if onset_time + time_frame_length - 1 <= size(d, 2)
        snipped_segments_all_channels{i} = d(:, onset_time:(onset_time + time_frame_length - 1));
        event_timeframe{i} = onset_time:(onset_time + time_frame_length - 1);
    else
        % If the segment would exceed the data length, snip only until the end of the data
        snipped_segments_all_channels{i} = d(:, onset_time:end);
        event_timeframe{i} = onset_time:size(d, 2);
    end
end

% Display a message confirming the snipping process is complete
disp('Snipping of segments across all channels is complete.');

% Save the snipped segments across all channels to a MATLAB file
%save('snipped_segments_all_channels.mat', 'snipped_segments_all_channels');

% Plot each snipped segment across all channels
% for i = 1:length(snipped_segments_all_channels)
%     figure;  % Create a new figure window
%     plot(snipped_segments_all_channels{i}');  % Plot the current snipped segment for all channels
%     xlabel('Time (frames)');
%     ylabel('Amplitude');
%     title(['Snipped Segment across all channels ', num2str(i)]);
%     
%     % Pause and wait for a key press to move to the next figure
%     disp('Press any key to continue to the next figure...');
%     pause;
%     
%     % Close the current figure after moving to the next
%     close(gcf);
% end

disp('All segments across all channels have been displayed.');

%% Load the "action potential" data and the unitType variable
ap_data_file = 'E:\seizure\2024-01-02_WT_HOM-male-adult\2024-01-02_13-43-59\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP/ap_data.npy';  % Replace with the actual path to ap_data.npy
unitType_file = 'E:\seizure\2024-01-02_WT_HOM-male-adult\2024-01-02_13-43-59\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP/unit1.mat';  % Replace with the actual path to unitType.mat

% Load the action potential data using readNPY function
action_potential = readNPY(ap_data_file);  % Do not convert to double, keep original precision
action_potential = reshape(action_potential,384,[]);
% No conversion to double to avoid memory issues

% Load the unitType variable
load(unitType_file);  % Assuming unitType is stored in a .mat file and contains channel information

% Extract the channel numbers from the 3rd column of unitType
channel_numbers = Unit_1(:, 3);  % Assuming unitType is a matrix or table and the third column has the channel numbers
 channel_numbers_real = channel_numbers+1;
% Extract the fr_ap values from the 6th column of unitType
fr_ap = Unit_1(:, 6);  % The firing rates or other relevant data from the 6th column

%% Sampling Rates and Snipping Parameters
lfp_fs = 2500;  % Sampling rate of LFP data
ap_fs = 30000;  % Sampling rate of "action potential" data

% Define the length of the time frame to snip (7500 frames in LFP corresponds to a different length in the action potential data)
time_frame_length_lfp = 7500;  % Length of the snipped segment in LFP frames
time_frame_length_ap = time_frame_length_lfp * (ap_fs / lfp_fs);  % Adjust length for action potential sampling rate

% Initialize a cell array to store the snipped segments from "action potential"
snipped_segments_ap = cell(1, length(swd_events));

%% Extract the onset times from the LFP data and convert them to the "action potential" time frame
for i = 1:length(swd_events)
    onset_time_lfp = swd_events{i}(1);  % Onset time in LFP data (2500 Hz)
    
    % Convert the onset time to "action potential" frame (30000 Hz)
    onset_time_ap = round(onset_time_lfp * (ap_fs / lfp_fs));
    
    % Ensure the snip does not exceed the length of the "action potential" data
    if onset_time_ap + time_frame_length_ap - 1 <= size(action_potential, 2)
        % Use only the specific channels from unitType (3rd column)
        snipped_segments_ap{i} = action_potential(channel_numbers_real, onset_time_ap:(onset_time_ap + time_frame_length_ap - 1));
         snipped_time_frames{i} = onset_time_ap:(onset_time_ap + time_frame_length_ap - 1);
    else
        % If the segment would exceed the data length, snip only until the end of the data
        snipped_segments_ap{i} = action_potential(channel_numbers_real, onset_time_ap:end);
        snipped_time_frames{i} = onset_time_ap:size(action_potential, 2);
    end
end

% Display a message confirming the snipping process is complete
disp('Snipping of segments from action potential data is complete.');

% % Save the snipped segments for "action potential" data
 %save('snipped_segments_ap.mat', 'snipped_segments_ap', '-v7.3');
 %save('firingrate.mat','fr_ap');
 save('snipped_time_frames.mat','snipped_time_frames');
 disp('snipped_segments_ap has been successfully saved.');
