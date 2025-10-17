%% Analysis trace from raw data, created by Yiyong Han, 20181219
clear all;
close all;
clc
% read excel
framerate = 3.26; % Hzc
Time_interval = 60; % for figure plot, unit in second

% Select Excel file
[xlsfilename, xlsfoldername, ~] = uigetfile({'*.xlsx'});
xlsfoldername = xlsfoldername(1:end-1);

mkdir([xlsfoldername '\Processing']);
mkdir([xlsfoldername '\Processing\Accepted Waves']);

% Import data
xlsData_ori = importdata([xlsfoldername '\' xlsfilename]);
N_sheets = length(fieldnames(xlsData_ori));
Name_sheets = fieldnames(xlsData_ori);
xlsData = struct2cell(xlsData_ori);
N_frames = size(xlsData{1}, 1);
N_cell = size(xlsData{1}, 2) - 1;
xvector = (0:N_frames-1) / framerate;

disp('Read xls file done!!');

%% Processing delta F/F for running and rest states
% (this part remains unchanged from your original code)

%% Plotting and manually selecting acceptable waveforms
% Predefine y-axis limits for consistency across all plots
y_axis_limits = [-1, 5];  % Adjust these values based on your data range

% Create a folder to save the accepted waveforms
mkdir([xlsfoldername '\Processing\Accepted Waves']);

% Initialize an empty cell array to store the accepted waveforms
accepted_waves = cell(N_sheets, N_cell);

% Variable to store which waveforms were accepted
accepted_wave_indices = [];

% Loop through each sheet and cell
for id_sheets = 1:N_sheets
    for id_cell = 1:N_cell
        % Plot the deltaF/F waveform
        k = figure(3435);
        plot(xvector, squeeze(deltaFoverF(id_sheets, :, id_cell)));
        title([Name_sheets{id_sheets} ' cell' num2str(id_cell)]);
        xlabel('Time/s');
        ylabel('\DeltaF/F');
        xlim([min(xvector) max(xvector)]);
        ylim(y_axis_limits);  % Fix the y-axis limits for consistent comparison
        set(gca, 'XTick', (min(xvector):Time_interval:round(max(xvector) / Time_interval) * Time_interval));

        % Prompt the user to check if the waveform is acceptable
        choice = questdlg('Is this waveform acceptable?', ...
            'Waveform Selection', ...
            'Yes', 'No', 'Yes');

        % Handle the user response
        if strcmp(choice, 'Yes')
            % Save the accepted waveform to the accepted_waves variable
            accepted_waves{id_sheets, id_cell} = squeeze(deltaFoverF(id_sheets, :, id_cell));
            accepted_wave_indices = [accepted_wave_indices; id_sheets, id_cell];  % Store the index of accepted waveforms

            % Save the figure of the accepted waveform as a .jpg file
            saveas(gcf, [xlsfoldername '\Processing\Accepted Waves\Accepted_deltaFoverF_' Name_sheets{id_sheets} '_cell' num2str(id_cell) '.jpg']);
        end

        % Close the figure after decision
        close(k);
    end
end

%% Save accepted waveforms to a new Excel file
accepted_waves_matrix = accepted_waves(~cellfun('isempty', accepted_waves));  % Filter out empty cells

% Initialize an empty array to store accepted waveforms in a linear format
accepted_data = [];

% Loop through accepted waveforms to prepare them for Excel
for idx = 1:numel(accepted_waves_matrix)
    accepted_data = [accepted_data; accepted_waves_matrix{idx}];  % Concatenate the accepted waveforms
end

% Save accepted waveforms to Excel
filename = [xlsfoldername '\Processing\Accepted Waves\Accepted_Waveforms.xlsx'];
xlswrite(filename, accepted_data);

disp('Accepted waveforms have been saved.');

%% Normalize the accepted data
% Number of cells and frames in accepted data
if isempty(accepted_data)
    disp('No accepted data to process.');
    return;
end

N_cell_accepted = size(accepted_data, 1);  % Each row is a cell
N_frames_accepted = size(accepted_data, 2);  % Each column is a time point

% Initialize a matrix to store the normalized data
normalized_data = zeros(N_cell_accepted, N_frames_accepted);

for ii = 1:N_cell_accepted
    data_ori = accepted_data(ii, :);  % Original data for this cell

    % Normalize each trace: check if max == min to avoid NaN
    if max(data_ori) == min(data_ori)
        data_norm = zeros(size(data_ori));  % If flat, set to zero
    else
        data_norm = (data_ori - min(data_ori)) / (max(data_ori) - min(data_ori));  % Min-max normalization
    end

    normalized_data(ii, :) = data_norm;  % Store normalized data

    % Plot normalized data
    figure;
    plot(data_norm, 'LineWidth', 2);
    title(['Normalized Sequence for Accepted Cell ' num2str(ii)]);
    xlabel('Time (frames)');
    ylabel('Normalized \DeltaF/F');
    ylim([0 1]);

    % Save the figure
    saveas(gcf, [xlsfoldername '\Processing\Normilzed sequence\Accepted_data_norm_cell_' num2str(ii) '.jpg']);
    saveas(gcf, [xlsfoldername '\Processing\Normilzed sequence\Accepted_data_norm_cell_' num2str(ii) '.pdf']);
    close(gcf);
end

% Save normalized data to Excel
filename_norm = [xlsfoldername '\Processing\Accepted Waves\Accepted_Normalized_Waveforms.xlsx'];
xlswrite(filename_norm, normalized_data);

disp('Normalization of accepted data is complete.');
