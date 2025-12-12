
% Create a folder within the current directory
output_folder_name = 'seizure_output';
current_folder = pwd;  % Get the current working directory
output_folder_path = fullfile(current_folder, output_folder_name);

% Check if the folder already exists, if not, create it
if ~exist(output_folder_path, 'dir')
    mkdir(output_folder_path);
end
% Save the figure as a MATLAB figure (.fig) in the new folder
%savefig(fullfile(output_folder_path, 'swd detection in LFP.fig'));

% Save the figure as a TIFF image (.tif) in the new folder
saveas(gcf, fullfile(output_folder_path, 'swd detection in LFPt.tif'));

save(fullfile(output_folder_path, 'swd_events.mat'), 'swd_events');