% Load the calcium traces data from an Excel file
% Prompt user for the Excel file name
[fileName, pathName] = uigetfile('D:\Gray\PTZ\*.xlsx', 'Select an Excel File');
if isequal(fileName, 0)
    disp('User selected Cancel');
    return;
else
    disp(['User selected ', fullfile(pathName, fileName)]);
end

% Remove the file extension to create the folder name
[~, folderName, ~] = fileparts(fileName);

% Create a new subfolder under the specified path
outputFolder = fullfile(pathName, folderName, 'Processed_Figures');
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Load the 'roiIntensity' data from the selected Excel file
filePath = fullfile(pathName, fileName);
calcium_traces = readmatrix(filePath);

% Calculate the correlation matrix
correlation_matrix = corr(calcium_traces');
calcium_traces_flip = calcium_traces;

%% Create a mask to hide the upper triangle
mask = triu(true(size(correlation_matrix)), 1);

%% Plot the heatmap using imagesc
figure;
imagesc(correlation_matrix);
colorbar;
title('Functional Connectivity Matrix HOMPTZ', 'FontSize', 20);
xlabel('Brain Regions', 'FontSize', 15);
ylabel('Brain Regions', 'FontSize', 15);
colormap('jet');
caxis([-1 1]); % Set color axis limits for better visualization

% Save the plot as MATLAB figure and PNG image
savefig(fullfile(outputFolder, 'Functional_Connectivity_Matrix.fig'));
saveas(gcf, fullfile(outputFolder, 'Functional_Connectivity_Matrix.png'));

%% Plot calcium traces of all neurons
[num_neurons, num_timepoints] = size(calcium_traces_flip);

% Generate time vector assuming each column represents a time point
time = 1:num_timepoints;

% Create a figure
figure;
hold on;
offset = 1;

% Plot each neuron's calcium trace
for i = 1:num_neurons
    plot(time, calcium_traces_flip(i, :) + (i-1) * offset);
end

% Customize the plot
title('Calcium Traces of All Neurons WT Base');
xlabel('Time (arbitrary units)');
ylabel('Fluorescence Intensity');
legend(arrayfun(@(x) sprintf('Neuron %d', x), 1:num_neurons, 'UniformOutput', false));
hold off;

% Save the plot as MATLAB figure and PNG image
savefig(fullfile(outputFolder, 'Calcium_Traces.fig'));
saveas(gcf, fullfile(outputFolder, 'Calcium_Traces.png'));

%% Analyze elements greater than 0.8 in the correlation matrix

% Find indices of elements greater than 0.8
[row, col] = find(correlation_matrix > 0.8);

% Number of elements greater than 0.8
num_elements = numel(row);

% Display the results
disp('Positions (row, col) of elements greater than 0.8:');
disp([row, col]);

disp(['Number of elements greater than 0.8: ', num2str(num_elements)]);

%% Proportion of off-diagonal elements greater than 0.8, 0.7, 0.6, and 0.5

% Remove diagonal by setting diagonal elements to NaN
correlation_matrix_duo = correlation_matrix;
correlation_matrix_duo(logical(eye(size(correlation_matrix_duo)))) = NaN;

% Total number of off-diagonal elements
total_A = sum(~isnan(correlation_matrix_duo(:)))/2;  % Total number of non-NaN elements in correlation_matrix_duo

% Calculate proportion of off-diagonal elements greater than 0.8
a_08 = sum(correlation_matrix_duo(:) > 0.8, 'omitnan');  % Count elements greater than 0.8
prop_A_08 = a_08 / total_A;  % Calculate proportion
disp(['Proportion of off-diagonal elements > 0.8 in the correlation matrix: ', num2str(prop_A_08)]);

% Calculate proportion of off-diagonal elements greater than 0.7
a_07 = sum(correlation_matrix_duo(:) > 0.7, 'omitnan');  % Count elements greater than 0.7
prop_A_07 = a_07 / total_A;  % Calculate proportion
disp(['Proportion of off-diagonal elements > 0.7 in the correlation matrix: ', num2str(prop_A_07)]);

% Calculate proportion of off-diagonal elements greater than 0.6
a_06 = sum(correlation_matrix_duo(:) > 0.6, 'omitnan');  % Count elements greater than 0.6
prop_A_06 = a_06 / total_A;  % Calculate proportion
disp(['Proportion of off-diagonal elements > 0.6 in the correlation matrix: ', num2str(prop_A_06)]);

% Calculate proportion of off-diagonal elements greater than 0.5
a_05 = sum(correlation_matrix_duo(:) > 0.5, 'omitnan');  % Count elements greater than 0.5
prop_A_05 = a_05 / total_A;  % Calculate proportion
disp(['Proportion of off-diagonal elements > 0.5 in the correlation matrix: ', num2str(prop_A_05)]);

%% Calculate 2.a, 2.b, and 2.c

% Extract off-diagonal elements
off_diag_elements = correlation_matrix_duo(~isnan(correlation_matrix_duo));

% 2.a Mean correlation coefficient
mean_corr = mean(off_diag_elements);
disp(['Mean correlation coefficient (off-diagonal): ', num2str(mean_corr)]);

% 2.b Standard deviation of the off-diagonal elements
std_corr = std(off_diag_elements);
disp(['Standard deviation of off-diagonal elements: ', num2str(std_corr)]);

% 2.c Skewness of the off-diagonal elements    
skew_corr = skewness(off_diag_elements);
disp(['Skewness of off-diagonal elements: ', num2str(skew_corr)]);

%% Save prop_A, mean_corr, std_corr, and skew_corr into an Excel file

% Create a table to save all results
results_table = table(prop_A_08,prop_A_07,prop_A_06,prop_A_05, mean_corr, std_corr, skew_corr, ...
    'VariableNames', {'Proportion_OffDiagonal_0.8', 'Proportion_OffDiagonal_0.7','Proportion_OffDiagonal_0.6','Proportion_OffDiagonal_0.5','Mean_Correlation', 'Std_Correlation', 'Skewness_Correlation'});

% Specify the file path
excelFilePath = fullfile(outputFolder, 'Proportion_OffDiagonal_Elements.xlsx');

% Save the results to Excel file
writetable(results_table, excelFilePath, 'Sheet', 1, 'Range', 'A1');
disp(['Results saved in Excel file at: ', excelFilePath]);
