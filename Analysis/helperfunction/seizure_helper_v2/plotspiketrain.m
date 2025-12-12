% Assume 'data' is your 384x1000 2D array
spiketrain  = d;  % Example data, replace with your actual data

% Number of columns to plot at a time
cols_per_plot = 10;

% Total number of columns in the data
total_cols = size(spiketrain, 2);

% Calculate the number of groups of 10 columns
num_groups = ceil(total_cols / cols_per_plot);

% Loop through each group and plot the columns
for i = 1:num_groups
    % Calculate the starting and ending column indices for the current group
    start_col = (i - 1) * cols_per_plot + 1;
    end_col = min(i * cols_per_plot, total_cols);
    
    % Extract the subset of columns for the current group
    subset_data = spiketrain(:, start_col:end_col);
    
    % Create a new figure for each group of columns
    figure;
    
    % Create a subplot for each column in the subset
    for j = 1:size(subset_data, 2)
        subplot(10, 1, j); % Arranges 10 subplots in a 2 rows x 5 columns grid
        plot(subset_data(:, j));
        title(sprintf('Column %d', start_col + j - 1));
    end
    
    % Add general title to the figure
    sgtitle(sprintf('Columns %d to %d', start_col, end_col));
    
    % Adjust subplot layouts for better visibility
    xlabel('Row Index');
    ylabel('Value');
    grid on; % Add a grid for better visibility of the plot lines
    
    % Pause between plots if you want to view them one at a time
    pause; % Press any key to continue to the next group of columns
end
