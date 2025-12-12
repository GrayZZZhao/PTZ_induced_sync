%% =========================================================
% ????? Bombcell GUI ? “unit 275” ? clusterID=275 ???
% ???workspace ?? base_path, spike_time_wt, qMetric, ephysData
% =========================================================
base_path = 'D:\Ylabrecording2025\2024-12-24_1347_WT-male-adult\2024-12-24_16-00-36\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP';
spike_times_path      = fullfile(base_path, 'spike_times.npy');
spike_time_wt = readNPY(spike_times_path);
%% =========================================================
% ????? "?? unit 275" vs "Bombcell GUI ? unit 275"
% =========================================================

% 0) ?? spike_clusters.npy ?Bombcell ?? clusterID?
cluster_path = fullfile(base_path, 'spike_clusters.npy');
if ~exist(cluster_path, 'file')
    error('Cannot find spike_clusters.npy at: %s', cluster_path);
end
spike_clusters = readNPY(cluster_path);
spike_clusters = double(spike_clusters) + 1;   % 0-based -> 1-based

% 1) Bombcell GUI ? unit ?? = qMetric.clusterID
unitList_gui = double(qMetric.clusterID(:));   % 347×1??????? clusterID

gui_index = 275;   % ?? GUI ???? “unit 275”

if gui_index > numel(unitList_gui)
    error('GUI unit index 275 > length(qMetric.clusterID) = %d', numel(unitList_gui));
end

clusterID_gui = unitList_gui(gui_index);   % GUI unit 275 ????? clusterID

fprintf('\n===== Step 1: GUI ?? =====\n');
fprintf('Bombcell GUI ?? "unit 275" -> ?? clusterID = %d\n', clusterID_gui);

% 2) ???????? unit?
clusterID_code = 275;        % ???????? “unit 275”?clusterID=275?
clusterID_bc   = clusterID_gui;  % Bombcell GUI ? “unit 275”

% 3) ??? spike ???? spike_times.npy ?????
idx_code = find(spike_clusters == clusterID_code);
idx_bc   = find(spike_clusters == clusterID_bc);

% ??????????????sampleIndex / 30000?
fs_ap = 30000;
t_code = double(spike_time_wt(idx_code)) / fs_ap;
t_bc   = double(spike_time_wt(idx_bc))   / fs_ap;

fprintf('\n===== Step 2: spike ???? =====\n');
fprintf('?? unit:  clusterID = %d  -> %d spikes\n', clusterID_code, numel(idx_code));
fprintf('GUI  unit:  clusterID = %d  -> %d spikes\n', clusterID_bc,   numel(idx_bc));

% 4) ?? index ???/??
idx_common = intersect(idx_code, idx_bc);
idx_only_code = setdiff(idx_code, idx_bc);
idx_only_bc   = setdiff(idx_bc, idx_code);

fprintf('\n===== Step 3: spike index ??/?? =====\n');
fprintf('?? spikes ??: %d\n', numel(idx_common));
fprintf('?? clusterID %d (?? unit 275) ?? spikes: %d\n', ...
        clusterID_code, numel(idx_only_code));
fprintf('?? clusterID %d (GUI unit 275) ?? spikes: %d\n', ...
        clusterID_bc, numel(idx_only_bc));

% 5) ???????? spike ???????????
maxShow = 10;

if ~isempty(idx_only_code)
    fprintf('\n-- ??: ?? clusterID %d (?? unit 275) ?? spikes (?? %d ?) --\n', ...
            clusterID_code, maxShow);
    nShow = min(maxShow, numel(idx_only_code));
    for k = 1:nShow
        ii = idx_only_code(k);
        t_this = double(spike_time_wt(ii))/fs_ap;
        fprintf('  #%d  index=%d  t=%.4f s\n', k, ii, t_this);
    end
else
    fprintf('\nclusterID %d ????? spikes?????? GUI unit ? cluster ???\n', ...
            clusterID_code);
end

if ~isempty(idx_only_bc)
    fprintf('\n-- ??: ?? clusterID %d (GUI unit 275) ?? spikes (?? %d ?) --\n', ...
            clusterID_bc, maxShow);
    nShow = min(maxShow, numel(idx_only_bc));
    for k = 1:nShow
        ii = idx_only_bc(k);
        t_this = double(spike_time_wt(ii))/fs_ap;
        fprintf('  #%d  index=%d  t=%.4f s\n', k, ii, t_this);
    end
else
    fprintf('\nclusterID %d ?????? spikes???????????\n', ...
            clusterID_bc);
end

fprintf('\n===============================================\n');

% 6) ?????? unit ? raster ???????????/????
figure('Name', sprintf('Your cluster %d vs Bombcell GUI unit 275 (cluster %d)', ...
                       clusterID_code, clusterID_bc), ...
       'Position',[200 200 900 250]);
hold on;
scatter(t_code, ones(size(t_code)), 12, 'r', '.');   % ????? clusterID=275
scatter(t_bc,   2*ones(size(t_bc)),   12, 'k', '.'); % ???GUI ??? cluster
yticks([1 2]);
yticklabels({sprintf('cluster %d (your code)',clusterID_code), ...
             sprintf('cluster %d (GUI unit 275)',clusterID_bc)});
xlabel('Time (s)');
ylabel('Label');
title('?? "unit 275" vs Bombcell GUI ? "unit 275"');
grid on; box on;
hold off;

