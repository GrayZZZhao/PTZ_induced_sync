Code with further modification: 

 
 

Manually Screening for LFP signals: 
Screen shots from Notion Notebooks: 
 

 
 
 
 
 

 
 
 
 
 
 
Code for plotting these figures: (Built this summer)

clear; clc; close all;

%% ------------------ Inputs (edit these) --------------------------
% If you already have lfp (T x C) and fs in your workspace, comment out this demo block.
if ~exist('lfp','var') || ~exist('fs','var')
    fs   = 2500;                 % sampling rate (Hz)
    Tsec = 170;                  % total duration for the demo (s)
    C    = 24;                   % number of channels for the demo
    t    = (0:1/fs:Tsec).';      % time axis
    % Demo LFP: 1/f-ish noise + occasional bursts around 140–148 s
    lfp  = 30*filter(1,[1 -0.98],randn(numel(t),C)); % colored noise (µV)
    burst_t = t>=140 & t<=148;
    for k = 1:C
        lfp(burst_t,k) = lfp(burst_t,k) + 120*sin(2*pi*(8+0.2*k)*t(burst_t)) ...
            .* exp(-((t(burst_t)-142-0.08*k)/2.0).^2);
    end
end

% Channel groups to plot (edit to match your probe map)
ch_groups = {11:21, 1:10};   % first figure: 11–21; second figure: 1–10

% Time window in seconds to display
win_sec = [130 150];

% Optional: event times (sec). Leave empty if you don’t want markers.
event_times = [137.5 143.0];   % e.g., onset / offset; set [] to disable

% Band-pass range (Hz)
bp = [5 80];

% Vertical spacing between traces (µV). If [], it will auto-scale.
y_gap = [];   % e.g., 800; leave empty for auto

%% ------------------ Precompute indices & filter -------------------
% Clip the window to recording length
N = size(lfp,1);
win_idx = max(1, round(win_sec*fs));
win_idx(2) = min(N, win_idx(2));
twin = (win_idx(1):win_idx(2)).'/fs;  % time axis in s

% Design a zero-phase Butterworth band-pass filter
[b,a] = butter(4, bp/(fs/2), 'bandpass');

% Helper to build one stacked panel
make_panel = @(ax, chans) plot_group(ax, lfp, chans, twin, win_idx, b, a, fs, y_gap, bp, event_times);

%% ------------------ Figure 1: channels 11–21 ---------------------
figure('Color','w','Position',[100 100 900 420]);
ax1 = axes; hold(ax1,'on');
make_panel(ax1, ch_groups{1});
title(ax1, sprintf('Filtered LFP Trace (%.0f–%.0f Hz) · Channels: %s', ...
    bp(1), bp(2), strjoin(string(ch_groups{1}), ' ')));

%% ------------------ Figure 2: channels 1–10 ----------------------
figure('Color','w','Position',[100 560 900 420]);
ax2 = axes; hold(ax2,'on');
make_panel(ax2, ch_groups{2});
title(ax2, sprintf('Filtered LFP Trace (%.0f–%.0f Hz) · Channels: %s', ...
    bp(1), bp(2), strjoin(string(ch_groups{2}), ' ')));

%% ======================== Functions ==============================
function plot_group(ax, lfp, chans, twin, win_idx, b, a, fs, y_gap, bp, event_times)
    % Extract the windowed data for the selected channels
    X = lfp(win_idx(1):win_idx(2), chans);

    % Band-pass filter, zero-phase
    Xf = filtfilt(b, a, X);

    % Remove per-channel DC for nicer stacking
    Xf = Xf - mean(Xf,1,"omitnan");

    % Choose vertical spacing
    if isempty(y_gap)
        % Auto: 1.2 * 95th percentile of abs amplitude across channels
        amp = prctile(abs(Xf), 95, 1);
        y_gap = 1.2 * median(amp);
        if y_gap <= 0, y_gap = 100; end
    end

    % Build stacked offsets (top trace highest)
    nCh = numel(chans);
    offsets = (nCh-1:-1:0) * y_gap;     % row vector

    % Plot each channel as a stacked line
    hold(ax,'on');
    for i = 1:nCh
        plot(ax, twin, Xf(:,i) + offsets(i), 'LineWidth', 1.0);
        % Optional: label channel number at the left edge
        text(ax, twin(1), offsets(i), sprintf('%d', chans(i)), ...
            'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
            'FontSize', 8, 'Color', [0.2 0.2 0.2]);
    end

    % Event markers (vertical lines)
    if ~isempty(event_times)
        for t0 = event_times(:).'
            xline(ax, t0, ':', 'Color', [0.2 0.2 0.2], 'LineWidth', 1);
        end
    end

    % Axes cosmetics
    xlim(ax, [twin(1) twin(end)]);
    ylim(ax, [-y_gap*0.6, offsets(1)+y_gap*0.6]);
    xlabel(ax, 'Time (s)');
    ylabel(ax, 'Amplitude (\muV) + offset');
    box(ax,'on');
    grid(ax,'off');

    % Improve tick density similar to your screenshot
    ax.YTick = []; % hide stacked absolute values (we already label channels)
end
