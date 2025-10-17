 
 
 
Fig 1 sup overall
 
Code for PSTH plots 
%% ======================= Demo data (replace) =======================
clear; clc; close all;

fs           = 30000;                 % sampling rate of spikes (Hz)
nNeurons     = 60;                    % number of units
nTrials      = 40;                    % trials per unit
alignTimes   = 0.0 + (0:nTrials-1)*3; % event time per trial (s), demo spacing
win          = [-1.5 2.0];            % PSTH window around event (s)
bin          = 0.05;                  % PSTH bin width (s)
smoothSigma  = 2;                     % smoothing in bins (Gaussian sigma)

% --- Demo spike times for each neuron (cell of 1×nNeurons, each is 1×Nspikes in seconds)
spikeTimes = cell(1,nNeurons);
rng(7);
for k = 1:nNeurons
    % baseline Poisson (5 Hz) + event-locked bump (k-dependent)
    base = sort(rand(poissrnd(5*nTrials*3),1)*nTrials*3);   % over full duration
    bump = [];
    for tr = 1:nTrials
        t0 = alignTimes(tr);
        % bump around t0 + ~0.2 s; strength varies slightly by neuron
        nb = poissrnd(8 + 2*sin(k/9));
        bump = [bump ; t0 + 0.2 + 0.15*randn(nb,1)];
    end
    spikeTimes{k} = sort([base; bump]);
end

% --- Demo proportions for pie chart
counts = [28 14 10 6 2];  % e.g., decreased/unchanged/increased/other categories
labels = {'Decrease','Slight ↓','Unchanged','Slight ↑','Increase'};

%% ======================= PSTH (peristimulus time histogram) ========
% Returns time centers, mean rate across neurons, and SEM.
[tPSTH, meanFR, semFR] = make_psth(spikeTimes, alignTimes, win, bin, smoothSigma);

figure('Color','w','Position',[100 100 520 360]); hold on;
shaded_band(tPSTH, meanFR-semFR, meanFR+semFR, [0.85 0.9 1.0]); % SEM band
plot(tPSTH, meanFR, 'k', 'LineWidth', 1.8);                      % mean trace
xline(0,'--','Event','LabelHorizontalAlignment','left','Color',[0.2 0.2 0.2]);

xlabel('Time from event (s)');
ylabel('Firing rate (Hz)');
title(sprintf('PSTH (%d neurons, %d trials, bin=%.0f ms)', nNeurons, nTrials, bin*1000));
xlim(win);
box on;

%% ======================= Pie chart =================================
figure('Color','w','Position',[660 100 420 360]);
p = pie(counts);
% Add percentage labels using category names
th = findobj(p,'Type','text');
vals = counts/sum(counts)*100;
for i = 1:numel(th)
    th(i).String = sprintf('%s: %.1f%%', labels{i}, vals(i));
    th(i).FontSize = 10;
end
title('Category composition');

% Optional custom colors (uncomment to set your own)
% ax = gca; patches = findobj(ax,'Type','patch');
% set(patches, {'FaceColor'}, num2cell([0.80 0.30 0.30;
%                                       0.95 0.55 0.55;
%                                       0.70 0.70 0.70;
%                                       0.55 0.75 0.95;
%                                       0.30 0.50 0.90],2));

%% ======================= Helper functions ==========================
function [tCenters, meanFR, semFR] = make_psth(spikeTimes, alignTimes, win, bin, smoothSigma)
% make_psth
% Inputs:
%   spikeTimes  : 1×N cell, each cell holds spike times in seconds for one neuron
%   alignTimes  : 1×T vector of event times (s), one per trial
%   win         : [tmin tmax] window around event (s)
%   bin         : scalar bin width (s)
%   smoothSigma : Gaussian sigma in bins (set [] to disable)
% Outputs:
%   tCenters    : bin centers (s)
%   meanFR      : mean firing rate across neurons (Hz)
%   semFR       : standard error of the mean across neurons (Hz)
    edges    = win(1):bin:win(2);
    tCenters = edges(1:end-1) + bin/2;
    nNeurons = numel(spikeTimes);
    T        = numel(alignTimes);
    frMat    = zeros(nNeurons, numel(tCenters));
    % Build counts per neuron across all trials aligned to each event
    for n = 1:nNeurons
        % Concatenate spikes relative to each event
        relSpikes = [];
        for tr = 1:T
            rel = spikeTimes{n} - alignTimes(tr);
            relSpikes = [relSpikes; rel(rel>=win(1) & rel<win(2))];
        end
        counts = histcounts(relSpikes, edges);  % total spikes across trials per bin
        fr     = counts / (T*bin);              % convert to Hz
        if ~isempty(smoothSigma) && smoothSigma>0
            fr = smoothdata(fr, 'gaussian', max(3, round(6*smoothSigma)));
        end
        frMat(n,:) = fr;
    end
    meanFR = mean(frMat, 1, 'omitnan');
    semFR  = std(frMat, 0, 1, 'omitnan') / sqrt(nNeurons);
end
function shaded_band(x, y1, y2, col)
% shaded_band  Plot a filled band between y1 and y2
    if nargin<4, col = [0.9 0.9 0.9]; end
    x = x(:); y1 = y1(:); y2 = y2(:);
    fill([x; flipud(x)], [y1; flipud(y2)], col, ...
        'EdgeColor','none','FaceAlpha',0.6);
end

