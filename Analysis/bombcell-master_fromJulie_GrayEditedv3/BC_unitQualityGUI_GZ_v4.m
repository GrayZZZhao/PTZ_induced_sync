%% unit quality gui: plot various quality metric plots for single units
% toggle between units with the right and left arrows
% the slowest part by far of this is plotting the raw data, need to figure out how
% to make this faster 
% to add: 
% - toggle next most similar units (ie space bar)
% - individual raw waveforms 
% - add raster plot
% - click on units 
% - probe locations 

function unitQualityGuiHandle = BC_unitQualityGUI_GZ_v4(memMapData, ephysData, qMetric, param, probeLocation, unitType, plotRaw)

%% set up dynamic figure
unitQualityGuiHandle = figure('color', 'w');
set(unitQualityGuiHandle, 'KeyPressFcn', @KeyPressCb);

%% initial conditions
iCluster = 1;
iCount   = 1;
uniqueTemps    = unique(ephysData.spike_templates);
goodUnit_idx   = find(unitType==1); 
multiUnit_idx  = find(unitType==2); 
noiseUnit_idx  = find(unitType==0); 

%% plot initial conditions
iChunk = 1;
initializePlot(unitQualityGuiHandle, ephysData, qMetric, unitType, uniqueTemps, plotRaw, param)
updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
    probeLocation, unitType, uniqueTemps, iChunk, plotRaw);

% ??????????? guidata????????
guiData = guidata(unitQualityGuiHandle);
guiData.memMapData    = memMapData;
guiData.ephysData     = ephysData;
guiData.qMetric       = qMetric;
guiData.param         = param;
guiData.probeLocation = probeLocation;
guiData.unitType      = unitType;
guiData.uniqueTemps   = uniqueTemps;
guiData.plotRaw       = plotRaw;
guiData.iCluster      = iCluster;
guiData.iChunk        = iChunk;
guidata(unitQualityGuiHandle, guiData);

%% key press callback
    function KeyPressCb(~, evnt)
        if strcmpi(evnt.Key, 'rightarrow')
            iCluster = iCluster + 1;
            if iCluster > numel(uniqueTemps)
                iCluster = numel(uniqueTemps);
            end
            updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
                probeLocation, unitType, uniqueTemps, iChunk, plotRaw);

        elseif strcmpi(evnt.Key, 'g') % next single-unit 
            nextIdx = goodUnit_idx(find(goodUnit_idx>iCluster,1,'first'));
            if ~isempty(nextIdx)
                iCluster = nextIdx;
                updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
                    probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
            end

        elseif strcmpi(evnt.Key, 'm') % next multi-unit 
            nextIdx = multiUnit_idx(find(multiUnit_idx>iCluster,1,'first'));
            if ~isempty(nextIdx)
                iCluster = nextIdx;
                updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
                    probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
            end

        elseif strcmpi(evnt.Key, 'n') % next noise/non-somatic unit
            nextIdx = noiseUnit_idx(find(noiseUnit_idx>iCluster,1,'first'));
            if ~isempty(nextIdx)
                iCluster = nextIdx;
                updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
                    probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
            end

        elseif strcmpi(evnt.Key, 'leftarrow')
            iCluster = max(1, iCluster - 1);
            updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
                probeLocation, unitType, uniqueTemps, iChunk, plotRaw);

        elseif strcmpi(evnt.Key, 'uparrow')
            iChunk = iChunk + 1;
            if iChunk> length(ephysData.spike_times_timeline(ephysData.spike_templates == iCluster))
                iChunk=1;
            end
            updateRawSnippet(unitQualityGuiHandle, memMapData, ephysData, iCluster, iCount, qMetric, param, ...
                probeLocation, uniqueTemps, iChunk, plotRaw);

        elseif strcmpi(evnt.Key, 'downarrow')
            iChunk = iChunk - 1;
            if iChunk==0
                iChunk = length(ephysData.spike_times_timeline(ephysData.spike_templates == iCluster));
            end
            updateRawSnippet(unitQualityGuiHandle, memMapData, ephysData, iCluster, iCount, qMetric, param, ...
                probeLocation, uniqueTemps, iChunk, plotRaw);

        elseif strcmpi(evnt.Key, 'u') % select particular unit 
            ansStr  = inputdlg('Go to unit:');
            if ~isempty(ansStr)
                tmp = str2double(ansStr{1});
                if ~isnan(tmp) && tmp>=1 && tmp<=numel(uniqueTemps)
                    iCluster = tmp;
                    updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
                        probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
                end
            end
        end

        % ?? guidata ???? unit / chunk
        guiData = guidata(unitQualityGuiHandle);
        guiData.iCluster = iCluster;
        guiData.iChunk   = iChunk;
        guidata(unitQualityGuiHandle, guiData);
    end
end

function initializePlot(unitQualityGuiHandle, ephysData, qMetric, unitType, uniqueTemps, plotRaw, param)

%% main title
mainTitle = sgtitle('');

%% 1) units over depth  —— ???????????

subplot(6, 13, [1, 14, 27, 40, 53, 66]);
hold on;

unitCmap = zeros(length(unitType), 3);
unitCmap(unitType == 1, :, :) = repmat([0, 0.5, 0],       length(find(unitType == 1)), 1);
unitCmap(unitType == 0, :, :) = repmat([1, 0, 0],         length(find(unitType == 0)), 1);
unitCmap(unitType == 2, :, :) = repmat([0.29, 0, 0.51],   length(find(unitType == 2)), 1);

norm_spike_n = mat2gray(log10(accumarray(ephysData.spike_templates, 1)+1));

unitDots = scatter(norm_spike_n(uniqueTemps), ...
    ephysData.channel_positions(qMetric.maxChannels(uniqueTemps), 2), ...
    5, unitCmap, 'filled', 'ButtonDownFcn', @unit_click);

currUnitDots = scatter(0, 0, 100, unitCmap(1, :, :), ...
    'filled', 'MarkerEdgeColor', [0, 0, 0], 'LineWidth', 4);

xlim([-0.1, 1.1]);
depth_vals = ephysData.channel_positions(:, 2);
ylim([min(depth_vals) - 50, max(depth_vals) + 50]);  % ???????????

ylabel('Depth (\mum)')
xlabel('Normalized log rate')
title('Location on probe')

%% 2) template waveforms
subplot(6, 13, [2:7, 15:20])
hold on;
max_n_channels_plot = 20;
templateWaveformLines    = arrayfun(@(x) plot(nan(82, 1), nan(82, 1), 'linewidth', 2, 'color', 'k'), 1:max_n_channels_plot);
maxTemplateWaveformLines = arrayfun(@(x) plot(nan(82, 1), nan(82, 1), 'linewidth', 2, 'color', 'b'), 1);
peaks   = scatter(nan(10, 1), nan(10, 1), [], rgb('Orange'), 'v', 'filled');
troughs = scatter(nan(10, 1), nan(10, 1), [], rgb('Gold'),   'v', 'filled');
xlabel('Position+Time');
ylabel('Position');
set(gca, 'YDir', 'reverse')
tempTitle  = title('');
tempLegend = legend([maxTemplateWaveformLines, peaks, troughs, templateWaveformLines(1)], ...
                    {'', '', '',''},'color','none');

%% 3) raw waveforms
subplot(6, 13, [8:13, 21:26])
hold on;
rawWaveformLines    = arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', 'k'), 1:max_n_channels_plot);
maxRawWaveformLines = arrayfun(@(x) plot(nan(82, 1), nan(82, 1), 'linewidth', 2, 'color', 'b'), 1);
set(gca, 'YDir', 'reverse')
xlabel('Position+Time');
ylabel('Position');
rawTitle  = title('');
rawLegend = legend([maxRawWaveformLines], {''},'color','none');

%% 4) ACG
if plotRaw && param.computeDistanceMetrics
    subplot(6, 13, 28:31)
else
    subplot(6, 13, 28:33)
end
hold on;
acgBar     = arrayfun(@(x) bar(0:0.1:25, nan(251, 1)), 1);
acgRefLine = line([NaN, NaN], [NaN, NaN], 'Color', 'r', 'linewidth', 1.2);
acgAsyLine = line([NaN, NaN], [NaN, NaN], 'Color', 'r', 'linewidth', 1.2);
xlabel('time (ms)');
ylabel('sp/s');
acgTitle = title('');

%% 5) ISI
if plotRaw && param.computeDistanceMetrics
    subplot(6, 13, 32:35)
else
    subplot(6, 13, 34:39)
end
hold on;
isiBar     = arrayfun(@(x) bar((0 + 0.25):0.5:(50 - 0.25), nan(100, 1)), 1);
isiRefLine = line([NaN, NaN], [NaN, NaN], 'Color', 'r', 'linewidth', 1.2);
xlabel('Interspike interval (ms)')
ylabel('# of spikes')
isiTitle  = title('');
isiLegend = legend([isiBar], {''});

%% 6) isoDistance
if param.computeDistanceMetrics
    if plotRaw
        subplot(6, 13, 36:39)
    else
        subplot(6, 13, 41:46)
    end
    hold on;
    currIsoD  = scatter(NaN, NaN, 10, '.b'); % this cluster
    rpvIsoD   = scatter(NaN, NaN, 10, '.m'); % rpv spikes
    otherIsoD = scatter(NaN, NaN, 10, NaN, 'o', 'filled');

    colormap(brewermap([], '*YlOrRd'))
    hb = colorbar;
    ylabel(hb, 'Mahalanobis Distance')
    legend('this cluster',  'rpv spikes', 'other clusters');
    isoDTitle = title('');
end

%% 7) raw data
if plotRaw
    rawPlotH = subplot(6, 13, [41:52, 55:59, 60:65]);
    hold on;
    title('Raw unwhitened data')
    set(rawPlotH, 'XColor','w', 'YColor','w')
    rawPlotLines  = arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', 'k'), 1:max_n_channels_plot);
    rawSpikeLines = arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', 'b'), 1:max_n_channels_plot);
    % ??????????? amplitude ????
    rawTimeLine   = line(rawPlotH, [NaN NaN], [NaN NaN], ...
                         'Color', [0 0 1], 'LineWidth', 1.5);
end

%% 8) amplitude * spikes
ampliAx = subplot(6, 13, [67:70, 73:76]);
hold on;
yyaxis left;

% ?? spikes??????? ButtonDownFcn
tempAmpli = scatter(NaN, NaN, 'black', 'filled', ...
    'ButtonDownFcn', @ampli_click);

% ??????? spike?????????
currTempAmpli = scatter(NaN, NaN, 'blue', 'filled', ...
    'ButtonDownFcn', @ampli_click);

% RPV spikes?????????
rpvAmpli= scatter(NaN, NaN, 10, 'magenta', 'filled', ...
    'ButtonDownFcn', @ampli_click);

xlabel('Experiment time (s)');
ylabel('Template amplitude scaling');
axis tight
set(gca, 'YColor', 'k')

yyaxis right
spikeFR = stairs(NaN, NaN, 'LineWidth', 2.0, 'Color', [1, 0.5, 0]);
set(gca, 'YColor', [1, 0.5, 0])
ylabel('Firing rate (sp/sec)');
ampliTitle  = title('');
ampliLegend = legend([tempAmpli,rpvAmpli], {'', ''});

%% 9) amplitude fit
ampliFitAx = subplot(6, 13, [78]);
hold on;
ampliBins = barh(NaN, NaN, 'blue');
ampliBins.FaceAlpha = 0.5;
ampliFit  = plot(NaN, NaN, 'Color', rgb('Orange'), 'LineWidth', 4);
ampliFitTitle  = title('');
ampliFitLegend = legend([ampliFit], {''}, 'Location', 'South');

%% 10) Unit raster —— ?????? thisUnit ? spike_times_timeline
rasterAx = subplot(6,13,71:72);
hold on;
unitRasterScatter = scatter(NaN,NaN,5,'k','filled');
xlabel('Time (s)');
ylabel('Trials');
title('Unit raster');
xlim([0 1]); ylim([0.5 1.5]);
set(rasterAx,'YTick',1,'YTickLabel',{'Trials'});
grid on;
hold off;

%% save handles
guiData = struct;
guiData.mainTitle = mainTitle;

% location plot
guiData.unitDots     = unitDots;
guiData.currUnitDots = currUnitDots;
guiData.unitCmap     = unitCmap;
guiData.norm_spike_n = norm_spike_n;

% template waveforms
guiData.templateWaveformLines    = templateWaveformLines;
guiData.maxTemplateWaveformLines = maxTemplateWaveformLines;
guiData.tempTitle  = tempTitle;
guiData.tempLegend = tempLegend;
guiData.peaks      = peaks;
guiData.troughs    = troughs;

% raw waveforms
guiData.rawWaveformLines    = rawWaveformLines;
guiData.maxRawWaveformLines = maxRawWaveformLines;
guiData.rawTitle  = rawTitle;
guiData.rawLegend = rawLegend;

% ACG
guiData.acgBar     = acgBar;
guiData.acgRefLine = acgRefLine;
guiData.acgAsyLine = acgAsyLine;
guiData.acgTitle   = acgTitle;

% ISI
guiData.isiBar     = isiBar;
guiData.isiRefLine = isiRefLine;
guiData.isiTitle   = isiTitle;
guiData.isiLegend  = isiLegend;

% isoD
if param.computeDistanceMetrics
    guiData.currIsoD  = currIsoD;
    guiData.otherIsoD = otherIsoD;
    guiData.isoDTitle = isoDTitle;
    guiData.rpvIsoD   = rpvIsoD;
end

% raw data
if plotRaw
    guiData.rawPlotH      = rawPlotH;
    guiData.rawPlotLines  = rawPlotLines;
    guiData.rawSpikeLines = rawSpikeLines;
    guiData.rawTimeLine   = rawTimeLine;
end

% amplitudes * spikes
guiData.ampliAx       = ampliAx;
guiData.tempAmpli     = tempAmpli;
guiData.currTempAmpli = currTempAmpli;
guiData.spikeFR       = spikeFR;
guiData.ampliTitle    = ampliTitle;
guiData.ampliLegend   = ampliLegend;
guiData.rpvAmpli      = rpvAmpli;

% amplitude fit
guiData.ampliFitAx     = ampliFitAx;
guiData.ampliBins      = ampliBins;
guiData.ampliFit       = ampliFit;
guiData.ampliFitTitle  = ampliFitTitle;
guiData.ampliFitLegend = ampliFitLegend;

% unit raster
guiData.rasterAx          = rasterAx;
guiData.unitRasterScatter = unitRasterScatter;

guidata(unitQualityGuiHandle, guiData);
end

function updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
    probeLocation, unitType, uniqueTemps, iChunk, plotRaw)

guiData  = guidata(unitQualityGuiHandle);
thisUnit = uniqueTemps(iCluster);
colorsGdBad = [1, 0, 0; 0, 0.5, 0];

%% main title
if unitType(iCluster) == 1
    set(guiData.mainTitle, 'String', ['Unit ', num2str(iCluster), ', single unit'], ...
        'Color', [0, .5, 0]);
elseif unitType(iCluster) == 0
    set(guiData.mainTitle, 'String', ['Unit ', num2str(iCluster), ', noise/non-somatic'], ...
        'Color', [1, 0, 0]);
elseif unitType(iCluster) == 2
    set(guiData.mainTitle, 'String', ['Unit ', num2str(iCluster), ', multi-unit'], ...
        'Color', [0.29, 0, 0.51]);
end

%% 1) ?? location plot ?? unit
set(guiData.currUnitDots, ...
    'XData', guiData.norm_spike_n(thisUnit), ...
    'YData', ephysData.channel_positions(qMetric.maxChannels(thisUnit), 2), ...
    'CData', guiData.unitCmap(iCluster, :));

for iCh = 1:20
    set(guiData.templateWaveformLines(iCh), 'XData', nan(82, 1), 'YData', nan(82, 1))
    set(guiData.rawWaveformLines(iCh),     'XData', nan(82, 1), 'YData', nan(82, 1))
end

%% 2) template waveform + peaks/troughs
maxChan = qMetric.maxChannels(thisUnit);
maxXC   = ephysData.channel_positions(maxChan, 1);
maxYC   = ephysData.channel_positions(maxChan, 2);
chanDistances = sqrt( ...
    (ephysData.channel_positions(:, 1) - maxXC).^2 + ...
    (ephysData.channel_positions(:, 2) - maxYC).^2 );
chansToPlot = find(chanDistances < 100);
vals = [];

for iChanToPlot = 1:min(20, size(chansToPlot, 1))
    vals(iChanToPlot) = max(abs(squeeze(ephysData.templates(thisUnit, :, chansToPlot(iChanToPlot)))));
    xShift = ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10;
    yBase  = (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) ./ 100);

    if maxChan == chansToPlot(iChanToPlot)
        set(guiData.maxTemplateWaveformLines, ...
            'XData', xShift, ...
            'YData', -squeeze(ephysData.templates(thisUnit, :, chansToPlot(iChanToPlot)))' + yBase);

        set(guiData.peaks, ...
            'XData', ephysData.waveform_t(qMetric.peakLocs{iCluster}) + ...
                     (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10, ...
            'YData', -squeeze(ephysData.templates(thisUnit, qMetric.peakLocs{iCluster}, chansToPlot(iChanToPlot)))' + yBase);

        set(guiData.troughs, ...
            'XData', ephysData.waveform_t(qMetric.troughLocs{iCluster}) + ...
                     (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10, ...
            'YData', -squeeze(ephysData.templates(thisUnit, qMetric.troughLocs{iCluster}, chansToPlot(iChanToPlot)))' + yBase);

        set(guiData.templateWaveformLines(iChanToPlot), ...
            'XData', nan(82, 1), 'YData', nan(82, 1));
    else
        set(guiData.templateWaveformLines(iChanToPlot), ...
            'XData', xShift, ...
            'YData', -squeeze(ephysData.templates(thisUnit, :, chansToPlot(iChanToPlot)))' + yBase);
    end
end

tempWvTitleText = ['\fontsize{9}Template waveform: {\color[rgb]{%s}# detected peaks/troughs, ', newline,...
                   '\color[rgb]{%s}is somatic  \color[rgb]{%s}spatial decay}'];

set(guiData.tempTitle, 'String', sprintf(tempWvTitleText, ...
    num2str(colorsGdBad(double(qMetric.nPeaks(iCluster) <= param.maxNPeaks || ...
                                qMetric.nTroughs(iCluster) <= param.maxNTroughs) + 1,:)), ...
    num2str(colorsGdBad(double(qMetric.somatic(iCluster) == 1) + 1,:)), ...
    num2str(colorsGdBad(double(qMetric.spatialDecaySlope(iCluster) > param.minSpatialDecaySlope) + 1,:))));

set(guiData.tempLegend, 'String', { ...
    ['is somatic =', num2str(qMetric.somatic(iCluster)), newline], ...
    [num2str(qMetric.nPeaks(iCluster)),   ' peak(s)'], ...
    [num2str(qMetric.nTroughs(iCluster)), ' trough(s)'], ...
    ['spatial decay slope =' , num2str(qMetric.spatialDecaySlope(iCluster))]});

%% 3) mean raw waveform
for iChanToPlot = 1:min(20, size(chansToPlot, 1))
    xShift = ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10;
    yBase  = (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) * 10);

    if maxChan == chansToPlot(iChanToPlot)
        set(guiData.maxRawWaveformLines, ...
            'XData', xShift, ...
            'YData', -squeeze(qMetric.rawWaveforms(iCluster).spkMapMean(chansToPlot(iChanToPlot), :))' + yBase);
        set(guiData.rawWaveformLines(iChanToPlot), ...
            'XData', nan(82, 1), 'YData', nan(82, 1));
    else
        set(guiData.rawWaveformLines(iChanToPlot), ...
            'XData', xShift, ...
            'YData', -squeeze(qMetric.rawWaveforms(iCluster).spkMapMean(chansToPlot(iChanToPlot), :))' + yBase);
    end
end

set(guiData.rawLegend, 'String', ['Amplitude =', num2str(qMetric.rawAmplitude(iCluster)), 'uV'])
if qMetric.rawAmplitude(iCluster) < param.minAmplitude
    set(guiData.rawTitle, 'String', '\color[rgb]{1 0 1}Mean raw waveform');
else
    set(guiData.rawTitle, 'String', '\color[rgb]{0 .5 0}Mean raw waveform');
end

%% 4) ACG
theseSpikeTimes = ephysData.spike_times_timeline(ephysData.spike_templates == thisUnit);

[ccg, ccg_t] = CCGBz( ...
    [double(theseSpikeTimes); double(theseSpikeTimes)], ...
    [ones(size(theseSpikeTimes, 1), 1); ones(size(theseSpikeTimes, 1), 1) * 2], ...
    'binSize', 0.001, 'duration', 0.5, 'norm', 'rate');

set(guiData.acgBar, ...
    'XData', ccg_t(250:501)*1000, ...
    'YData', squeeze(ccg(250:501, 1, 1)));
set(guiData.acgRefLine, ...
    'XData', [2, 2], ...
    'YData', [0, max(ccg(:, 1, 1))])

[ccg2, ~] = CCGBz( ...
    [double(theseSpikeTimes); double(theseSpikeTimes)], ...
    [ones(size(theseSpikeTimes, 1), 1); ones(size(theseSpikeTimes, 1), 1) * 2], ...
    'binSize', 0.1, 'duration', 10, 'norm', 'rate');

asymptoteLine = nanmean(ccg2(end-100:end));
set(guiData.acgAsyLine, 'XData', [0, 250], 'YData', [asymptoteLine, asymptoteLine])

if qMetric.Fp(iCluster) > param.maxRPVviolations
    set(guiData.acgTitle, 'String', '\color[rgb]{1 0 1}ACG');
else
    set(guiData.acgTitle, 'String', '\color[rgb]{0 .5 0}ACG');
end

%% 5) ISI
theseISI      = diff(theseSpikeTimes);
theseISIclean = theseISI(theseISI >= param.tauC); % removed duplicate spikes
theseOffendingSpikes = find(theseISIclean < (2/1000)); 

[isiProba, edgesISI] = histcounts(theseISIclean*1000, 0:0.5:50);

set(guiData.isiBar, ...
    'XData', edgesISI(1:end-1)+mean(diff(edgesISI)), ...
    'YData', isiProba);
set(guiData.isiRefLine, ...
    'XData', [2, 2], ...
    'YData', [0, max(isiProba)])

if qMetric.Fp(iCluster) > param.maxRPVviolations
    set(guiData.isiTitle, 'String', '\color[rgb]{1 0 1}ISI');
else
    set(guiData.isiTitle, 'String', '\color[rgb]{0 .5 0}ISI');
end
set(guiData.isiLegend, 'String', [num2str(qMetric.Fp(iCluster)), ' % r.p.v.'])

%% 6) isolation distance
if param.computeDistanceMetrics
    set(guiData.currIsoD, ...
        'XData', qMetric.Xplot{iCluster}(:, 1), ...
        'YData', qMetric.Xplot{iCluster}(:, 2));
    set(guiData.rpvIsoD, ...
        'XData', qMetric.Xplot{iCluster}(theseOffendingSpikes, 1), ...
        'YData', qMetric.Xplot{iCluster}(theseOffendingSpikes, 2));
    set(guiData.otherIsoD, ...
        'XData', qMetric.Yplot{iCluster}(:, 1), ...
        'YData', qMetric.Yplot{iCluster}(:, 2), ...
        'CData', qMetric.d2_mahal{iCluster});
end

%% 7) amplitude fit
set(guiData.ampliBins, ...
    'XData', qMetric.ampliBinCenters{iCluster}, ...
    'YData', qMetric.ampliBinCounts{iCluster});

set(guiData.ampliFit, ...
    'XData', qMetric.ampliFit{iCluster}, ...
    'YData', qMetric.ampliBinCenters{iCluster});

if qMetric.percSpikesMissing(iCluster) > param.maxPercSpikesMissing
    set(guiData.ampliFitTitle, 'String', '\color[rgb]{1 0 1}% spikes missing');
else
    set(guiData.ampliFitTitle, 'String', '\color[rgb]{0 .5 0}% spikes missing');
end
set(guiData.ampliFitLegend, 'String', { ...
    [num2str(qMetric.percSpikesMissing(iCluster)), ' % spikes missing'], ...
    'rpv spikes'});
set(guiData.ampliFitAx, 'YLim', ...
    [min(qMetric.ampliBinCenters{iCluster}), max(qMetric.ampliBinCenters{iCluster})]);

%% 8) amplitude & firing rate over recording
% === ??????? tempAmpli ??????????? amplitude panel ===
if ~isfield(guiData,'tempAmpli')        || ~isgraphics(guiData.tempAmpli) || ...
   ~isfield(guiData,'currTempAmpli')    || ~isgraphics(guiData.currTempAmpli) || ...
   ~isfield(guiData,'rpvAmpli')         || ~isgraphics(guiData.rpvAmpli) || ...
   ~isfield(guiData,'spikeFR')          || ~isgraphics(guiData.spikeFR) || ...
   ~isfield(guiData,'ampliAx')          || ~isgraphics(guiData.ampliAx)

    ampliAx = subplot(6,13,[67:70, 73:76]);
    hold(ampliAx,'on');
    yyaxis(ampliAx,'left');
    guiData.tempAmpli = scatter(ampliAx,NaN,NaN,'black','filled','ButtonDownFcn',@ampli_click);
    guiData.currTempAmpli = scatter(ampliAx,NaN,NaN,'blue','filled','ButtonDownFcn',@ampli_click);
    guiData.rpvAmpli = scatter(ampliAx,NaN,NaN,10,'magenta','filled','ButtonDownFcn',@ampli_click);
    xlabel(ampliAx,'Experiment time (s)');
    ylabel(ampliAx,'Template amplitude scaling');
    set(ampliAx,'YColor','k');

    yyaxis(ampliAx,'right');
    guiData.spikeFR = stairs(ampliAx,NaN,NaN,'LineWidth',2.0,'Color',[1 0.5 0]);
    ylabel(ampliAx,'Firing rate (sp/sec)');

    guiData.ampliTitle  = title(ampliAx,'');
    guiData.ampliLegend = legend(ampliAx,[guiData.tempAmpli, guiData.rpvAmpli],{'',''});
    guiData.ampliAx     = ampliAx;

    guidata(unitQualityGuiHandle, guiData);
end

% ????????? guidata ???
guiData = guidata(unitQualityGuiHandle);

theseAmplis = ephysData.template_amplitudes(ephysData.spike_templates == thisUnit);

set(guiData.tempAmpli, 'XData', theseSpikeTimes, 'YData', theseAmplis)
set(guiData.rpvAmpli,  'XData', theseSpikeTimes(theseOffendingSpikes), ...
                       'YData', theseAmplis(theseOffendingSpikes))

currTimes  = theseSpikeTimes(theseSpikeTimes >= theseSpikeTimes(iChunk)-0.1 & ...
                             theseSpikeTimes <= theseSpikeTimes(iChunk)+0.1);
currAmplis = theseAmplis(theseSpikeTimes >= theseSpikeTimes(iChunk)-0.1 & ...
                         theseSpikeTimes <= theseSpikeTimes(iChunk)+0.1);
set(guiData.currTempAmpli, 'XData', currTimes, 'YData', currAmplis);
set(guiData.ampliAx.YAxis(1), 'Limits', [0, round(max(theseAmplis))])

binSize  = 20;
timeBins = 0:binSize:ceil(ephysData.spike_times(end)/ephysData.ephys_sample_rate);
while length(timeBins)==1    
    binSize  = binSize/2;
    timeBins = 0:binSize:ceil(ephysData.spike_times(end)/ephysData.ephys_sample_rate);
end
[n, x] = hist(theseSpikeTimes, timeBins);
n = n ./ binSize;

set(guiData.spikeFR, 'XData', x, 'YData', n);
set(guiData.ampliAx.YAxis(2), 'Limits', [0, 2 * ceil(max(n))])

if qMetric.nSpikes(iCluster) > param.minNumSpikes
    set(guiData.ampliTitle, 'String', '\color[rgb]{0 .5 0}Spikes');
else
    set(guiData.ampliTitle, 'String', '\color[rgb]{1 0 1}Spikes');
end
set(guiData.ampliLegend, 'String', { ...
    ['# spikes = ', num2str(qMetric.nSpikes(iCluster))], 'rpv spikes'})

%% 9) raw snippet
if plotRaw
    plotSubRaw(guiData.rawPlotH, guiData.rawPlotLines, guiData.rawSpikeLines, ...
        memMapData, ephysData, iCluster, uniqueTemps, iChunk);
end

%% 10) Unit raster —— ? theseSpikeTimes ????
if isempty(theseSpikeTimes)
    set(guiData.unitRasterScatter,'XData',NaN,'YData',NaN);
    set(guiData.rasterAx,'XLim',[0 1],'YLim',[0.5 1.5]);
else
    set(guiData.unitRasterScatter,'XData',theseSpikeTimes,...
                                  'YData',ones(size(theseSpikeTimes)));
    set(guiData.rasterAx,'XLim',[min(theseSpikeTimes) max(theseSpikeTimes)],...
                         'YLim',[0.5 1.5]);
end
set(guiData.rasterAx,'YTick',1,'YTickLabel',{'Trials'});
title(guiData.rasterAx,'Unit raster');

% ??? unit / chunk ?? guidata????????
guiData.iCluster = iCluster;
guiData.iChunk   = iChunk;
guidata(unitQualityGuiHandle, guiData);
end

function updateRawSnippet(unitQualityGuiHandle, memMapData, ephysData, iCluster, iCount, qMetric, param, ...
    probeLocation, uniqueTemps, iChunk, plotRaw)

if plotRaw
    guiData  = guidata(unitQualityGuiHandle);
    thisUnit = uniqueTemps(iCluster);

    % === ????? amplitude ?????????????????? ===
    if ~isfield(guiData,'tempAmpli')        || ~isgraphics(guiData.tempAmpli) || ...
       ~isfield(guiData,'currTempAmpli')    || ~isgraphics(guiData.currTempAmpli) || ...
       ~isfield(guiData,'rpvAmpli')         || ~isgraphics(guiData.rpvAmpli) || ...
       ~isfield(guiData,'spikeFR')          || ~isgraphics(guiData.spikeFR) || ...
       ~isfield(guiData,'ampliAx')          || ~isgraphics(guiData.ampliAx)

        ampliAx = subplot(6,13,[67:70, 73:76]);
        hold(ampliAx,'on');
        yyaxis(ampliAx,'left');
        guiData.tempAmpli = scatter(ampliAx,NaN,NaN,'black','filled','ButtonDownFcn',@ampli_click);
        guiData.currTempAmpli = scatter(ampliAx,NaN,NaN,'blue','filled','ButtonDownFcn',@ampli_click);
        guiData.rpvAmpli = scatter(ampliAx,NaN,NaN,10,'magenta','filled','ButtonDownFcn',@ampli_click);
        xlabel(ampliAx,'Experiment time (s)');
        ylabel(ampliAx,'Template amplitude scaling');
        set(ampliAx,'YColor','k');

        yyaxis(ampliAx,'right');
        guiData.spikeFR = stairs(ampliAx,NaN,NaN,'LineWidth',2.0,'Color',[1 0.5 0]);
        ylabel(ampliAx,'Firing rate (sp/sec)');

        guiData.ampliTitle  = title(ampliAx,'');
        guiData.ampliLegend = legend(ampliAx,[guiData.tempAmpli, guiData.rpvAmpli],{'',''});
        guiData.ampliAx     = ampliAx;

        guidata(unitQualityGuiHandle, guiData);
    end

    guiData  = guidata(unitQualityGuiHandle);

    %% raw snippet
    plotSubRaw(guiData.rawPlotH, guiData.rawPlotLines, guiData.rawSpikeLines, ...
        memMapData, ephysData, iCluster, uniqueTemps, iChunk);

    %% amplitudes & FR around current chunk
    theseSpikeTimes = ephysData.spike_times_timeline(ephysData.spike_templates == thisUnit);
    theseAmplis     = ephysData.template_amplitudes(ephysData.spike_templates == thisUnit);

    set(guiData.tempAmpli, 'XData', theseSpikeTimes, 'YData', theseAmplis)

    currTimes  = theseSpikeTimes(theseSpikeTimes >= theseSpikeTimes(iChunk)-0.1 & ...
                                 theseSpikeTimes <= theseSpikeTimes(iChunk)+0.1);
    currAmplis = theseAmplis(theseSpikeTimes >= theseSpikeTimes(iChunk)-0.1 & ...
                             theseSpikeTimes <= theseSpikeTimes(iChunk)+0.1);
    set(guiData.currTempAmpli, 'XData', currTimes, 'YData', currAmplis);
    set(guiData.ampliAx.YAxis(1), 'Limits', [0, round(max(theseAmplis))])

    binSize  = 20;
    timeBins = 0:binSize:ceil(ephysData.spike_times(end)/ephysData.ephys_sample_rate);
    [n, x]   = hist(theseSpikeTimes, timeBins);
    n = n ./ binSize;

    set(guiData.spikeFR, 'XData', x, 'YData', n);
    set(guiData.ampliAx.YAxis(2), 'Limits', [0, 2 * round(max(n))])

    if qMetric.nSpikes(iCluster) > param.minNumSpikes
        set(guiData.ampliTitle, 'String', '\color[rgb]{0 .5 0}Spikes');
    else
        set(guiData.ampliTitle, 'String', '\color[rgb]{1 0 0}Spikes');
    end
    set(guiData.ampliLegend, 'String', ['# spikes = ', num2str(qMetric.nSpikes(iCluster))])

    % ?? chunk
    guiData.iChunk = iChunk;
    guidata(unitQualityGuiHandle, guiData);
end
end

function plotSubRaw(rawPlotH, rawPlotLines, rawSpikeLines, memMapData, ephysData, iCluster, uniqueTemps, iChunk)

chanAmps = squeeze(max(ephysData.templates(iCluster, :, :)) - ...
                   min(ephysData.templates(iCluster, :, :)));
[~, maxChan] = max(chanAmps);
maxXC = ephysData.channel_positions(maxChan, 1);
maxYC = ephysData.channel_positions(maxChan, 2);
chanDistances = sqrt( ...
    (ephysData.channel_positions(:, 1) - maxXC).^2 + ...
    (ephysData.channel_positions(:, 2) - maxYC).^2 );
chansToPlot = find(chanDistances < 100);

timeToPlot  = 0.1;
pull_spikeT = -40:41;
thisC       = uniqueTemps(iCluster);
fs          = ephysData.ephys_sample_rate;

theseTimesCenter = ephysData.spike_times(ephysData.spike_templates == thisC) ./ fs;

if iChunk < 0
    iChunk = 1;
end
if length(theseTimesCenter) > 10 + iChunk
    firstSpike = theseTimesCenter(iChunk+10) - 0.05;
else
    firstSpike = theseTimesCenter(iChunk)   - 0.05;
end

theseTimesCenter = theseTimesCenter(theseTimesCenter >= firstSpike & ...
                                    theseTimesCenter <= firstSpike + timeToPlot);
if ~isempty(theseTimesCenter)
    theseTimesFull = theseTimesCenter * fs + pull_spikeT;
end

cCount = cumsum(repmat(1000, size(chansToPlot, 1), 1), 1);

tSamples = int32(firstSpike * fs): ...
           int32((firstSpike + timeToPlot) * fs);

subplot(rawPlotH)
plotidx = tSamples;

valid = ~(plotidx<1 | plotidx>size(memMapData,2));
plotidx = plotidx(valid);
tSamples = tSamples(valid);

thisMemMap = double(memMapData(chansToPlot,plotidx)) + double(cCount);

for iClear = 1:length(rawSpikeLines)
    set(rawSpikeLines(iClear), 'XData', NaN, 'YData', NaN)
end

if length(rawSpikeLines) < length(chansToPlot)
    rawSpikeLines(end+1:length(chansToPlot)) = rawSpikeLines(end);
    rawPlotLines(end+1:length(chansToPlot))  = rawPlotLines(end);
end

for iChanToPlot = 1:length(chansToPlot)
    set(rawPlotLines(iChanToPlot), 'XData', tSamples, 'YData', thisMemMap(iChanToPlot,:));
    if ~isempty(theseTimesCenter) && exist('theseTimesFull','var')
        for iTimes = 1:size(theseTimesCenter, 1)
            if ~any(mod(theseTimesFull(iTimes, :), 1))
                set(rawSpikeLines(iChanToPlot), ...
                    'XData', theseTimesFull(iTimes, :), ...
                    'YData', thisMemMap(iChanToPlot, int32(theseTimesFull(iTimes, :))-tSamples(1)));
            end
        end
    end
end

% ??? raw snippet ? sample ???? guiData????????
fig = ancestor(rawPlotH, 'figure');
guiData = guidata(fig);
if isfield(guiData, 'rawTimeLine')
    set(guiData.rawTimeLine, 'XData', [NaN NaN], 'YData', [NaN NaN]);
end
guiData.rawSnippetSampleRange = [tSamples(1) tSamples(end)];
guidata(fig, guiData);
end

function ampli_click(src, evnt)
% ?? amplitude–time ??????????????? raw snippet?
% ?? raw waveform ????????????sample??

    fig = ancestor(src, 'figure');
    guiData = guidata(fig);

    if ~isfield(guiData, 'plotRaw') || ~guiData.plotRaw
        return;
    end

    % ?? spike ????XData?????
    spikeTimes = get(guiData.tempAmpli, 'XData');
    if isempty(spikeTimes) || all(isnan(spikeTimes))
        return;
    end

    cp = evnt.IntersectionPoint;
    clickT = cp(1);   % ??????s?

    [~, idx] = min(abs(spikeTimes - clickT));
    if isempty(idx) || isnan(idx)
        return;
    end

    guiData.iChunk = idx;
    guidata(fig, guiData);

    % ????????? raw snippet
    updateRawSnippet(fig, ...
        guiData.memMapData, ...
        guiData.ephysData, ...
        guiData.iCluster, ...
        1, ... % iCount??????
        guiData.qMetric, ...
        guiData.param, ...
        guiData.probeLocation, ...
        guiData.uniqueTemps, ...
        guiData.iChunk, ...
        guiData.plotRaw);

    % ?? raw plot ???????????
    guiData = guidata(fig); % ??????updateRawSnippet ??????
    if ~isfield(guiData, 'rawSnippetSampleRange') || ~isfield(guiData, 'rawTimeLine')
        return;
    end

    fs = guiData.ephysData.ephys_sample_rate;
    sample = round(clickT * fs);  % ??? sample index

    r = guiData.rawSnippetSampleRange;
    if sample >= r(1) && sample <= r(2)
        yl = get(guiData.rawPlotH,'YLim');
        set(guiData.rawTimeLine, 'XData', [sample sample], 'YData', yl);
    else
        % ????????? snippet ??????
        set(guiData.rawTimeLine, 'XData', [NaN NaN], 'YData', [NaN NaN]);
    end
end

function unit_click(src, evnt)
% ??? depth ?????? unit ?????? unit

    fig     = ancestor(src, 'figure');
    guiData = guidata(fig);

    if ~isfield(guiData, 'uniqueTemps')
        return;
    end

    % ??????? axes ???????
    cp = evnt.IntersectionPoint;
    cx = cp(1);
    cy = cp(2);

    % ?? unit ????
    unit_x = get(guiData.unitDots, 'XData');
    unit_y = get(guiData.unitDots, 'YData');

    if isempty(unit_x) || isempty(unit_y)
        return;
    end

    % ??????????? unit
    [~, idx] = min( (unit_x - cx).^2 + (unit_y - cy).^2 );

    % ???? unit index
    guiData.iCluster = idx;
    guidata(fig, guiData);

    % ?? updateUnit ???? panel
    updateUnit(fig, ...
        guiData.memMapData, ...
        guiData.ephysData, ...
        idx, ...
        guiData.qMetric, ...
        guiData.param, ...
        guiData.probeLocation, ...
        guiData.unitType, ...
        guiData.uniqueTemps, ...
        guiData.iChunk, ...
        guiData.plotRaw);
end
