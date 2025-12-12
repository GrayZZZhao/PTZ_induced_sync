function hFig = BC_probeRasterRawGUI_GZ_v2(memMapData, ephysData, qMetric, param, probeLocation, unitType)
% BC_probeRasterRawGUI_GZ
%
% Neuropixels GUI:
% - ?? unit ? probe ?????scatter?
% - ?? unit ? template waveform
% - ?? unit ? mean raw waveform
% - ??? unit ? raster?x ????
% - ?? raster ?? spike ???? raw unwhitened snippet?x ????
% - ?????? Unit ID ??
%
% ???
%   memMapData : AP memmap ?? [nChan x nSamples]
%   ephysData  : ??????
%       - spike_templates
%       - spike_times        (samples)
%       - ephys_sample_rate  (e.g. 30000)
%       - templates          (nTemplates x nTime x nChan)
%       - channel_positions  (nChan x 2)
%       - waveform_t
%   qMetric, param, probeLocation, unitType: ????/????
%

%% ???
uniqueTemps = unique(ephysData.spike_templates);
nUnits      = numel(uniqueTemps); %#ok<NASGU>

% ????
unitCmap = zeros(length(unitType), 3);
unitCmap(unitType == 1, :, :) = repmat([0, 0.5, 0],       length(find(unitType == 1)), 1);   % single
unitCmap(unitType == 0, :, :) = repmat([1, 0, 0],         length(find(unitType == 0)), 1);   % noise/non-somatic
unitCmap(unitType == 2, :, :) = repmat([0.29, 0, 0.51],   length(find(unitType == 2)), 1);   % multi

% ?? template ? firing rate?log ?????
norm_spike_n = mat2gray(log10(accumarray(ephysData.spike_templates, 1)+1));

%% Figure
hFig = figure('Color','w', ...
              'Name','Probe depth + waveform + raster + raw snippet', ...
              'NumberTitle','off', ...
              'Units','normalized', ...
              'Position',[0.05 0.05 0.9 0.85]);
set(hFig,'KeyPressFcn',@KeyPressCb);

% Unit ID label + edit
uicontrol('Parent',hFig, 'Style','text', ...
    'Units','normalized', 'Position',[0.02 0.95 0.08 0.035], ...
    'String','Unit ID:', 'HorizontalAlignment','left', ...
    'BackgroundColor','w', 'FontSize',10);

editUnit = uicontrol('Parent',hFig, 'Style','edit', ...
    'Units','normalized', 'Position',[0.10 0.95 0.08 0.04], ...
    'String','1', 'BackgroundColor','w', 'FontSize',10);

btnGo = uicontrol('Parent',hFig, 'Style','pushbutton', ...
    'Units','normalized', 'Position',[0.19 0.95 0.06 0.04], ...
    'String','Go', 'FontSize',10, 'Callback',@goButtonCallback);

mainTitle = uicontrol('Parent',hFig, 'Style','text', ...
    'Units','normalized', 'Position',[0.28 0.95 0.7 0.04], ...
    'String','', 'HorizontalAlignment','left', ...
    'BackgroundColor','w', 'FontSize',10, 'FontWeight','bold');

%% Axes (3 x 3 grid)
probeAx   = subplot(3,3,[1 4 7],'Parent',hFig);
hold(probeAx,'on');

templateAx= subplot(3,3,2,'Parent',hFig);
hold(templateAx,'on');
set(templateAx,'YDir','reverse');
xlabel(templateAx,'Position + Time');
ylabel(templateAx,'Depth-based offset');
title(templateAx,'Template waveform (multi-channel)');

rawMeanAx = subplot(3,3,3,'Parent',hFig);
hold(rawMeanAx,'on');
set(rawMeanAx,'YDir','reverse');
xlabel(rawMeanAx,'Position + Time');
ylabel(rawMeanAx,'Depth-based offset');
title(rawMeanAx,'Mean raw waveform (multi-channel)');

rasterAx  = subplot(3,3,[5 6],'Parent',hFig);
hold(rasterAx,'on');

rawAx     = subplot(3,3,[8 9],'Parent',hFig);
hold(rawAx,'on');
title(rawAx,'Raw unwhitened data');
set(rawAx,'XColor','k','YColor','k');
ylabel(rawAx,'Channel (offset)');
xlabel(rawAx,'Time (s)');          % raw snippet ??
grid(rawAx,'on');

%% 1) probe scatter
depth_vals = ephysData.channel_positions(:,2);
probeScatter = scatter(probeAx, ...
    norm_spike_n(uniqueTemps), ...
    ephysData.channel_positions(qMetric.maxChannels(uniqueTemps), 2), ...
    10, unitCmap, 'filled', ...
    'ButtonDownFcn',@probe_click);

currUnitDot = scatter(probeAx,0,0,80,[0 0 0],'o','LineWidth',1.5);

xlabel(probeAx,'Normalized log rate');
ylabel(probeAx,'Depth (\mum)');
title(probeAx,'Units on probe');
set(probeAx,'YDir','normal');
xlim(probeAx,[-0.1 1.1]);
ylim(probeAx,[min(depth_vals)-50, max(depth_vals)+50]);

%% 2) template waveform
max_n_channels_plot = 20;
templateWaveformLines = arrayfun(@(x) plot(templateAx,NaN,NaN,'Color',[0.5 0.5 0.5],'LineWidth',1), ...
                                 1:max_n_channels_plot);
maxTemplateWaveformLine = plot(templateAx,NaN,NaN,'b','LineWidth',1.5);

%% 3) mean raw waveform
rawMeanWaveformLines = arrayfun(@(x) plot(rawMeanAx,NaN,NaN,'Color',[0.5 0.5 0.5],'LineWidth',1), ...
                                1:max_n_channels_plot);
maxRawMeanWaveformLine = plot(rawMeanAx,NaN,NaN,'Color',[0 0 0],'LineWidth',1.5);

%% 4) raster
unitRasterScatter = scatter(rasterAx,NaN,NaN,5,'k','filled', ...
    'ButtonDownFcn',@raster_click);

% 
rasterHighlight   = scatter(rasterAx,NaN,NaN,30,'r','o','LineWidth',1.2);


xlabel(rasterAx,'Time (s)');
ylabel(rasterAx,'Trials');
title(rasterAx,'Unit raster');
set(rasterAx,'YTick',1,'YTickLabel',{'Trials'});
grid(rasterAx,'on');
try
    disableDefaultInteractivity(rasterAx);
catch
end

%% 5) raw snippet
rawPlotLines  = arrayfun(@(x) plot(rawAx,NaN,NaN,'k','LineWidth',1), 1:max_n_channels_plot);
rawSpikeLines = arrayfun(@(x) plot(rawAx,NaN,NaN,'b','LineWidth',1), 1:max_n_channels_plot);
rawTimeLine   = line(rawAx,[NaN NaN],[NaN NaN], ...
    'Color',[1 0.8 0.8], ...   % ??????
    'LineWidth',0.8, ...
    'LineStyle','--', ...      % ??
    'HitTest','off');

% ????????waveform ?????????
uistack(rawTimeLine,'bottom');

%% ? guiData
guiData = struct;
guiData.memMapData   = memMapData;
guiData.ephysData    = ephysData;
guiData.qMetric      = qMetric;
guiData.param        = param;
guiData.probeLocation= probeLocation;
guiData.unitType     = unitType;
guiData.uniqueTemps  = uniqueTemps;
guiData.unitCmap     = unitCmap;
guiData.norm_spike_n = norm_spike_n;

guiData.hFig         = hFig;
guiData.editUnit     = editUnit;
guiData.btnGo        = btnGo;
guiData.mainTitle    = mainTitle;

guiData.probeAx      = probeAx;
guiData.probeScatter = probeScatter;
guiData.currUnitDot  = currUnitDot;

guiData.templateAx   = templateAx;
guiData.templateWaveformLines    = templateWaveformLines;
guiData.maxTemplateWaveformLine  = maxTemplateWaveformLine;

guiData.rawMeanAx    = rawMeanAx;
guiData.rawMeanWaveformLines     = rawMeanWaveformLines;
guiData.maxRawMeanWaveformLine   = maxRawMeanWaveformLine;

guiData.rasterAx          = rasterAx;
guiData.unitRasterScatter = unitRasterScatter;
guiData.rasterHighlight   = rasterHighlight;

guiData.rawAx        = rawAx;
guiData.rawPlotLines = rawPlotLines;
guiData.rawSpikeLines= rawSpikeLines;
guiData.rawTimeLine  = rawTimeLine;

guiData.iCluster     = 1;   % ?? unit index
guiData.iChunk       = 1;   % ?? spike index
guiData.rawSnippetSampleRange = [NaN NaN];

guidata(hFig, guiData);

%% ????
updateUnitPlot(hFig);

%% ================= ???? =================

    function goButtonCallback(~,~)
        gd = guidata(hFig);
        str = get(gd.editUnit,'String');
        val = str2double(str);
        if isnan(val) || val<1 || val>numel(gd.uniqueTemps)
            warndlg('Unit ID ??','Warning');
            return;
        end
        gd.iCluster = round(val);
        gd.iChunk   = 1;
        guidata(hFig,gd);
        updateUnitPlot(hFig);
    end

    function KeyPressCb(~,evnt)
        gd = guidata(hFig);
        switch lower(evnt.Key)
            case 'rightarrow'
                gd.iCluster = min(numel(gd.uniqueTemps), gd.iCluster+1);
                gd.iChunk   = 1;
            case 'leftarrow'
                gd.iCluster = max(1, gd.iCluster-1);
                gd.iChunk   = 1;
            case 'u'
                answer = inputdlg('Go to unit ID (index in uniqueTemps):','Go to unit',1,{num2str(gd.iCluster)});
                if ~isempty(answer)
                    v = str2double(answer{1});
                    if ~isnan(v) && v>=1 && v<=numel(gd.uniqueTemps)
                        gd.iCluster = round(v);
                        gd.iChunk   = 1;
                        set(gd.editUnit,'String',num2str(gd.iCluster));
                    end
                end
        end
        guidata(hFig,gd);
        updateUnitPlot(hFig);
    end

    function probe_click(~,evnt)
        gd = guidata(hFig);
        cp = evnt.IntersectionPoint;
        cx = cp(1);
        cy = cp(2);

        ux = get(gd.probeScatter,'XData');
        uy = get(gd.probeScatter,'YData');

        if isempty(ux) || isempty(uy)
            return;
        end

        [~,idx] = min((ux - cx).^2 + (uy - cy).^2);   % ???? unit
        gd.iCluster = idx;
        gd.iChunk   = 1;
        set(gd.editUnit,'String',num2str(gd.iCluster));
        guidata(hFig,gd);
        updateUnitPlot(hFig);
    end

    function raster_click(~,evnt)
        gd = guidata(hFig);

        spikeTimes = get(gd.unitRasterScatter,'XData');  % ????
        if isempty(spikeTimes) || all(isnan(spikeTimes))
            return;
        end

        cp = evnt.IntersectionPoint;
        clickT = cp(1);   % ???????

        [~,idx] = min(abs(spikeTimes - clickT));
        if isempty(idx) || isnan(idx)
            return;
        end

        gd.iChunk = idx;
        guidata(hFig,gd);

        % ????? spike
        set(gd.rasterHighlight,'XData',spikeTimes(idx),'YData',1);

        % ?? raw snippet
        updateRawSnippet(hFig);
    end

end % ?????

%% =================================================================
%  ????updateUnitPlot / updateRawSnippet / plotSubRaw / getAxisTightY
% ==================================================================

function updateUnitPlot(hFig)
gd = guidata(hFig);
iCluster   = gd.iCluster;
uniqueTemps= gd.uniqueTemps;
thisUnit   = uniqueTemps(iCluster);

ephysData  = gd.ephysData;
qMetric    = gd.qMetric;
unitType   = gd.unitType;
unitCmap   = gd.unitCmap;
norm_spike_n = gd.norm_spike_n;

% 1) ?? + ??
if unitType(iCluster) == 1
    unitTypeStr = 'single unit';
    col = [0 0.5 0];
elseif unitType(iCluster) == 2
    unitTypeStr = 'multi-unit';
    col = [0.29 0 0.51];
else
    unitTypeStr = 'noise/non-somatic';
    col = [1 0 0];
end

set(gd.mainTitle,'String',sprintf('Unit ID = %d (template %d), %s', ...
    iCluster, thisUnit, unitTypeStr), ...
    'ForegroundColor',col);

% 2) probe ??? unit ??
set(gd.currUnitDot, ...
    'XData', norm_spike_n(thisUnit), ...
    'YData', gd.ephysData.channel_positions(qMetric.maxChannels(thisUnit),2), ...
    'MarkerFaceColor', unitCmap(iCluster,:), ...
    'MarkerEdgeColor', [0 0 0]);

% 3) template waveform + mean raw waveform
chanAmps = squeeze(max(ephysData.templates(thisUnit, :, :)) - ...
                   min(ephysData.templates(thisUnit, :, :)));
[~, maxChan] = max(chanAmps);
maxXC = ephysData.channel_positions(maxChan, 1);
maxYC = ephysData.channel_positions(maxChan, 2);
chanDistances = sqrt( ...
    (ephysData.channel_positions(:, 1) - maxXC).^2 + ...
    (ephysData.channel_positions(:, 2) - maxYC).^2 );
chansToPlot = find(chanDistances < 100);
if isempty(chansToPlot)
    chansToPlot = maxChan;
end

max_n_channels_plot = numel(gd.templateWaveformLines);

% ?????
for iCh = 1:max_n_channels_plot
    set(gd.templateWaveformLines(iCh),'XData',NaN,'YData',NaN);
    set(gd.rawMeanWaveformLines(iCh),'XData',NaN,'YData',NaN);
end
set(gd.maxTemplateWaveformLine,'XData',NaN,'YData',NaN);
set(gd.maxRawMeanWaveformLine,'XData',NaN,'YData',NaN);

nPlot = min(max_n_channels_plot, numel(chansToPlot));

for k = 1:nPlot
    ch = chansToPlot(k);
    % template waveform: x = time + ???y = depth-based offset
    xShift = ephysData.waveform_t + (ephysData.channel_positions(ch,1)-11)/10;
    yBaseT = (ephysData.channel_positions(ch,2) ./ 100);
    waveT  = -squeeze(ephysData.templates(thisUnit,:,ch))' + yBaseT;

    if ch == maxChan
        set(gd.maxTemplateWaveformLine,'XData',xShift,'YData',waveT);
    else
        set(gd.templateWaveformLines(k),'XData',xShift,'YData',waveT);
    end

    % mean raw waveform
    if isfield(qMetric,'rawWaveforms') && numel(qMetric.rawWaveforms)>=iCluster ...
            && ~isempty(qMetric.rawWaveforms(iCluster).spkMapMean)
        yBaseR = (ephysData.channel_positions(ch,2) * 10);
        waveR  = -squeeze(qMetric.rawWaveforms(iCluster).spkMapMean(ch,:))' + yBaseR;
        if ch == maxChan
            set(gd.maxRawMeanWaveformLine,'XData',xShift,'YData',waveR);
        else
            set(gd.rawMeanWaveformLines(k),'XData',xShift,'YData',waveR);
        end
    end
end

set(gd.templateAx,'YLim',getAxisTightY(gd.templateAx));
set(gd.rawMeanAx,'YLim',getAxisTightY(gd.rawMeanAx));

% 4) raster?? spike_times (samples) ???
fs_ap = ephysData.ephys_sample_rate;  % Neuropixels AP sampling rate (?? 30000)
spikeSamples = ephysData.spike_times(ephysData.spike_templates == thisUnit);
theseSpikeTimes = double(spikeSamples) ./ fs_ap;   % ?

if isempty(theseSpikeTimes)
    set(gd.unitRasterScatter,'XData',NaN,'YData',NaN);
    set(gd.rasterAx,'XLim',[0 1],'YLim',[0.5 1.5]);
    set(gd.rasterHighlight,'XData',NaN,'YData',NaN);
else
    set(gd.unitRasterScatter,'XData',theseSpikeTimes,'YData',ones(size(theseSpikeTimes)));
    set(gd.rasterAx,'XLim',[min(theseSpikeTimes) max(theseSpikeTimes)],'YLim',[0.5 1.5]);
    set(gd.rasterHighlight,'XData',NaN,'YData',NaN);
end
set(gd.rasterAx,'YTick',1,'YTickLabel',{'Trials'});
title(gd.rasterAx,'Unit raster');

% 5) raw snippet ??? spike
gd.iChunk = 1;
guidata(hFig,gd);
updateRawSnippet(hFig);
end

function yL = getAxisTightY(ax)
ch = get(ax,'Children');
ys = [];
for k = 1:numel(ch)
    if isprop(ch(k),'YData')
        y = get(ch(k),'YData');
        ys = [ys; y(:)];
    end
end
if isempty(ys) || all(isnan(ys))
    yL = [0 1];
else
    yL = [min(ys) max(ys)];
end
end

function updateRawSnippet(hFig)
gd = guidata(hFig);
iCluster   = gd.iCluster;
iChunk     = gd.iChunk;
uniqueTemps= gd.uniqueTemps;
thisUnit   = uniqueTemps(iCluster);

memMapData = gd.memMapData;
ephysData  = gd.ephysData;

% ?? unit ?? spike (samples)
spikeSamp = ephysData.spike_times(ephysData.spike_templates == thisUnit);
if isempty(spikeSamp)
    % ? spikes??? raw plot
    for k = 1:numel(gd.rawPlotLines)
        set(gd.rawPlotLines(k),'XData',NaN,'YData',NaN);
        set(gd.rawSpikeLines(k),'XData',NaN,'YData',NaN);
    end
    set(gd.rawTimeLine,'XData',[NaN NaN],'YData',[NaN NaN]);
    title(gd.rawAx,'Raw unwhitened data (no spikes)');
    return;
end

iChunk = max(1,min(iChunk,numel(spikeSamp)));   % clamp
gd.iChunk = iChunk;
guidata(hFig,gd);

plotSubRaw(gd.rawAx, gd.rawPlotLines, gd.rawSpikeLines, gd.rawTimeLine, ...
    memMapData, ephysData, thisUnit, spikeSamp, iChunk);
end

function plotSubRaw(rawAx, rawPlotLines, rawSpikeLines, rawTimeLine, ...
    memMapData, ephysData, thisUnit, spikeSamp, iChunk)
% ?? unit ?? iChunk ? spike????? raw snippet
% ???memMapData ??? sample index ??????? x ??“?”

fs = ephysData.ephys_sample_rate;   % e.g. 30000 Hz

%% 1) ?? channels —— ? template ??????? ±100 µm
chanAmps = squeeze(max(ephysData.templates(thisUnit, :, :)) - ...
                   min(ephysData.templates(thisUnit, :, :)));
[~, maxChan] = max(chanAmps);
maxXC = ephysData.channel_positions(maxChan, 1);
maxYC = ephysData.channel_positions(maxChan, 2);
chanDistances = sqrt( ...
    (ephysData.channel_positions(:, 1) - maxXC).^2 + ...
    (ephysData.channel_positions(:, 2) - maxYC).^2 );
chansToPlot = find(chanDistances < 100);
if isempty(chansToPlot)
    chansToPlot = maxChan; % ??????????
end

%% 2) ??????
timeToPlot  = 0.1;            % 100 ms ??
pull_spikeT = -40:41;         % spike snippet ?????

theseTimesCenter = double(spikeSamp) ./ fs;  % ?? spike ???? (s)

iChunk = max(1,min(iChunk,numel(theseTimesCenter)));
centerT = theseTimesCenter(iChunk);          % ?? spike ?? (s)

firstSpike = centerT - 0.05;                % ?????? (s)
if firstSpike < 0
    firstSpike = 0;
end

% ????????? spike
theseTimesCenterWin = theseTimesCenter(theseTimesCenter >= firstSpike & ...
                                       theseTimesCenter <= firstSpike + timeToPlot);
if ~isempty(theseTimesCenterWin)
    % ???? sample index???? memMap ??
    theseTimesFull = theseTimesCenterWin * fs + pull_spikeT;  % [nSpike x nWindow] samples
end

% ?? channel ?????
cCount = cumsum(repmat(1000, size(chansToPlot, 1), 1), 1);

% ??????sample index?
tSamples = int32(firstSpike * fs):int32((firstSpike + timeToPlot) * fs);

plotidx = tSamples;
valid   = ~(plotidx<1 | plotidx>size(memMapData,2));
plotidx = plotidx(valid);
tSamples= tSamples(valid);

% ? memMapData ?????? sample index?
thisMemMap = double(memMapData(chansToPlot,plotidx)) + double(cCount);

% === ? sample index ??“?”??? x ? ===
tSec = double(tSamples) ./ fs;

% ???? spikeLines
for iClear = 1:length(rawSpikeLines)
    set(rawSpikeLines(iClear), 'XData', NaN, 'YData', NaN)
end

% ?? plotLines/SpikeLines ????
if length(rawPlotLines) < length(chansToPlot)
    rawPlotLines(end+1:length(chansToPlot))  = rawPlotLines(end);
    rawSpikeLines(end+1:length(chansToPlot)) = rawSpikeLines(end);
end

% ???????? spike????????????
if exist('theseTimesFull','var') && ~isempty(theseTimesCenterWin)
    theseTimesFullSec = theseTimesFull ./ fs;  % spike snippet ??? (?)
end

% 3) ??? channel ? raw trace ? spike snippet?x ????
for iChanToPlot = 1:length(chansToPlot)
    % raw trace?XData ? tSec
    set(rawPlotLines(iChanToPlot), 'XData', tSec, 'YData', thisMemMap(iChanToPlot,:));

    % spike snippet
    if exist('theseTimesFullSec','var')
        for iTimes = 1:size(theseTimesFullSec, 1)
            idxSamp = int32(theseTimesFull(iTimes,:)) - tSamples(1) + 1;
            validIdx = idxSamp>=1 & idxSamp<=size(thisMemMap,2);
            if any(validIdx)
                set(rawSpikeLines(iChanToPlot), ...
                    'XData', theseTimesFullSec(iTimes,validIdx), ...   % ?
                    'YData', thisMemMap(iChanToPlot, idxSamp(validIdx)));
            end
        end
    end
end

% 4) ??????? spike????? (?) —— ?????????? waveforms
centerSample = round(centerT * fs);  % ?? spike ? sample index
if centerSample>=tSamples(1) && centerSample<=tSamples(end)
    centerTimeSec = centerSample / fs;
    yl = [min(thisMemMap(:)) max(thisMemMap(:))];
    set(rawTimeLine,'XData',[centerTimeSec centerTimeSec],'YData',yl);
else
    set(rawTimeLine,'XData',[NaN NaN],'YData',[NaN NaN]);
end

title(rawAx,sprintf('Raw unwhitened data (unit template %d, spike #%d)', ...
    thisUnit, iChunk));
xlabel(rawAx,'Time (s)');
ylabel(rawAx,'Channel (offset)');

end
