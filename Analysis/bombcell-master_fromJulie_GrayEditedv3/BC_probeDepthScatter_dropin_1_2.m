function [hFig, out] = BC_probeDepthScatter_dropin_1_2(memMapData, ephysData, qMetric, param, probeLocation, unitType, varargin) 
% BC_probeDepthScatter_dropin
% “Units on probe”散点图，用 log(spike数)/µm 显示各 unit 在探针上的分布
% 这是非 GUI 的 drop-in 版本
%
% Inputs:
%   memMapData  : AP memmap [nChan x nSamples] (可为空)
%   ephysData   : 结构体
%                 - spike_templates (nSpikes x 1)
%                 - spike_times      (nSpikes x 1)
%                 - ephys_sample_rate(Hz)
%                 - templates        (nTemplates x nTime x nChan)
%                 - channel_positions(nChan x 2) [x,y] (µm)
%   qMetric     : 质量指标结构体，包含 maxChannels
%   param       : 参数结构体
%   probeLocation: 探针信息（可为空）
%   unitType    : 每个 unique template 的类型标签 (0/1/2)
%
% Name-Value 参数:
%   'DepthThreshold'  : y 方向分界线 (默认空)
%   'Title'           : 标题
%   'ShowLegend'      : 是否显示图例 (true/false)
%
% Outputs:
%   hFig : figure 句柄
%   out  : 输出结构体，包含 depth, color, counts 等信息
%
% Example:
%   [hFig,out] = BC_probeDepthScatter_dropin([], ephysData, qMetric, [], [], unitType, ...
%                                            'DepthThreshold', 2625, 'Title','WT_1347_rec2');

%% ------------ Parse options ------------
p = inputParser;
p.addParameter('DepthThreshold', [], @(x) isempty(x) || (isscalar(x) && isnumeric(x)));
p.addParameter('Title', 'Units on probe', @(x) ischar(x) || isstring(x));
p.addParameter('ShowLegend', true, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});
depth_thr  = p.Results.DepthThreshold;
titleStr   = string(p.Results.Title);
showLegend = p.Results.ShowLegend;

%% ------------ Sanity checks ------------
assert(isfield(ephysData,'spike_templates') && ~isempty(ephysData.spike_templates), 'ephysData.spike_templates missing');
assert(isfield(ephysData,'channel_positions') && size(ephysData.channel_positions,2) == 2, 'channel_positions must be nChan x 2');

spike_templates = double(ephysData.spike_templates(:));

% 确保 template ID 为 1-based
if min(spike_templates) == 0
    spike_templates_1b = spike_templates + 1;
else
    spike_templates_1b = spike_templates;
end

uniqueTemps = unique(spike_templates_1b,'stable');
nU = numel(uniqueTemps);

% 检查 unitType 长度
assert(numel(unitType) >= nU, 'unitType length (%d) < number of unique templates (%d).', numel(unitType), nU);

%% ------------ 每个 template 的 spike 数 (log) ------------
countsPerTemplate = accumarray(spike_templates_1b, 1, [max(spike_templates_1b), 1], @sum, 0);
countsPerTemplate = countsPerTemplate(uniqueTemps);
norm_spike_n = mat2gray(log10(countsPerTemplate + 1));  % nU x 1, 0~1

%% ------------ 每个 unit 的最大通道 (depth) ------------
if isfield(qMetric,'maxChannels') && ~isempty(qMetric.maxChannels)
    maxChAll = qMetric.maxChannels(:);
    assert(max(uniqueTemps) <= numel(maxChAll), 'qMetric.maxChannels length < max template id');
    maxChannels = maxChAll(uniqueTemps);
else
    assert(isfield(ephysData,'templates') && ndims(ephysData.templates) == 3, ...
        'qMetric.maxChannels missing and ephysData.templates unavailable');
    tmpl = ephysData.templates; % nTemplates x nTime x nChan
    nTemplates = size(tmpl,1);
    assert(max(uniqueTemps) <= nTemplates, 'templates size < max template id');

    maxChannels = nan(nU,1);
    for k = 1:nU
        tID = uniqueTemps(k);
        chAmp = squeeze(max(tmpl(tID,:,:),[],2) - min(tmpl(tID,:,:),[],2)); % 1 x nChan
        [~, mx] = max(chAmp,[],2);
        maxChannels(k) = mx;
    end
end

% 转换成 µm
chanPos = ephysData.channel_positions;   % nChan x 2, [x,y]
assert(all(maxChannels>=1) && all(maxChannels<=size(chanPos,1)), 'maxChannels out of bounds');
unit_depths = chanPos(maxChannels, 2);

%% ------------ unitType 颜色 ------------
% 1: 单元(singe)，2: 多单元(multi)，0: 噪声(noise)
unit_colors = zeros(nU,3);
is1 = (unitType(1:nU) == 1);
is2 = (unitType(1:nU) == 2);

% === 新增：统计 type1 / type2 数量 ===
nType1 = sum(is1);
nType2 = sum(is2);

% is0 = (unitType(1:nU) == 0);   % 不再绘制 type0
unit_colors(is1,:) = repmat([0,   0.5, 0], sum(is1), 1);
unit_colors(is2,:) = repmat([0.29,0,0.51], sum(is2), 1);

%% ------------ 绘图 ------------
hFig = figure('Color','w','Name','Units on probe (drop-in)','NumberTitle','off');
ax = axes('Parent',hFig); hold(ax,'on');

%% === MODIFIED: 仅绘制 type1 和 type2 ===
keep = (unitType(1:nU) == 1) | (unitType(1:nU) == 2);

hS = gobjects(2,1);
if any(is1)
    hS(1) = scatter(ax, norm_spike_n(is1), unit_depths(is1), 12, [0,0.5,0], 'filled', 'MarkerEdgeColor','k');
end
if any(is2)
    hS(2) = scatter(ax, norm_spike_n(is2), unit_depths(is2), 12, [0.29,0,0.51], 'filled', 'MarkerEdgeColor','k');
end

% 深度分界线
if ~isempty(depth_thr) && isfinite(depth_thr)
    yline(ax, depth_thr, '--', 'LineWidth', 1.2, 'Color', [0 0 0], ...
        'Label','depth threshold', 'LabelVerticalAlignment','bottom');
end

% 轴与标题
xlabel(ax,'Normalized log rate');
ylabel(ax,'Depth (\mum)');
title(ax, titleStr);
set(ax,'YDir','normal');

% 范围仅根据 type1/2
if any(keep)
    xlim(ax,[-0.05 1.05]);
    ymin = min(unit_depths(keep));
    ymax = max(unit_depths(keep));
    ylim(ax,[ymin-50, ymax+50]);
end
grid(ax,'on');

% 图例
if showLegend
    legLabs = {};
    legHandles = [];
    if any(is1), legHandles(end+1) = hS(1); legLabs{end+1} = 'single (type 1)'; end
    if any(is2), legHandles(end+1) = hS(2); legLabs{end+1} = 'multi (type 2)';  end
    if ~isempty(legHandles)
        legend(ax, legHandles, legLabs, 'Location','best', 'Box','off');
    end
end

% === 新增：打印统计信息 ===
fprintf('Plotted neurons — type1: %d, type2: %d (total: %d)\n', ...
        nType1, nType2, nType1 + nType2);

% 可选：在图上显示数量（不改原图结构）
if any(keep)
    text(ax, 0.02, ymax, sprintf('type1 = %d,  type2 = %d', nType1, nType2), ...
        'Units','data','VerticalAlignment','top','FontSize',9,'Color',[0 0 0]);
end

%% ------------ 输出结构 ------------
out = struct();
out.uniqueTemps         = uniqueTemps;
out.norm_spike_n        = norm_spike_n;
out.unit_depths         = unit_depths;
out.unit_colors         = unit_colors;
out.maxChannels         = maxChannels;
out.countsPerTemplate   = countsPerTemplate;

% === 新增：输出 type1 / type2 数量 ===
out.nType1_plotted      = nType1;
out.nType2_plotted      = nType2;
out.nPlotted_total      = nType1 + nType2;

end
