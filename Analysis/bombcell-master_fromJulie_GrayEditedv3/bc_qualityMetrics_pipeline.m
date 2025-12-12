%% load data 
ephysPath = 'D:\npxl_kv11\2025-02-10_1754_HOM_kv11-female-adult\2025-02-10_18-42-06\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP';%pathToFolderYourEphysDataIsIn; % eg /home/netshare/zinu/JF067/2022-02-17/ephys/kilosort2/site1, whre this path contains 
                                           % kilosort output

% ephysPath = AP_cortexlab_filenameJF(animal,day,experiment,'ephys',site);
[spikeTimes, spikeTemplates, ...
    templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions] = bc_loadEphysData(ephysPath);
ephysap_path = 'D:\npxl_kv11\2025-02-10_1754_HOM_kv11-female-adult\2025-02-10_18-42-06\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP\continuous.dat';%pathToEphysRawFile; %eg /home/netshare/zinu/JF067/2022-02-17/ephys/site1/2022_02_17-JF067_g0_t0.imec0.ap.bin 
ephysDirPath = 'D:\npxl_kv11\2025-02-10_1754_HOM_kv11-female-adult\2025-02-10_18-42-06\Record Node 101\experiment1\recording2\continuous\Neuropix-PXI-100.ProbeA-AP';%pathToEphysRawFileFolder ;% eg /home/netshare/zinu/JF067/2022-02-17/ephys/site1
% ephysap_path =
% AP_cortexlab_filenameJF(animal,day,experiment,'ephys_ap',site);c
% ephysDirPath = AP_cortexlab_filenameJF(animal,day,experiment,'ephys_dir',site);
savePath = fullfile(ephysDirPath, 'qMetrics'); 

%% quality metric parameters and thresholds 
bc_qualityParamValues; %param.plotGlobal = 0;

%% compute quality metrics 
rerun = 0;
qMetricsExist = dir(fullfile(savePath, 'qMetric*.mat'));

if isempty(qMetricsExist) || rerun
    [qMetric, unitType] = bc_runAllQualityMetrics(param, spikeTimes, spikeTemplates, ...
        templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, savePath);
else
    load(fullfile(savePath, 'qMetric.mat'))
    load(fullfile(savePath, 'param.mat'))
    bc_getQualityUnitType;
    %bc_getQualityUnitType_GZ_haoduo;
end

%% save waveform
% 假设 ephysDirPath 已经在 workspace 里，是当前这只老鼠的 ephys 路径

% 1) 在 ephysDirPath 下面创建 “waveform” 文件夹
waveDir = fullfile(ephysDirPath, 'waveform');
if ~exist(waveDir, 'dir')
    mkdir(waveDir);
end

% 2) 取出刚才画的 figure(3)
hFig = figure(4);  % 如果 figure 3 已经存在，这行会把它调出来并设为当前

% 3) 定义保存文件的基础名字
baseName = fullfile(waveDir, 'figure3_waveform');

% 4) 保存为 jpg（位图）
print(hFig, [baseName '.jpg'], '-djpeg', '-r300');   % 300 dpi

% 5) 保存为矢量 pdf
set(hFig, 'PaperPositionMode', 'auto');              % 防止尺寸奇怪
print(hFig, [baseName '.pdf'], '-dpdf', '-painters');


%% view units + quality metrics in GUI 
%get memmap
bc_getRawMemMap;

% put ephys data into structure 
ephysData = struct;
ephysData.spike_times = spikeTimes;
ephysData.spike_times_timeline = spikeTimes ./ 30000;
ephysData.spike_templates = spikeTemplates;
ephysData.templates = templateWaveforms;
ephysData.template_amplitudes = templateAmplitudes;
ephysData.channel_positions = channelPositions;
ephysData.ephys_sample_rate = 30000;
ephysData.waveform_t = 1e3*((0:size(templateWaveforms, 2) - 1) / 30000);
ephysParams = struct;
plotRaw = 1;
probeLocation=[];

% GUI guide: 
% left/right arrow: toggle between units 
% g : go to next good unit 
% m : go to next multi-unit 
% n : go to next noise unit 
% up/down arrow: toggle between time chunks in the raw data
% u: brings up a input dialog to enter the unit you want to go to 
%%
%bc_unitQualityGUI(memMapData,ephysData,qMetric, param, probeLocation, unitType, plotRaw);

%%
%BC_unitQualityGUI_GZ_v2(memMapData,ephysData,qMetric, param, probeLocation, unitType, plotRaw);
%%
%BC_probeRasterRawGUI_GZ_v2(memMapData, ephysData, qMetric, param, probeLocation, unitType);
%%
% ???????????
% BC_probeDepthScatter_dropin(memMapData, ephysData, qMetric, param, probeLocation, unitType, varargin);
%%
% ????
if exist('varargin','var')
    BC_probeDepthScatter_dropin_1_2(memMapData, ephysData, qMetric, param, probeLocation, unitType, varargin{:});
else
    BC_probeDepthScatter_dropin_1_2(memMapData, ephysData, qMetric, param, probeLocation, unitType);
end
