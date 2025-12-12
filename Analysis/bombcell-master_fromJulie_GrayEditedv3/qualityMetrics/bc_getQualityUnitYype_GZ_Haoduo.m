% % bc_getQualityUnitType
% 
% if param.computeDistanceMetrics && ~isnan(param.isoDmin)
%     unitType = nan(length(qMetric.percSpikesMissing), 1);
%     unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | qMetric.somatic ~= param.somatic ...
%         | qMetric.spatialDecaySlope <=  param.minSpatialDecaySlope | qMetric.waveformDuration < param.minWvDuration |...
%         qMetric.waveformDuration > param.maxWvDuration | qMetric.waveformBaseline >= param.maxWvBaselineFraction) = 0; % NOISE or NON-SOMATIC
%     unitType(any(qMetric.percSpikesMissing <= param.maxPercSpikesMissing, 2)' & qMetric.nSpikes > param.minNumSpikes & ...
%         any(qMetric.Fp <= param.maxRPVviolations, 2)' & ...
%         qMetric.rawAmplitude > param.minAmplitude & qMetric.isoDmin >= param.isoDmin & isnan(unitType)) = 1; % SINGLE SEXY UNIT
%     unitType(isnan(unitType)) = 2; % MULTI UNIT
% 
% else
%     unitType = nan(length(qMetric.percSpikesMissing), 1);
%     unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | qMetric.somatic ~= param.somatic ...
%         | qMetric.spatialDecaySlope <=  param.minSpatialDecaySlope | qMetric.waveformDuration < param.minWvDuration |...
%         qMetric.waveformDuration > param.maxWvDuration  | qMetric.waveformBaseline >= param.maxWvBaselineFraction) = 0; % NOISE or NON-SOMATIC
%     unitType(any(qMetric.percSpikesMissing <= param.maxPercSpikesMissing, 2)' & qMetric.nSpikes > param.minNumSpikes & ...
%         any(qMetric.Fp <= param.maxRPVviolations, 2)' & ...
%         qMetric.rawAmplitude > param.minAmplitude & isnan(unitType)') = 1; % SINGLE SEXY UNIT
%     unitType(isnan(unitType)') = 2; % MULTI UNIT
% 
% end



%% ===== 在调用 bc_getQualityUnitType 之前，先放宽一些阈值 =====
param_relaxed = param;

% 允许更多 missing spikes
param_relaxed.maxPercSpikesMissing   = param.maxPercSpikesMissing * 1;

% 要求的最小 spike 数降低一点
param_relaxed.minNumSpikes           = round(param.minNumSpikes * 1);

% 允许更多 refractory violation
param_relaxed.maxRPVviolations       = param.maxRPVviolations * 2;

% 振幅阈值降低一点（小一点的 unit 也算好）
param_relaxed.minAmplitude           = param.minAmplitude * 1;

% isoDmin 降低一点（聚类分离度要求更宽松）
if param.computeDistanceMetrics && ~isnan(param.isoDmin)
    param_relaxed.isoDmin            = param.isoDmin * 1;
end

% 波形空间扩散/宽度的要求放宽
param_relaxed.minSpatialDecaySlope   = param.minSpatialDecaySlope * 2; %%jia
param_relaxed.minWvDuration          = param.minWvDuration * 1;
param_relaxed.maxWvDuration          = param.maxWvDuration * 2;   %% jia

% baseline fraction 放宽一点（波形基线不那么“完美”也可以）
param_relaxed.maxWvBaselineFraction  = param.maxWvBaselineFraction * 1; %% jia dan meishenmeyong

%% ===== 用放宽后的 param_relaxed 重新计算 unitType =====
param = param_relaxed;   % 如果后面只用这一套，直接覆盖 param 即可

if param.computeDistanceMetrics && ~isnan(param.isoDmin)
    unitType = nan(length(qMetric.percSpikesMissing), 1);
    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | qMetric.somatic ~= param.somatic ...
        | qMetric.spatialDecaySlope <=  param.minSpatialDecaySlope | qMetric.waveformDuration < param.minWvDuration |...
        qMetric.waveformDuration > param.maxWvDuration | qMetric.waveformBaseline >= param.maxWvBaselineFraction) = 0; % NOISE or NON-SOMATIC

    unitType(any(qMetric.percSpikesMissing <= param.maxPercSpikesMissing, 2)' & ...
             qMetric.nSpikes > param.minNumSpikes & ...
             any(qMetric.Fp <= param.maxRPVviolations, 2)' & ...
             qMetric.rawAmplitude > param.minAmplitude & ...
             qMetric.isoDmin >= param.isoDmin & ...
             isnan(unitType)) = 1; % SINGLE UNIT

    unitType(isnan(unitType)) = 2; % MULTI UNIT

else
    unitType = nan(length(qMetric.percSpikesMissing), 1);
    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | qMetric.somatic ~= param.somatic ...
        | qMetric.spatialDecaySlope <=  param.minSpatialDecaySlope | qMetric.waveformDuration < param.minWvDuration |...
        qMetric.waveformDuration > param.maxWvDuration  | qMetric.waveformBaseline >= param.maxWvBaselineFraction) = 0; % NOISE or NON-SOMATIC

    unitType(any(qMetric.percSpikesMissing <= param.maxPercSpikesMissing, 2)' & ...
             qMetric.nSpikes > param.minNumSpikes & ...
             any(qMetric.Fp <= param.maxRPVviolations, 2)' & ...
             qMetric.rawAmplitude > param.minAmplitude & ...
             isnan(unitType)') = 1; % SINGLE UNIT

    unitType(isnan(unitType)') = 2; % MULTI UNIT
end

% ===== 看看现在 0 / 1 / 2 的数量变化 =====
nType0 = sum(unitType == 0);
nType1 = sum(unitType == 1);
nType2 = sum(unitType == 2);
fprintf('Type0: %d, Type1: %d, Type2: %d\n', nType0, nType1, nType2);
