% bc_getQualityUnitType

if param.computeDistanceMetrics && ~isnan(param.isoDmin)
    unitType = nan(length(qMetric.percSpikesMissing), 1);
    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | qMetric.somatic ~= param.somatic ...
        | qMetric.spatialDecaySlope <=  param.minSpatialDecaySlope | qMetric.waveformDuration < param.minWvDuration |...
        qMetric.waveformDuration > param.maxWvDuration | qMetric.waveformBaseline >= param.maxWvBaselineFraction) = 0; % NOISE or NON-SOMATIC
    unitType(any(qMetric.percSpikesMissing <= param.maxPercSpikesMissing, 2)' & qMetric.nSpikes > param.minNumSpikes & ...
        any(qMetric.Fp <= param.maxRPVviolations, 2)' & ...
        qMetric.rawAmplitude > param.minAmplitude & qMetric.isoDmin >= param.isoDmin & isnan(unitType)) = 1; % SINGLE SEXY UNIT
    unitType(isnan(unitType)) = 2; % MULTI UNIT

else
    unitType = nan(length(qMetric.percSpikesMissing), 1);
    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | qMetric.somatic ~= param.somatic ...
        | qMetric.spatialDecaySlope <=  param.minSpatialDecaySlope | qMetric.waveformDuration < param.minWvDuration |...
        qMetric.waveformDuration > param.maxWvDuration  | qMetric.waveformBaseline >= param.maxWvBaselineFraction) = 0; % NOISE or NON-SOMATIC
    unitType(any(qMetric.percSpikesMissing <= param.maxPercSpikesMissing, 2)' & qMetric.nSpikes > param.minNumSpikes & ...
        any(qMetric.Fp <= param.maxRPVviolations, 2)' & ...
        qMetric.rawAmplitude > param.minAmplitude & isnan(unitType)') = 1; % SINGLE SEXY UNIT
    unitType(isnan(unitType)') = 2; % MULTI UNIT

end



%%


% bc_getQualityUnitType

if param.computeDistanceMetrics && ~isnan(param.isoDmin)
    unitType = nan(length(qMetric.percSpikesMissing), 1);

    % ===== 这行是 0 的判定，保持不变 =====
    unitType(qMetric.nPeaks > param.maxNPeaks | ...
             qMetric.nTroughs > param.maxNTroughs | ...
             qMetric.somatic ~= param.somatic ...
        |    qMetric.spatialDecaySlope <=  param.minSpatialDecaySlope | ...
             qMetric.waveformDuration <  param.minWvDuration | ...
             qMetric.waveformDuration >  param.maxWvDuration | ...
             qMetric.waveformBaseline >= param.maxWvBaselineFraction) = 0; % NOISE or NON-SOMATIC

    % ===== 这里是 1 的判定，只放宽阈值，不影响 0 的数量 =====
    ok_missing = any(qMetric.percSpikesMissing <= param.maxPercSpikesMissing * 2, 2)';  % 允许稍多 missing
    ok_rpv     = any(qMetric.Fp               <= param.maxRPVviolations   * 2, 2)';      % 允许稍多 RPV violation

    unitType(ok_missing & ...
             qMetric.nSpikes      > param.minNumSpikes   * 0.5 & ...   % 少一点 spike 也行
             ok_rpv            & ...
             qMetric.rawAmplitude  > param.minAmplitude   * 0.5 & ...   % 振幅低一点也行
             qMetric.isoDmin       >= param.isoDmin       * 1 & ...   % isoD 要求稍微放宽
             isnan(unitType)) = 1;                                      % SINGLE UNIT

    % ===== 剩下既不是 0 又没被标成 1 的，就标成 2 =====
    unitType(isnan(unitType)) = 2; % MULTI UNIT

else
    unitType = nan(length(qMetric.percSpikesMissing), 1);

    % ===== 这行是 0 的判定，保持不变 =====
    unitType(qMetric.nPeaks > param.maxNPeaks | ...
             qMetric.nTroughs > param.maxNTroughs | ...
             qMetric.somatic ~= param.somatic ...
        |    qMetric.spatialDecaySlope <=  param.minSpatialDecaySlope | ...
             qMetric.waveformDuration <  param.minWvDuration | ...
             qMetric.waveformDuration >  param.maxWvDuration  | ...
             qMetric.waveformBaseline >= param.maxWvBaselineFraction) = 0; % NOISE or NON-SOMATIC

    % ===== 这里只放宽 1 的条件 =====
    ok_missing = any(qMetric.percSpikesMissing <= param.maxPercSpikesMissing * 2, 2)';  % 允许稍多 missing
    ok_rpv     = any(qMetric.Fp               <= param.maxRPVviolations   * 2, 2)';      % 允许稍多 RPV violation

    unitType(ok_missing & ...
             qMetric.nSpikes      > param.minNumSpikes   * 0.5 & ...   % 少一点 spike 也行
             ok_rpv            & ...
             qMetric.rawAmplitude  > param.minAmplitude   * 0.5 & ...   % 振幅低一点也行
             isnan(unitType)') = 1;                                    % SINGLE UNIT

    unitType(isnan(unitType)') = 2; % MULTI UNIT
end

%
fprintf('Type0=%d, Type1=%d, Type2=%d\n', ...
    sum(unitType==0), sum(unitType==1), sum(unitType==2));

