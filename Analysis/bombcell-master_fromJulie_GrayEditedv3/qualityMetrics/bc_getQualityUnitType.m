% bc_getQualityUnitType
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

% %% GZ 2 to 0 
%% ========= 1. 原始分类逻辑：�?�?�?�?� =========
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


% ========= 2. �?�把一部分“差的 type2�?改�? 0（noise） =========
% �?�? unitType 长度�?�?�

nUnits = numel(unitType);            % 比如 725
isType2 = (unitType == 2);           % Nx1 logical

% ----- �?�调阈值：这些�?��?决定有多少 type2 会 → 0 -----
thr_nSpikes2noise      = 0.5* param.minNumSpikes;        % spike 数下�?
thr_percMissing2noise  = param.maxPercSpikesMissing; % spike missing 上�?
thr_Fp2noise           = param.maxRPVviolations;     % RPV 上�?
thr_amp2noise          = 0.13*param.minAmplitude;         % 振幅下�?

% ----- 1) nSpikes 与 amplitude：它们肯定是一维，与 unitType 对�? -----
nSpikes   = qMetric.nSpikes(:);          % 强制�?��? nUnits×1
rawAmp    = qMetric.rawAmplitude(:);     % 强制�?��? nUnits×1

if numel(nSpikes) ~= nUnits
    error('qMetric.nSpikes 的长度 (%d) 与 unitType (%d) �?一致�?', numel(nSpikes), nUnits);
end
if numel(rawAmp) ~= nUnits
    error('qMetric.rawAmplitude 的长度 (%d) 与 unitType (%d) �?一致�?', numel(rawAmp), nUnits);
end

isFewSpikes = nSpikes < thr_nSpikes2noise;
isSmallAmp  = rawAmp  < thr_amp2noise;

% ----- 2) percSpikesMissing：�?�能是 N×C，也�?�能是一维 -----
pm = qMetric.percSpikesMissing;
if isvector(pm)
    pm_best = pm(:);                               % 一维：直接拉�?列
else
    % 多列：�?�设�?行对应一个 unit，从中�?�“最好�?的 channel（missing 最�?）
    pm_best = min(pm, [], 2);                      % N×1
end

if numel(pm_best) ~= nUnits
    % 维度�?匹�?就直接忽略这个�?�件，�?��?撑大数组
    warning('percSpikesMissing 维度与 unitType �?匹�?，忽略该�?�件。');
    isHighMissing = false(nUnits,1);
else
    pm_best = pm_best(:);
    isHighMissing = pm_best > thr_percMissing2noise;
end

% ----- 3) Fp：�?��?�，�?�能是 N×C，也�?�能是一维 -----
Fp = qMetric.Fp;
if isvector(Fp)
    Fp_best = Fp(:);
else
    % 多列：�?�最好的 channel（Fp 最�?），差的就会大于阈值
    Fp_best = min(Fp, [], 2);
end

if numel(Fp_best) ~= nUnits
    warning('Fp 维度与 unitType �?匹�?，忽略该�?�件。');
    isHighFp = false(nUnits,1);
else
    Fp_best = Fp_best(:);
    isHighFp = Fp_best > thr_Fp2noise;
end

% ----- 综�?��?�件：�?�在 type2 里把“差�?的改�? 0 -----
badType2 = isType2 & (isFewSpikes | isHighMissing | isHighFp | isSmallAmp);  % 全部都是 nUnits×1

unitType(badType2) = 0;

% sanity check
if numel(unitType) ~= nUnits
    error('unitType 大�?被�?外改�?��?现在是 %d, 原本是 %d', numel(unitType), nUnits);
end

%%

fprintf('Type2 总数: %d\n', sum(isType2));
fprintf('  振幅太�?: %d\n', sum(isType2 & isSmallAmp));
fprintf('  spike太少: %d\n', sum(isType2 & isFewSpikes));
fprintf('  missing高: %d\n', sum(isType2 & isHighMissing));
fprintf('  RPV高: %d\n', sum(isType2 & isHighFp));


%% ========= Drop-in: �?�视化 unitType 分布�?�质�?指标 =========
% �?求: 已有�?��? unitType, qMetric (�?�字段 rawAmplitude, nSpikes, Fp, percSpikesMissing)

figure('Name','UnitType 分类检查','Color','w','Position',[200 100 1000 500]);
tiledlayout(2,3,'Padding','compact','TileSpacing','tight');

% ---------- Panel 1: 分类柱状图 ----------
nexttile(1);
counts = [sum(unitType==0), sum(unitType==1), sum(unitType==2)];
bar([0 1 2], counts, 'FaceColor',[0.2 0.5 0.8]);
set(gca,'XTick',[0 1 2],'XTickLabel',{'0','1','2'});
xlabel('unitType');
ylabel('数�?');
title('UnitType 分类检查: 0=Noise, 1=Single, 2=Multi');
text(0, counts(1)+5, sprintf('n=%d',counts(1)),'HorizontalAlignment','center');
text(1, counts(2)+5, sprintf('n=%d',counts(2)),'HorizontalAlignment','center');
text(2, counts(3)+5, sprintf('n=%d',counts(3)),'HorizontalAlignment','center');
grid on;

% ---------- Panel 2: 振幅分布 ----------
nexttile(2); hold on;
for t = 0:2
    vals = qMetric.rawAmplitude(unitType==t);
    if ~isempty(vals)
        histogram(vals, 'Normalization','probability', 'DisplayStyle','stairs','LineWidth',1.5);
    end
end
xlabel('rawAmplitude');
ylabel('Probability');
title('振幅分布（按 unitType）');
legend({'type 0','type 1','type 2'},'Box','off');

% ---------- Panel 3: spike 数分布 ----------
nexttile(3); hold on;
for t = 0:2
    vals = qMetric.nSpikes(unitType==t);
    if ~isempty(vals)
        histogram(vals, 'Normalization','probability', 'DisplayStyle','stairs','LineWidth',1.5);
    end
end
xlabel('nSpikes');
ylabel('Probability');
title('spike 数分布（按 unitType）');
legend({'type 0','type 1','type 2'},'Box','off');

% ---------- Panel 4: RPV violations ----------
nexttile(4); hold on;
for t = 0:2
    Fpvals = qMetric.Fp(unitType==t,:);
    if ~isempty(Fpvals)
        bestFp = min(Fpvals,[],2); % �?��?个unit最好的通�?�
        histogram(bestFp, 'Normalization','probability', 'DisplayStyle','stairs','LineWidth',1.5);
    end
end
xlabel('Fp (best)');
ylabel('Probability');
title('RPV violations（按 unitType）');
legend({'type 0','type 1','type 2'},'Box','off');

% ---------- Panel 5: Spike missing ----------
nexttile(5); hold on;
for t = 0:2
    missvals = qMetric.percSpikesMissing(unitType==t,:);
    if ~isempty(missvals)
        bestMiss = min(missvals,[],2); % �?�missing最好的通�?�
        histogram(bestMiss, 'Normalization','probability', 'DisplayStyle','stairs','LineWidth',1.5);
    end
end
xlabel('percSpikesMissing (best)');
ylabel('Probability');
title('Spike missing（按 unitType）');
legend({'type 0','type 1','type 2'},'Box','off');

% ---------- Panel 6: �?��?（无效数�?��??示） ----------
nexttile(6);
axis off;
text(0.1,0.5,'无有效数�?�','FontSize',12,'Color',[0.5 0.5 0.5]);

sgtitle('UnitType 分类检查: 0=Noise, 1=Single, 2=Multi','FontWeight','bold');
