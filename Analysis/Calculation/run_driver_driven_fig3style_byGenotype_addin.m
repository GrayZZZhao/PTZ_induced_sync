%% ===================== DROP-IN: CTX one plot, STR one plot (4 genotypes together) =====================
% Requires: res from run_driver_driven_fig3style_byGenotype()

if ~exist('res','var')
    error('Variable "res" not found. Run run_driver_driven_fig3style_byGenotype first.');
end

% --- read meta safely ---
if isfield(res,'meta')
    meta = res.meta;
else
    meta = struct();
end

% genotype order (prefer meta.genoList, else use fixed 4)
if isfield(meta,'genoList') && ~isempty(meta.genoList)
    genoList = string(meta.genoList);
else
    genoList = ["WT","HOM","HOMcon","HOMkv11"];
end

% modules & bins
if isfield(meta,'modNames') && ~isempty(meta.modNames)
    modNames = string(meta.modNames);
else
    modNames = ["Driver","Driven"];
end
if isfield(meta,'binNames') && ~isempty(meta.binNames)
    binNames = string(meta.binNames);
else
    binNames = ["CTX","STR"];
end

% indices for CTX/STR in meanFrac(:,bin)
iCTX = find(binNames=="CTX",1);
iSTR = find(binNames=="STR",1);
if isempty(iCTX) || isempty(iSTR)
    error('Cannot find CTX/STR in res.meta.binNames.');
end

% --- collect mean/sem for each genotype ---
nG = numel(genoList);
nM = numel(modNames);

MeanCTX = nan(nG,nM);
SemCTX  = nan(nG,nM);
MeanSTR = nan(nG,nM);
SemSTR  = nan(nG,nM);
miceN   = nan(nG,1);
nMatch  = nan(nG,1);

for i = 1:nG
    geno  = genoList(i);
    field = matlab.lang.makeValidName(geno);

    if ~isfield(res, field), continue; end
    out = res.(field);

    if isfield(out,'meanFrac') && ~isempty(out.meanFrac)
        % out.meanFrac: (mod x bin)
        MeanCTX(i,:) = out.meanFrac(:,iCTX).';
        SemCTX(i,:)  = out.semFrac(:,iCTX).';
        MeanSTR(i,:) = out.meanFrac(:,iSTR).';
        SemSTR(i,:)  = out.semFrac(:,iSTR).';
    end

    if isfield(out,'nMice'),          miceN(i)  = out.nMice; end
    if isfield(out,'nPerBinMatched'), nMatch(i) = out.nPerBinMatched; end
end

% x tick labels
xlab = strings(nG,1);
for i = 1:nG
    if ~isnan(miceN(i)) && ~isnan(nMatch(i))
        xlab(i) = sprintf('%s (m=%d, n=%d/bin)', genoList(i), miceN(i), nMatch(i));
    else
        xlab(i) = string(genoList(i));
    end
end

% --- plot two panels ---
figure('Color','w');
tl = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% ---------------- CTX ----------------
nexttile;
bh = bar(MeanCTX); hold on;

for j = 1:numel(bh)
    x = bh(j).XEndPoints;
    y = bh(j).YEndPoints;
    e = SemCTX(:,j);
    eh = errorbar(x, y, e, 'k', 'LineStyle','none', 'LineWidth', 1);
    eh.HandleVisibility = 'off';
end

set(gca,'XTick',1:nG,'XTickLabel',cellstr(xlab));
xtickangle(25);
xlabel('Genotype');
ylabel('Fraction (normalized across CTX+STR)');
title('CTX','Interpreter','none');
ylim([0 1]); grid on;
legend(cellstr(modNames),'Location','best');

% mean±sem text
for j = 1:numel(bh)
    x = bh(j).XEndPoints;
    y = MeanCTX(:,j);
    e = SemCTX(:,j);
    for i = 1:numel(x)
        if isnan(y(i)) || isnan(e(i)), continue; end
        txt  = sprintf('%.3f \\pm %.3f', y(i), e(i));
        yPos = y(i) + e(i) + 0.02;
        text(x(i), yPos, txt, 'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', 'FontSize', 8, 'Interpreter','tex');
    end
end
hold off;

% ---------------- STR ----------------
nexttile;
bh = bar(MeanSTR); hold on;

for j = 1:numel(bh)
    x = bh(j).XEndPoints;
    y = bh(j).YEndPoints;
    e = SemSTR(:,j);
    eh = errorbar(x, y, e, 'k', 'LineStyle','none', 'LineWidth', 1);
    eh.HandleVisibility = 'off';
end

set(gca,'XTick',1:nG,'XTickLabel',cellstr(xlab));
xtickangle(25);
xlabel('Genotype');
ylabel('Fraction (normalized across CTX+STR)');
title('STR','Interpreter','none');
ylim([0 1]); grid on;
legend(cellstr(modNames),'Location','best');

% mean±sem text
for j = 1:numel(bh)
    x = bh(j).XEndPoints;
    y = MeanSTR(:,j);
    e = SemSTR(:,j);
    for i = 1:numel(x)
        if isnan(y(i)) || isnan(e(i)), continue; end
        txt  = sprintf('%.3f \\pm %.3f', y(i), e(i));
        yPos = y(i) + e(i) + 0.02;
        text(x(i), yPos, txt, 'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', 'FontSize', 8, 'Interpreter','tex');
    end
end
hold off;

% overall title (include nBoot if present)
if isfield(meta,'nBoot')
    title(tl, sprintf('Fig3-style bootstrap: CTX vs STR separated | nBoot=%d', meta.nBoot), 'Interpreter','none');
else
    title(tl, 'Fig3-style bootstrap: CTX vs STR separated', 'Interpreter','none');
end
