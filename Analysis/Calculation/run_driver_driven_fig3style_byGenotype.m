function [connData, res] = run_driver_driven_fig3style_byGenotype(connData, varargin)
% RUN_DRIVER_DRIVEN_FIG3STYLE_BYGENOTYPE
% ------------------------------------------------------------
% Per-genotype Fig3-style bootstrap with matched sampling across CTX/STR bins.
%
% Driver/Driven definition (y-axis neurons = rows):
%   rowScore > thr_pos (=0.2e-7)  => driver (label=2)
%   rowScore < thr_neg (=-0.2e-7) => driven (label=3)
%   else                          => other  (label=1)
%
% Requires connData struct array with fields:
%   - mouseName (string/cellstr/char)
%   - connMat   (NxN double)
%   - is_CTX    (Nx1 logical)
%   - is_STR    (Nx1 logical)
%
% Adds fields:
%   - connData(i).moduleScore
%   - connData(i).moduleLabel
%   - connData(i).genotype
%
% Output res.(geno) with:
%   - meanFrac (2 mods x 2 bins), sdFrac, semFrac, fracBoot
%   - pSig (mods x 1): p-value for CTX vs STR within each module
%   - nMice, nUnitsCTX, nUnitsSTR, nPerBinMatched, etc.
%
% Plot:
%   - show mean±SEM on bars
%   - show significance between CTX and STR bars within each module (Driver/Driven)

% ---------------- Parse inputs ----------------
p = inputParser;
p.addParameter('thr_pos', 0.2e-7, @(x)isscalar(x) && isnumeric(x));
p.addParameter('thr_neg', -0.2e-7, @(x)isscalar(x) && isnumeric(x));
p.addParameter('scoreMethod', 'rowMean', @(s)ischar(s) || isstring(s));
p.addParameter('nBoot', 100, @(x)isscalar(x) && x>=10);
p.addParameter('seed', 1, @(x)isscalar(x) && isnumeric(x));
p.addParameter('plotFig', true, @(x)islogical(x) && isscalar(x));
p.addParameter('genoOrder', ["WT","HOM","HOMcon","HOMkv11"], @(x)isstring(x) || iscellstr(x));

% NEW plot controls
p.addParameter('showMeanSemText', true, @(x)islogical(x) && isscalar(x));
p.addParameter('showSignificance', true, @(x)islogical(x) && isscalar(x));
p.addParameter('sigShowPValue', true, @(x)islogical(x) && isscalar(x)); % show "p=..." under stars
p.addParameter('textFontSize', 8, @(x)isscalar(x) && x>=6);

p.parse(varargin{:});

thr_pos     = p.Results.thr_pos;
thr_neg     = p.Results.thr_neg;
scoreMethod = string(p.Results.scoreMethod);
nBoot       = p.Results.nBoot;
seed        = p.Results.seed;
plotFig     = p.Results.plotFig;
genoOrder   = string(p.Results.genoOrder);

showMeanSemText = p.Results.showMeanSemText;
showSignificance = p.Results.showSignificance;
sigShowPValue = p.Results.sigShowPValue;
textFontSize = p.Results.textFontSize;

rng(seed);

% ---------------- Step 0: infer genotype for each mouse ----------------
for iM = 1:numel(connData)
    if ~isfield(connData(iM),'mouseName') || isempty(connData(iM).mouseName)
        connData(iM).genotype = "Unknown";
    else
        connData(iM).genotype = infer_genotype(connData(iM).mouseName);
    end
end

% ---------------- Step 1: label driver/driven per mouse ----------------
for iM = 1:numel(connData)
    A = double(connData(iM).connMat);
    N = size(A,1);

    isCTX = logical(connData(iM).is_CTX(:));
    isSTR = logical(connData(iM).is_STR(:));
    if numel(isCTX) ~= N || numel(isSTR) ~= N
        error('Mouse %d: is_CTX/is_STR length does not match connMat size.', iM);
    end

    % remove diagonal for score calculation
    A(1:N+1:end) = NaN;

    score = compute_row_score(A, thr_pos, thr_neg, scoreMethod);

    label = ones(N,1);           % 1=other
    label(score > thr_pos) = 2;  % 2=driver
    label(score < thr_neg) = 3;  % 3=driven

    connData(iM).moduleScore = score;
    connData(iM).moduleLabel = label;
end

% ---------------- Step 2: per-genotype bootstrap ----------------
modIDs   = [2 3];
modNames = ["Driver","Driven"];
binNames = ["CTX","STR"];

% collect genotype list present
allG = string({connData.genotype});
present = unique(allG);

% keep desired order, then append other genotypes if any
genoList = genoOrder(ismember(genoOrder, present));
extra    = present(~ismember(present, genoOrder) & present~="Unknown");
genoList = [genoList, sort(extra)];

res = struct();
res.meta = struct('thr_pos',thr_pos,'thr_neg',thr_neg,'scoreMethod',char(scoreMethod), ...
                  'nBoot',nBoot,'seed',seed,'modIDs',modIDs,'modNames',modNames,'binNames',binNames, ...
                  'genoList',genoList);

for g = 1:numel(genoList)
    geno = genoList(g);
    miceIdx = find(allG == geno);

    out = struct();
    out.genotype = geno;
    out.miceIdx  = miceIdx(:);
    out.nMice    = numel(miceIdx);

    % ---- Pool across mice (within genotype) ----
    allBin   = []; % 1=CTX, 2=STR
    allLabel = []; % 1/2/3
    for k = 1:numel(miceIdx)
        iM = miceIdx(k);
        N  = size(connData(iM).connMat,1);

        isCTX = logical(connData(iM).is_CTX(:));
        isSTR = logical(connData(iM).is_STR(:));

        bin = nan(N,1);
        bin(isCTX) = 1;
        bin(isSTR) = 2;

        label = double(connData(iM).moduleLabel(:));

        keep = ~isnan(bin) & ~isnan(label);
        allBin   = [allBin;   bin(keep)];
        allLabel = [allLabel; label(keep)];
    end

    idxCTX = find(allBin==1);
    idxSTR = find(allBin==2);

    out.nUnitsCTX = numel(idxCTX);
    out.nUnitsSTR = numel(idxSTR);

    if isempty(idxCTX) || isempty(idxSTR)
        warning('Genotype %s: CTX or STR empty after pooling. Skipping.', geno);
        out.fracBoot = [];
        out.meanFrac = nan(numel(modIDs),2);
        out.sdFrac   = nan(numel(modIDs),2);
        out.semFrac  = nan(numel(modIDs),2);
        out.pSig     = nan(numel(modIDs),1);
        out.nPerBinMatched = 0;
        res.(matlab.lang.makeValidName(geno)) = out;
        continue;
    end

    % match sampling numbers across bins
    nPerBin = min(numel(idxCTX), numel(idxSTR));
    out.nPerBinMatched = nPerBin;

    % ---- Bootstrap ----
    fracBoot = nan(nBoot, numel(modIDs), 2);
    for b = 1:nBoot
        sCTX = idxCTX(randperm(numel(idxCTX), nPerBin));
        sSTR = idxSTR(randperm(numel(idxSTR), nPerBin));

        for m = 1:numel(modIDs)
            mid = modIDs(m);

            pCTX = mean(allLabel(sCTX) == mid);
            pSTR = mean(allLabel(sSTR) == mid);

            denom = pCTX + pSTR;
            if denom == 0
                fracCTX = NaN; fracSTR = NaN;
            else
                fracCTX = pCTX / denom;
                fracSTR = pSTR / denom;
            end

            fracBoot(b,m,1) = fracCTX;
            fracBoot(b,m,2) = fracSTR;
        end
    end

    out.fracBoot = fracBoot;
    out.meanFrac = squeeze(mean(fracBoot,1,'omitnan')); % (mod x bin)
    out.sdFrac   = squeeze(std(fracBoot,0,1,'omitnan'));
    out.semFrac  = out.sdFrac ./ sqrt(nBoot);

    % ---- Significance: CTX vs STR within each module using bootstrap distribution ----
    out.pSig = nan(numel(modIDs),1);
    for m = 1:numel(modIDs)
        d = fracBoot(:,m,1) - fracBoot(:,m,2); % CTX - STR
        out.pSig(m) = p_from_bootstrap_diff(d); % two-sided
    end

    res.(matlab.lang.makeValidName(geno)) = out;
end

% ---------------- Step 3: Plot 2x2 panels (optional) ----------------
if plotFig
    figure('Color','w');
    tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    for g = 1:min(4,numel(genoOrder)) % enforce 4 panels for your 4 genotypes
        geno = genoOrder(g);
        nexttile;

        field = matlab.lang.makeValidName(geno);
        if ~isfield(res, field)
            title(sprintf('%s (no data)', geno));
            axis off;
            continue;
        end

        out = res.(field);
        M = out.meanFrac';   % bins x mods
        E = out.semFrac';    % bins x mods

        bh = bar(M); hold on;
        set(gca,'XTick',1:2,'XTickLabel',cellstr(binNames));
        xlabel('Region Bin');
        ylabel('Fraction (normalized across CTX+STR)');
        title(sprintf('%s | nBoot=%d | matched n=%d/bin | mice=%d', ...
            geno, nBoot, out.nPerBinMatched, out.nMice), 'Interpreter','none');

        % error bars
        for j = 1:numel(bh)
            x = bh(j).XEndPoints;
            y = bh(j).YEndPoints;
            e = E(:,j);
            eh = errorbar(x, y, e, 'k', 'LineStyle','none', 'LineWidth', 1);
            eh.HandleVisibility = 'off'; % prevent "data1/data2" legend clutter
        end

        ylim([0 1]);
        grid on;

        if g == 1
            legend(cellstr(modNames), 'Location','bestoutside');
        end

        % ---- Show mean±SEM text on each bar ----
        if showMeanSemText
            add_mean_sem_text(bh, M, E, textFontSize);
        end

        % ---- Significance brackets: CTX vs STR within Driver; CTX vs STR within Driven ----
        if showSignificance && ~isempty(out.fracBoot)
            % positions: bh(1)=Driver, bh(2)=Driven
            for j = 1:numel(bh)
                x1 = bh(j).XEndPoints(1); % CTX
                x2 = bh(j).XEndPoints(2); % STR
                y1 = M(1,j); e1 = E(1,j);
                y2 = M(2,j); e2 = E(2,j);

                yTop = max([y1+e1, y2+e2]) + 0.06 + 0.03*(j-1); % stagger
                pval = out.pSig(j);

                add_sig_bracket(x1, x2, yTop, pval, sigShowPValue, textFontSize);
            end
        end

        hold off;
    end

    title(tl, 'Fig3-style bootstrap by genotype (CTX vs STR): mean±SEM + significance');
end

end

% ============================================================
% Helper: compute row score (y-axis neuron score)
% ============================================================
function score = compute_row_score(A, thr_pos, thr_neg, scoreMethod)
% A: NxN with diagonal already NaN
switch lower(scoreMethod)
    case 'rowmean'
        score = mean(A, 2, 'omitnan');
    case 'rowmedian'
        score = median(A, 2, 'omitnan');
    case 'rowsum'
        score = sum(A, 2, 'omitnan');
    case 'rowposfrac'
        % unitless fraction in [0,1]
        score = mean(A > thr_pos, 2, 'omitnan');
    case 'rownegfrac'
        % unitless negative fraction in [-1,0]
        score = -mean(A < thr_neg, 2, 'omitnan');
    otherwise
        error('Unknown scoreMethod: %s', scoreMethod);
end
score = double(score(:));
end

% ============================================================
% Helper: infer genotype from mouseName (robust)
% ============================================================
function geno = infer_genotype(mouseName)
s = string(mouseName);
s = lower(strtrim(s));
s = replace(s, "_", "");
s = replace(s, "-", "");
s = replace(s, " ", "");

% IMPORTANT: check longer patterns first (e.g., homcon, homkv11)
if contains(s, "homcon")
    geno = "HOMcon";
elseif contains(s, "homkv11") || contains(s, "homkv1.1") || contains(s, "kv11")
    geno = "HOMkv11";
elseif startsWith(s, "wt")
    geno = "WT";
elseif startsWith(s, "hom")
    geno = "HOM";
else
    geno = "Unknown";
end
end

% ============================================================
% Helper: p-value from bootstrap difference (two-sided)
% ============================================================
function p = p_from_bootstrap_diff(d)
d = d(:);
d = d(~isnan(d));
if isempty(d)
    p = NaN;
    return;
end
pPos = mean(d >= 0);
pNeg = mean(d <= 0);
p = 2 * min(pPos, pNeg);
p = max(min(p,1),0);
end

% ============================================================
% Helper: add mean±SEM on bars
% bh: bar handles (1 x nMods)
% M: bins x mods, E: bins x mods
% ============================================================
function add_mean_sem_text(bh, M, E, fs)
for j = 1:numel(bh)
    x = bh(j).XEndPoints; % 1x2 (CTX, STR)
    y = M(:,j);           % 2x1
    e = E(:,j);           % 2x1

    for b = 1:numel(x)
        txt = sprintf('%.3f \\pm %.3f', y(b), e(b));
        yPos = y(b) + e(b) + 0.02;
        text(x(b), yPos, txt, 'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', 'FontSize', fs, 'Interpreter','tex');
    end
end
end

% ============================================================
% Helper: add significance bracket + stars (+ optional p text)
% ============================================================
function add_sig_bracket(x1, x2, y, pval, showP, fs)
if isnan(pval)
    label = 'n/a';
else
    label = stars_from_p(pval);
    if showP
        label = sprintf('%s\np=%.3g', label, pval);
    end
end

% bracket height
h = 0.02;

plot([x1 x1 x2 x2], [y-h y y y-h], 'k-', 'LineWidth', 1);

text(mean([x1 x2]), y+0.005, label, ...
    'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
    'FontSize', fs, 'Interpreter','none');
end

function s = stars_from_p(p)
if isnan(p)
    s = 'n/a';
elseif p < 1e-4
    s = '****';
elseif p < 1e-3
    s = '***';
elseif p < 1e-2
    s = '**';
elseif p < 0.05
    s = '*';
else
    s = 'ns';
end
end
