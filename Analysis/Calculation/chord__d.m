%% ===================== 0) Clean up =====================
clear; clc; 
%close all;

%% ===================== 1) 路径设置（改这里） =====================
% R 矩阵（带行列标签）的文件
pathR.WT  = 'F:\SeizurePTZ_idisco\YZ\YZ\5.WT.xlsx';
pathR.HOM = 'F:\SeizurePTZ_idisco\YZ\YZ\5.HOM.xlsx';

% 节点强度（含 Region 第一列 + 指标列）的文件
pathS.WT  = 'F:\SeizurePTZ_idisco\YZ\YZ\9.WT_75Module_results_regions.xlsx';
pathS.HOM = 'F:\SeizurePTZ_idisco\YZ\YZ\9.HOM_75Module_results_regions.xlsx';

% 指标列名（可用列名 'manual_wmdz' 或列号 5）
%%size_col = 'manual_wmdz';
size_col = 'degree';

% 节点面积范围（越大图上点越大）
nodeSizeRange = [28 160];

%% ===================== 2) 大脑宏组映射表 =====================
regionTable = {
'ACAd','isocortex'
'ACAv','isocortex'
'AId','isocortex'
'AIp','isocortex'
'Alv','isocortex'
'AOB','isocortex'
'AON','isocortex'
'AUDd','isocortex'
'AUDp','isocortex'
'AUDpo','isocortex'
'AUDv','isocortex'
'CA1','isocortex'
'CA2','isocortex'
'CA3','isocortex'
'COAa','isocortex'
'COAp','isocortex'
'DG','isocortex'
'DP','isocortex'
'ECT','isocortex'
'ENTl','isocortex'
'ENTm','isocortex'
'ENTmv','isocortex'
'FRP','isocortex'
'GU','isocortex'
'HPF','isocortex'
'ILA','isocortex'
'MOB','isocortex'
'MOp','isocortex'
'MOs','isocortex'
'NLOT','isocortex'
'ORBl','isocortex'
'ORBm','isocortex'
'ORBvl','isocortex'
'PAA','isocortex'
'PAR','isocortex'
'PERI','isocortex'
'PIR','isocortex'
'PL','isocortex'
'POST','isocortex'
'PRE','isocortex'
'RSPagl','isocortex'
'RSPd','isocortex'
'RSPv','isocortex'
'SSp-bfd','isocortex'
'SSp-ll','isocortex'
'SSp-m','isocortex'
'SSp-n','isocortex'
'SSp-tr','isocortex'
'SSp-ul','isocortex'
'SSs','isocortex'
'SUB','isocortex'
'TEa','isocortex'
'TR','isocortex'
'TT','isocortex'
'VISal','isocortex'
'VISam','isocortex'
'VISC','isocortex'
'VISl','isocortex'
'VISp','isocortex'
'VISpl','isocortex'
'VISpm','isocortex'
'VISrl6b','isocortex'
'BLA','cortical_subplate'
'BMA','cortical_subplate'
'CLA','cortical_subplate'
'EP','cortical_subplate'
'LA','cortical_subplate'
'PA','cortical_subplate'
'AAA','striatum'
'ACB','striatum'
'BA','striatum'
'CEA','striatum'
'CENT','striatum'
'FS','striatum'
'IA','striatum'
'LSX','striatum'
'MEA','striatum'
'OT','striatum'
'SF','striatum'
'BST','pallidum'
'GPe','pallidum'
'GPi','pallidum'
'MA','pallidum'
'MS','pallidum'
'NDB','pallidum'
'PAL','pallidum'
'SI','pallidum'
'TRS','pallidum'
'AD','thalamus'
'AM','thalamus'
'AV','thalamus'
'CL','thalamus'
'CM','thalamus'
'IAD','thalamus'
'IAM','thalamus'
'IGL','thalamus'
'IMD','thalamus'
'LD','thalamus'
'LGd','thalamus'
'LGv','thalamus'
'LH','thalamus'
'LP','thalamus'
'MD','thalamus'
'MG','thalamus'
'MH','thalamus'
'PCN','thalamus'
'PF','thalamus'
'PO','thalamus'
'POL','thalamus'
'PP','thalamus'
'PT','thalamus'
'PVT','thalamus'
'RE','thalamus'
'RH','thalamus'
'RT','thalamus'
'SGN','thalamus'
'SMT','thalamus'
'SPA','thalamus'
'SPF','thalamus'
'SubG','thalamus'
'VAL','thalamus'
'VM','thalamus'
'VPL','thalamus'
'VPLpc','thalamus'
'VPM','thalamus'
'VPMpc','thalamus'
'ADP','hypothalamus'
'AHN','hypothalamus'
'ARH','hypothalamus'
'AVP','hypothalamus'
'AVPV','hypothalamus'
'DMH','hypothalamus'
'FF','hypothalamus'
'LHA','hypothalamus'
'LM','hypothalamus'
'LPO','hypothalamus'
'MEPO','hypothalamus'
'MM','hypothalamus'
'MPN','hypothalamus'
'MPO','hypothalamus'
'OV','hypothalamus'
'PD','hypothalamus'
'PH','hypothalamus'
'PMd','hypothalamus'
'PMv','hypothalamus'
'PS','hypothalamus'
'PST','hypothalamus'
'PSTN','hypothalamus'
'pva','hypothalamus'
'PVH','hypothalamus'
'PVHd','hypothalamus'
'Pvi','hypothalamus'
'PVp','hypothalamus'
'PVpo','hypothalamus'
'RCH','hypothalamus'
'SBPV','hypothalamus'
'SCH','hypothalamus'
'SO','hypothalamus'
'STN','hypothalamus'
'SUM','hypothalamus'
'TM','hypothalamus'
'TU','hypothalamus'
'VLPO','hypothalamus'
'VMH','hypothalamus'
'ZI','hypothalamus'
'APN','midbrain'
'AT','midbrain'
'CLI','midbrain'
'CUN','midbrain'
'IC','midbrain'
'IF','midbrain'
'INC','midbrain'
'IPN','midbrain'
'IV','midbrain'
'LT','midbrain'
'MPT','midbrain'
'MRN','midbrain'
'NB','midbrain'
'ND','midbrain'
'NOT','midbrain'
'NPC','midbrain'
'OP','midbrain'
'PAG','midbrain'
'PBG','midbrain'
'PPN','midbrain'
'PPT','midbrain'
'PRC','midbrain'
'RN','midbrain'
'RR','midbrain'
'SAG','midbrain'
'SCm','midbrain'
'SCs','midbrain'
'SNc','midbrain'
'SNr','midbrain'
'VTA','midbrain'
'VTN','midbrain'
'CS','hindbrain'
'DTN','hindbrain'
'GRN','hindbrain'
'IRN','hindbrain'
'LC','hindbrain'
'LDT','hindbrain'
'MARN','hindbrain'
'NI','hindbrain'
'NLL','hindbrain'
'NTB','hindbrain'
'P','hindbrain'
'PARN','hindbrain'
'PB','hindbrain'
'PCG','hindbrain'
'PG','hindbrain'
'PRNc','hindbrain'
'PRNr','hindbrain'
'PSV','hindbrain'
'SOC','hindbrain'
'SG','hindbrain'
'SLC','hindbrain'
'SLD','hindbrain'
'SUT','hindbrain'
'TRN','hindbrain'
'VCO','hindbrain'
'VI','hindbrain'
'VII','hindbrain'
};

%% ===================== 3) 画图（WT + HOM 各一张） =====================
conds = {'WT','HOM'};
for c = 1:numel(conds)
    tag = conds{c};

    % ---- 3.1 读取 R（宽容对齐） ----
    pathR.(tag) = ensure_file(pathR.(tag), [tag '-R']);
    [labels, R] = load_R_with_headers_relaxed(pathR.(tag));
    [nR,nC] = size(R); assert(nR==nC && nR>0);
    N = nR;

    % ---- 3.2 宏组映射 + 规范化 ----
    groups = map_groups_from_regionTable(labels, regionTable);
    for k = 1:numel(groups)
        g = lower(strtrim(groups{k}));
        if strcmp(g,'insocortex'), g='isocortex'; end
        if strcmp(g,'hypothalamic'), g='hypothalamus'; end
        groups{k} = g;
    end

    % ---- 3.3 读取节点强度列并对齐（manual_wmdz） ----
    pathS.(tag) = ensure_file(pathS.(tag), [tag '-sizes']);
    [reg_all, sval_all] = read_region_s_relaxed(pathS.(tag), size_col);
    [nodeSize, miss_in_file, extra_in_file] = map_sizes_with_report(labels, reg_all, sval_all, nodeSizeRange);
    if ~isempty(miss_in_file)
        fprintf('[%s] ⚠ Missing in %s (size filled by median): %s\n', tag, pathS.(tag), strjoin(miss_in_file, ', '));
    end
    if ~isempty(extra_in_file)
        fprintf('[%s] ℹ Extra in %s (ignored): %s\n', tag, pathS.(tag), strjoin(extra_in_file, ', '));
    end

    % ---- 3.4 颜色：宏组为基色，按 manual_wmdz 调亮度 ----
    baseHex = struct('isocortex','#4F81BD','cortical_subplate','#9BBB59','striatum','#C0504D', ...
                     'pallidum','#8064A2','thalamus','#F79646','hypothalamus','#2E8B57', ...
                     'midbrain','#0099C6','hindbrain','#8B4513','unknown','#999999');
    hex2rgb = @(h) sscanf(char(regexprep(h,'#','')), '%2x%2x%2x', [3,1]).'/255;

    glist = unique(groups,'stable');
    gColor = containers.Map;
    for i = 1:numel(glist)
        key = matlab.lang.makeValidName(glist{i});
        if isfield(baseHex,key), gColor(glist{i}) = hex2rgb(baseHex.(key));
        else, gColor(glist{i}) = [0.6 0.6 0.6];
        end
    end

    [s_aligned, miss_c, ~, vmin, vmax] = align_values(labels, reg_all, sval_all);
    t = (s_aligned - vmin) / max(vmax - vmin, eps);
    t = max(0,min(1,t)).^0.8;  % γ=0.8 提升对比
    nodeColors = zeros(N,3);
    for i=1:N
        base = gColor(groups{i});
        nodeColors(i,:) = (1-t(i))*[1 1 1] + t(i)*base;
    end
    if ~isempty(setdiff(miss_c, miss_in_file))
        fprintf('[%s] ⚠ Some regions missing for color only (filled by median).\n', tag);
    end

    % ---- 3.5 布局与参数（含正负阈值）----
    theta = linspace(0,2*pi,N+1); theta(end) = [];
    r = 1; cx = r*cos(theta); cy = r*sin(theta);

    edgeAlpha      = 0.12;
    edgeWidthBase  = 1.2/4;
    edgeWidthScale = 1.2/4;
    labelFS        = 7;

    % 边颜色模式：'fixed' | 'source' | 'target' | 'blend'
    edgeMode = 'source';
    fixedPos = [199,46,41]/255;   % 正相关固定色（红）
    fixedNeg = [55,126,184]/255;  % 负相关固定色（蓝）

    % ——阈值（你可以改）——
    negThr       = -0.5;  % 负相关阈值：rij <= negThr
    posThr       =  0.9;  % 正相关阈值：rij >= posThr
    drawPositive = true;  % 是否绘制正相关
    drawNegative = false;  % 是否绘制负相关

    % ---- 3.6 绘图 ----
    f = figure('Color','w','Position',[100 100 900 900]); set(f,'Renderer','painters');
    hold on; axis equal off
    th = linspace(0,2*pi,360);
    plot(r*cos(th), r*sin(th),'Color',[0.85 0.85 0.85],'LineWidth',1);

    % 分段色环（宏组）
    for i=1:N
        p0=[cx(i) cy(i)]; p1=0.97*p0;
        plot([p0(1) p1(1)],[p0(2) p1(2)],'-','Color',gColor(groups{i}),'LineWidth',3);
    end

    % 取边颜色
    getRGB = @(i,j,pos) ...
        (strcmp(edgeMode,'fixed'))  * (pos*fixedPos + (~pos)*fixedNeg) + ...
        (strcmp(edgeMode,'source')) * nodeColors(i,:) + ...
        (strcmp(edgeMode,'target')) * nodeColors(j,:) + ...
        (strcmp(edgeMode,'blend'))  * ((nodeColors(i,:)+nodeColors(j,:))/2);

    % 弦
    for i=1:N
        for j=i+1:N
            rij = R(i,j);

            % ------- 负相关 -------
            if drawNegative && (rij <= negThr)
                x1=cx(i); y1=cy(i); x2=cx(j); y2=cy(j);
                Cx=(x1+x2)/2; Cy=(y1+y2)/2 + 0.5*abs(x1-x2);   % 向内拱
                cr=hypot(Cx,Cy); if cr>0.98, s=0.9/cr; Cx=Cx*s; Cy=Cy*s; end
                t_ = linspace(0,1,120);
                bx=(1-t_).^2*x1 + 2*(1-t_).*t_*Cx + t_.^2*x2;
                by=(1-t_).^2*y1 + 2*(1-t_).*t_*Cy + t_.^2*y2;
                lw=edgeWidthBase + edgeWidthScale*(abs(rij)-abs(negThr))/(1-abs(negThr));
                plot(bx,by,'Color',[getRGB(i,j,false) edgeAlpha],'LineWidth',max(lw,0.5/4));
            end

            % ------- 正相关（新加） -------
            if drawPositive && (rij >= posThr)
                x1=cx(i); y1=cy(i); x2=cx(j); y2=cy(j);
                Cx=(x1+x2)/2; Cy=(y1+y2)/2 - 0.5*abs(x1-x2);   % 反向拱，便于区分
                cr=hypot(Cx,Cy); if cr>0.98, s=0.9/cr; Cx=Cx*s; Cy=Cy*s; end
                t_ = linspace(0,1,120);
                bx=(1-t_).^2*x1 + 2*(1-t_).*t_*Cx + t_.^2*x2;
                by=(1-t_).^2*y1 + 2*(1-t_).*t_*Cy + t_.^2*y2;
                lw=edgeWidthBase + edgeWidthScale*(abs(rij)-abs(posThr))/(1-abs(posThr));
                plot(bx,by,'Color',[getRGB(i,j,true) edgeAlpha],'LineWidth',max(lw,0.5/4));
            end
        end
    end

    % 节点 + 标签
    scatter(cx, cy, nodeSize, nodeColors, 'filled','MarkerEdgeColor',[0.2 0.2 0.2]);
    for i=1:N
        tx=1.06*cx(i); ty=1.06*cy(i); ang=rad2deg(theta(i));
        if ang>90 && ang<270, rot=ang+180; ha='right'; else, rot=ang; ha='left'; end
        text(tx,ty, labels{i}, 'FontSize',labelFS, 'Rotation',rot, ...
            'HorizontalAlignment',ha, 'VerticalAlignment','middle', ...
            'Color', gColor(groups{i}));
    end
    title(sprintf('Chord diagram (Neg ≤ %.2f, Pos ≥ %.2f) — %s', negThr, posThr, tag),'FontSize',14);

    % 图例
    lgdEntries = {}; lgdColors = [];
    for i=1:numel(glist)
        if any(strcmp(groups,glist{i})), lgdEntries{end+1}=glist{i}; lgdColors(end+1,:)=gColor(glist{i}); end
    end
    hL = gobjects(numel(lgdEntries),1);
    for i=1:numel(lgdEntries)
        hL(i) = scatter(nan,nan,60,lgdColors(i,:), 'filled', 'MarkerEdgeColor',[0.2 0.2 0.2]);
    end
    legend(hL, lgdEntries, 'Location','southoutside','NumColumns',3,'Box','off');

    % 可选导出
    % exportgraphics(gcf, sprintf('chord_%s_pos%.2f_neg%.2f.png', tag, posThr, negThr), 'Resolution', 300);
end

%% ===================== 4) 辅助函数 =====================
function p = ensure_file(p, tag)
    if isstring(p), p=char(p); end
    if isfile(p), return; end
    [f, fp] = uigetfile({'*.xlsx;*.csv','Tables (*.xlsx,*.csv)'}, sprintf('Select file for %s', tag));
    if isequal(f,0), error('[%s] No file selected.', tag); end
    p = fullfile(fp,f);
end

% ——行列标签宽容对齐的 R 读入（保留原列名 + 把 空格/连字符/下划线 视作等价）——
function [labels, R] = load_R_with_headers_relaxed(xlsxPath)
    T = readtable(xlsxPath,'ReadVariableNames',true,'VariableNamingRule','preserve');
    colNames = string(T.Properties.VariableNames(2:end));
    rowNames = string(T{:,1});
    data = T{:,2:end}; if ~isnumeric(data), data = str2double(string(data)); end

    normKey = @(s) lower(regexprep(strtrim(string(s)),'[\s\-_]+','_'));
    rowKey = normKey(rowNames);
    colKey = normKey(colNames);

    % 列名规范化后去重（保留首次）
    if numel(unique(colKey)) < numel(colKey)
        [~, ia] = unique(colKey,'stable');
        colKey   = colKey(ia);
        colNames = colNames(ia);
        data     = data(:, ia);
        warning('%s: duplicate column names after normalization; first occurrences kept.', xlsxPath);
    end

    [tf, loc] = ismember(rowKey, colKey);
    if all(tf)
        data = data(:,loc);
        labels = string(rowNames);
    else
        common = intersect(rowKey, colKey, 'stable');
        onlyRow = setdiff(rowKey, colKey, 'stable');
        onlyCol = setdiff(colKey, rowKey, 'stable');
        if ~isempty(onlyRow)
            fprintf('Only in ROW (dropped): %s\n', strjoin(string(rowNames(ismember(rowKey,onlyRow))), ', '));
        end
        if ~isempty(onlyCol)
            fprintf('Only in COLUMN (dropped): %s\n', strjoin(string(colNames(ismember(colKey,onlyCol))), ', '));
        end
        [~, iR] = ismember(common, rowKey);
        [~, iC] = ismember(common, colKey);
        data   = data(iR, iC);
        labels = string(rowNames(iR));
        warning('Aligned to intersection of %d regions.', numel(labels));
    end

    % 对称化、对角置零、截断
    R = (data + data.')/2;
    R(1:size(R,1)+1:end) = 0;
    R = max(min(R,1),-1);
end

% ——根据 regionTable 赋宏组——
function groups = map_groups_from_regionTable(labels, regionTable)
    K = lower(regionTable(:,1)); V = regionTable(:,2);
    M = containers.Map(K, V);
    groups = cell(numel(labels),1);
    for i=1:numel(labels)
        k = lower(strtrim(labels{i}));
        if isKey(M,k), groups{i}=M(k); else, groups{i}='unknown'; end
    end
end

% ——宽容列名匹配读取“节点强度”列（manual_wmdz）——
function [regions, svals] = read_region_s_relaxed(path, colsel)
    T = readtable(path,'ReadVariableNames',true,'VariableNamingRule','preserve');
    regions = string(T{:,1});
    if ischar(colsel) || isstring(colsel)
        want = lower(regexprep(strtrim(string(colsel)),'[\s_]+',''));   % manual_wmdz -> manualwmdz
        rawNames  = string(T.Properties.VariableNames);
        normNames = lower(regexprep(strtrim(rawNames),'[\s_]+',''));
        hit = find(strcmp(normNames, want), 1);
        assert(~isempty(hit), 'Column "%s" not found in %s. Available: %s', string(colsel), path, strjoin(rawNames, ', '));
        svals = T.(rawNames(hit));
    else
        colidx = double(colsel);
        assert(colidx>=1 && colidx<=width(T), 'Column index %d out of range 1..%d', colidx, width(T));
        svals = T{:, colidx};
    end
    if ~isnumeric(svals), svals = str2double(string(svals)); end
end

% ——把节点强度映射为点面积（并报告缺失/冗余）——
function [nodeSize, missingIn, extraIn] = map_sizes_with_report(labels, regionExcel, sExcel, nodeSizeRange)
    key = lower(strtrim(cellstr(string(regionExcel))));
    val = sExcel(:); if ~isnumeric(val), val = str2double(string(val)); end
    good = ~cellfun(@isempty,key) & ~isnan(val); key=key(good); val=val(good);
    M = containers.Map; for i=1:numel(key), M(key{i}) = val(i); end

    L  = numel(labels); labKey = lower(strtrim(cellstr(string(labels))));
    raw = nan(L,1); for i=1:L, k=labKey{i}; if isKey(M,k), raw(i)=M(k); end, end

    missingMask = isnan(raw);
    missingIn   = string(labels(missingMask));
    extraIn     = setdiff(string(regionExcel), string(labels), 'stable');

    if all(isnan(raw)), error('No overlapping node-size values.'); end
    medv = median(raw(~isnan(raw))); raw(missingMask) = medv;

    mn=min(raw); mx=max(raw);
    if mx==mn
        nodeSize = mean(nodeSizeRange)*ones(L,1);
    else
        t=(raw-mn)/(mx-mn);
        nodeSize = nodeSizeRange(1) + t*diff(nodeSizeRange);
    end
end

% ——把一列值对齐到 labels（用于颜色密度）——
function [vals_aligned, missingIn, extraIn, vmin, vmax] = align_values(labels, regionExcel, vExcel)
    key = lower(strtrim(cellstr(string(regionExcel))));
    val = vExcel(:); if ~isnumeric(val), val = str2double(string(val)); end
    good = ~cellfun(@isempty,key) & ~isnan(val); key=key(good); val=val(good);
    M = containers.Map; for i=1:numel(key), M(key{i})=val(i); end

    L  = numel(labels); labKey = lower(strtrim(cellstr(string(labels))));
    raw = nan(L,1); for i=1:L, k=labKey{i}; if isKey(M,k), raw(i)=M(k); end, end

    missingIn = string(labels(isnan(raw)));
    extraIn   = setdiff(string(regionExcel), string(labels), 'stable');

    medv = median(raw(~isnan(raw))); raw(isnan(raw)) = medv;
    vals_aligned = raw; vmin=min(raw); vmax=max(raw);
end
