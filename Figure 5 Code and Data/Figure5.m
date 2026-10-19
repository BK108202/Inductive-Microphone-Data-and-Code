%% Figure 4: grouped application plots (MATLAB R2020a+)
% Signal Processing Toolbox required. Put CSVs beside this script, or edit
% cfg.dataDir. Input columns: timestamp in microseconds, LDC output in counts.
% Change input filenames ONLY in the first column of records below.
% Plot selection uses task labels, never filenames or filename-based fields.
% Outputs: respiration/mastication/swallowing 3x3; each pulse 1x3; combined 1x2.
% Only PNG files are saved. No panel letters, FIG, PDF, SVG, CSV or MAT output.
clear; close all; clc;
scriptDir=fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir=pwd; end
cfg.dataDir=scriptDir;
cfg.outDir=fullfile(scriptDir,'Figure5_results');
cfg.fontName='Arial'; cfg.fontSize=17;
cfg.resolution=300; cfg.keepFiguresOpen=true;
% Shared physical geometry: identical plotting rectangles in all PNGs.
% Combined spans exactly the last two column slots of a three-column figure.
cfg.plotWidth_cm=8.0; cfg.plotHeight_cm=5.5;
cfg.columnPitch_cm=11.5; cfg.rowPitch_cm=7.8;
cfg.leftMargin_cm=2.5; cfg.rightMargin_cm=3.0;
cfg.bottomMargin_cm=1.9; cfg.topMargin_cm=1.1;
cfg.colorbarGap_cm=0.25; cfg.colorbarWidth_cm=0.30;
cfg.showColumnTitles=true;
cfg.filterOrder=3;
cfg.motionSTFTWindow_s=2;
cfg.STFTOverlap=0.9;
cfg.spectrogramDynamicRange_dB=50;
cfg.spectrogramCLim_dB=[]; % e.g. [30 90]; [] = max minus dynamic range
cfg.maxGap_s=0.01;

%% INPUT FILES: edit filenames here only
% Columns: filename, task label, highpass Hz (0 = none), lowpass Hz.
% Keep the task labels unchanged when replacing datasets.
% Every Pulse row is plotted automatically; no separate pulseFiles list.
% Use '' as a filename to disable a recording, or remove its row.
records={'1.csv','Respiration',0,2; ...
    '2.csv','Mastication',0,20; ...
    '3.csv','Swallowing',0,20; ...
    '4.csv','Combined',0,20; ...
    '5.csv','Pulse',0.5,15};
% Optional crops [start end] in seconds; NaN end = full duration.
% Filtering uses the full record; STFT and Welch use the selected crop.
cfg.viewWindows_s=repmat([0 NaN],size(records,1),1);

if ~exist(cfg.outDir,'dir'), mkdir(cfg.outDir); end

% Cell entries stay aligned with records, including missing/disabled files.
% A filename can contain spaces, hyphens or other non-identifier characters.
Results=cell(size(records,1),1); Summary=table();
for k=1:size(records,1)
    name=records{k,1}; label=records{k,2};
    if isempty(name), continue; end
    path=fullfile(cfg.dataDir,name);
    if ~isfile(path), warning('Missing %s; skipped.',path); continue; end
    M=readmatrix(path);
    if size(M,2)<2, warning('Need two columns: %s; skipped.',name); continue; end
    M=M(:,1:2); M=M(all(isfinite(M),2),:);
    if size(M,1)<30, warning('Too few samples: %s',name); continue; end
    t=(M(:,1)-M(1,1))*1e-6; y=M(:,2);
    if any(diff(t)<=0)
        error('%s: timestamps must increase; check resets/duplicates.',name);
    end
    dt=diff(t); fs=1/median(dt);
    if max(dt)>cfg.maxGap_s
        warning('%s has a %.4fs gap; inspect before changing maxGap_s. Skipped.',name,max(dt));
        continue;
    end
    tu=(0:1/fs:t(end))'; u=interp1(t,y,tu,'linear');
    centered=u-median(u);
    motion=bandFilter(centered,fs,[records{k,3} records{k,4}],cfg.filterOrder);
    lo=cfg.viewWindows_s(k,1); hi=cfg.viewWindows_s(k,2);
    if isnan(hi), hi=tu(end); end
    hi=min(hi,tu(end)); mask=tu>=lo & tu<=hi;
    if nnz(mask)<30, error('Invalid view window for %s',name); end
    R=struct('file',name,'label',label,'time_s',tu, ...
        'centered_counts',centered,'motion_counts',motion,'Fs_Hz',fs, ...
        'viewWindow_s',[lo hi],'filterBand_Hz',[records{k,3} records{k,4}]);
    R.spec=calculateSpectra(tu(mask),centered(mask),fs,cfg);
    Results{k}=R;
    row=table(string(name),string(label),t(end),fs,1/mean(dt),max(dt), ...
        std(dt)/mean(dt),numel(unique(y)), ...
        'VariableNames',{'File','Task','Duration_s','MedianFs_Hz', ...
        'MeanFs_Hz','MaxGap_s','IntervalCV','UniqueOutputLevels'});
    Summary=[Summary;row]; %#ok<AGROW>
end

% First figure: columns = tasks, rows = waveform / spectrogram / ASD.
appTasks={'Respiration','Mastication','Swallowing'};
appRows=zeros(1,numel(appTasks));
for c=1:numel(appTasks)
    idx=find(strcmp(records(:,2),appTasks{c}));
    if numel(idx)>1
        error('Use only one %s row in records.',appTasks{c});
    elseif ~isempty(idx)
        appRows(c)=idx;
    end
end
appReady=all(appRows>0);
if appReady, appReady=all(~cellfun(@isempty,Results(appRows))); end
if appReady
    R=Results(appRows);
    f=newFigure(cfg,3,3,'Respiration, mastication and swallowing');
    ax=gobjects(3,3);
    % One shared colorbar; only display limits are changed.
    clim=sharedColorLimits(R,cfg);
    asdLimits=sharedASDLimits(R);
    commonBand=[0 min(cellfun(@(r) r.spec.band_Hz(2),R))];
    commonFreq=[max(cellfun(@(r) r.spec.asdXLim(1),R)) commonBand(2)];
    for c=1:3
        ax(1,c)=panelAxes(f,3,1,c,cfg);
        plotWaveform(ax(1,c),R{c},true,cfg);
        if cfg.showColumnTitles, title(ax(1,c),R{c}.label,'FontWeight','normal'); end
        if c==1, ylabel(ax(1,c),'Output change (counts)'); end
        ax(1,c).XTickLabel=[];
        if c==1
            lg=legend(ax(1,c),{'Median-centered raw','Filtered'}, ...
                'Location','best','Box','off');
            set(lg,'FontName',cfg.fontName,'FontSize',cfg.fontSize);
        end
        ax(2,c)=panelAxes(f,3,2,c,cfg);
        plotSpectrogram(ax(2,c),R{c},clim,cfg);
        ylim(ax(2,c),commonBand); limitYTicks(ax(2,c));
        if c==1, ylabel(ax(2,c),'Frequency (Hz)');
        else, ax(2,c).YTickLabel=[]; end
        if c==2, xlabel(ax(2,c),'Time (s)'); end
        linkaxes(ax(1:2,c),'x');
        xlim(ax(2,c),R{c}.viewWindow_s);
        ax(3,c)=panelAxes(f,3,3,c,cfg);
        plotASD(ax(3,c),R{c},cfg);
        xlim(ax(3,c),commonFreq); ylim(ax(3,c),asdLimits);
        limitYTicks(ax(3,c));
        if c==1, ylabel(ax(3,c),'Output ASD (counts/\surdHz)');
        else, ax(3,c).YTickLabel=[]; end
    end
    xlabel(ax(3,2),'Frequency (Hz)');
    addColorbar(ax(2,3),cfg);
    savePNG(f,'Figure4_01_respiration_mastication_swallowing',cfg);
else
    warning(['The 3x3 figure was skipped. Check the filenames and data for ' ...
        'the Respiration, mastication and Swallowing rows in records.']);
end

% Second layout: every Pulse row, waveform / spectrogram / ASD.
pulseRows=find(strcmp(records(:,2),'Pulse'));
for p=1:numel(pulseRows)
    k=pulseRows(p);
    if isempty(Results{k}), continue; end
    R=Results{k};
    f=newFigure(cfg,1,1,['Pulse: ' R.file]);
    a1=panelAxes(f,1,1,1,cfg); plotWaveform(a1,R,true,cfg);
    xlabel(a1,'Time (s)'); ylabel(a1,'Output change (counts)');
    lg=legend(a1,{'Median-centered raw','Filtered'},'Location','best','Box','off');
    set(lg,'FontName',cfg.fontName,'FontSize',cfg.fontSize);
    [~,fileStem]=fileparts(R.file);
    savePNG(f,['Figure4_02_pulse_' fileStem],cfg);
end

% Third figure: filtered combined waveform / centered-raw spectrogram.
combinedRow=find(strcmp(records(:,2),'Combined'));
if numel(combinedRow)>1
    error('Use only one Combined row in records.');
end
if ~isempty(combinedRow) && ~isempty(Results{combinedRow})
    R=Results{combinedRow};
    f=newFigure(cfg,1,2,'Combined');
    a1=panelAxes(f,1,1,1,cfg); plotWaveform(a1,R,false,cfg);
    ylabel(a1,'Output change (counts)');
    a2=panelAxes(f,1,1,2,cfg); plotSpectrogram(a2,R,sharedColorLimits({R},cfg),cfg);
    ylabel(a2,'Frequency (Hz)'); addColorbar(a2,cfg);
    sharedTimeLabel(f,2,cfg);
    linkaxes([a1 a2],'x'); xlim(a1,R.viewWindow_s);
    savePNG(f,'Figure4_03_combined',cfg);
end
disp(Summary);
fprintf('PNG output folder:\n%s\n',cfg.outDir);

function z=bandFilter(y,fs,band,n)
    z=y;
    % Original SOS, zero-phase filtering, applied before display cropping.
    if band(1)>0
        [zz,p,g]=butter(n,band(1)/(fs/2),'high');
        [sos,g]=zp2sos(zz,p,g); z=filtfilt(sos,g,z);
    end
    if band(2)>0 && band(2)<fs/2
        [zz,p,g]=butter(n,band(2)/(fs/2),'low');
        [sos,g]=zp2sos(zz,p,g); z=filtfilt(sos,g,z);
    end
end
function S=calculateSpectra(t,y,fs,cfg)
    band=[0 min(30,.4*fs)];
    n=min(numel(y),max(16,round(cfg.motionSTFTWindow_s*fs)));
    overlap=min(n-1,round(cfg.STFTOverlap*n));
    nfft=2^nextpow2(n); % zero padding does not improve physical resolution
    [~,f,tt,P]=spectrogram(y,hann(n,'periodic'),overlap,nfft,fs,'psd');
    selected=f>=band(1) & f<=band(2);
    S.time_s=tt+t(1); S.frequency_Hz=f(selected);
    S.psd_dB=10*log10(max(P(selected,:),realmin)); S.band_Hz=band;
    nw=min(numel(y),max(16,round(2*fs)));
    [p,ff]=pwelch(y,hann(nw,'periodic'),floor(nw/2),2^nextpow2(nw),fs,'onesided');
    sel=ff>0 & ff<=band(2);
    S.asdFrequency_Hz=ff(sel); S.asd_counts_sqrtHz=sqrt(p(sel));
    S.asdXLim=[max(ff(2),.1) band(2)];
    if isempty(S.asdFrequency_Hz) || S.asdXLim(1)>=S.asdXLim(2)
        error('Record is too short for an ASD plot in the requested frequency band.');
    end
end
function f=newFigure(cfg,nRows,nCols,name)
    w=cfg.leftMargin_cm+(nCols-1)*cfg.columnPitch_cm+ ...
        cfg.plotWidth_cm+cfg.rightMargin_cm;
    h=cfg.bottomMargin_cm+(nRows-1)*cfg.rowPitch_cm+ ...
        cfg.plotHeight_cm+cfg.topMargin_cm;
    f=figure('Visible','off','WindowStyle','normal', ...
        'Color','w','Units','centimeters','Position',[1 1 w h], ...
        'Name',name,'NumberTitle','off','InvertHardcopy','off', ...
        'PaperUnits','centimeters','PaperSize',[w h], ...
        'PaperPosition',[0 0 w h],'PaperPositionMode','manual');
    setappdata(f,'ExportSize_cm',[w h]);
end
function ax=panelAxes(f,nRows,row,col,cfg)
    x=cfg.leftMargin_cm+(col-1)*cfg.columnPitch_cm;
    y=cfg.bottomMargin_cm+(nRows-row)*cfg.rowPitch_cm;
    sz=getappdata(f,'ExportSize_cm');
    ax=axes('Parent',f,'Units','normalized', ...
        'Position',[x y cfg.plotWidth_cm cfg.plotHeight_cm]./sz([1 2 1 2]));
    if isprop(ax,'PositionConstraint'), ax.PositionConstraint='innerposition';
    else, ax.ActivePositionProperty='position'; end
end
function sharedTimeLabel(f,nCols,cfg)
    sz=getappdata(f,'ExportSize_cm');
    x=cfg.leftMargin_cm/sz(1);
    w=((nCols-1)*cfg.columnPitch_cm+cfg.plotWidth_cm)/sz(1);
    annotation(f,'textbox',[x .25/sz(2) w .65/sz(2)], ...
        'String','Time (s)','EdgeColor','none', ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontName',cfg.fontName,'FontSize',cfg.fontSize);
end
function styleAxes(ax,cfg)
    set(ax,'FontName',cfg.fontName,'FontSize',cfg.fontSize, ...
        'LabelFontSizeMultiplier',1,'TitleFontSizeMultiplier',1, ...
        'TitleFontWeight','normal','Box','off','TickDir','out', ...
        'LineWidth',.8,'YMinorTick','off');
end
function plotWaveform(ax,R,showRaw,cfg)
    hold(ax,'on');
    if showRaw
        plot(ax,R.time_s,R.centered_counts,'Color',[.78 .78 .78],'LineWidth',.7);
    end
    plot(ax,R.time_s,R.motion_counts,'Color',[0 .45 .70],'LineWidth',1.1);
    xlim(ax,R.viewWindow_s); styleAxes(ax,cfg); limitYTicks(ax);
end
function plotSpectrogram(ax,R,clim,cfg)
    imagesc(ax,R.spec.time_s,R.spec.frequency_Hz,R.spec.psd_dB);
    axis(ax,'xy'); colormap(ax,parula); caxis(ax,clim);
    xlim(ax,R.viewWindow_s); ylim(ax,R.spec.band_Hz);
    styleAxes(ax,cfg); limitYTicks(ax);
end
function plotASD(ax,R,cfg)
    loglog(ax,R.spec.asdFrequency_Hz,R.spec.asd_counts_sqrtHz, ...
        'Color',[0 .45 .70],'LineWidth',1);
    xlim(ax,R.spec.asdXLim); styleAxes(ax,cfg); limitYTicks(ax);
end
function lim=sharedColorLimits(R,cfg)
    if ~isempty(cfg.spectrogramCLim_dB)
        lim=cfg.spectrogramCLim_dB; return;
    end
    top=-Inf;
    for k=1:numel(R), top=max(top,max(R{k}.spec.psd_dB(:))); end
    lim=[top-cfg.spectrogramDynamicRange_dB top];
end
function lim=sharedASDLimits(R)
    low=Inf; high=0;
    for k=1:numel(R)
        v=R{k}.spec.asd_counts_sqrtHz;
        v=v(isfinite(v) & v>0);
        if ~isempty(v), low=min(low,min(v)); high=max(high,max(v)); end
    end
    if ~isfinite(low) || high<=0, lim=[1e-3 1]; return; end
    lim=[10^floor(log10(low)) 10^ceil(log10(high))];
    if lim(1)==lim(2), lim=lim.*[.1 10]; end
end
function cb=addColorbar(ax,cfg,placementAx)
    ax.Units='normalized'; p=ax.Position;
    if nargin<3, placementAx=ax; end
    placementAx.Units='normalized'; anchor=placementAx.Position;
    sz=getappdata(ancestor(ax,'figure'),'ExportSize_cm');
    cb=colorbar(ax);
    cb.Units='normalized';
    cb.Position=[anchor(1)+anchor(3)+cfg.colorbarGap_cm/sz(1) anchor(2) ...
        cfg.colorbarWidth_cm/sz(1) anchor(4)];
    ax.Position=p;
    set(cb,'FontName',cfg.fontName,'FontSize',cfg.fontSize);
    ylabel(cb,'PSD (dB re 1 count^2/Hz)', ...
        'FontName',cfg.fontName,'FontSize',cfg.fontSize);
    cb.Ticks=niceLinearTicks(caxis(ax),5);
end
function limitYTicks(ax)
    lim=ylim(ax);
    if strcmp(ax.YScale,'log')
        ticks=10.^(ceil(log10(lim(1))):floor(log10(lim(2))));
        if numel(ticks)>5
            ticks=ticks(unique(round(linspace(1,numel(ticks),5))));
        elseif numel(ticks)<2
            ticks=logspace(log10(lim(1)),log10(lim(2)),3);
        end
    else
        ticks=niceLinearTicks(lim,5);
    end
    set(ax,'YTick',ticks,'YMinorTick','off');
end
function ticks=niceLinearTicks(lim,maxCount)
    raw=diff(lim)/(maxCount-1);
    if ~isfinite(raw) || raw<=0, ticks=lim(1); return; end
    base=10^floor(log10(raw));
    candidates=[1 2 2.5 5 10]*base;
    step=candidates(find(candidates>=raw,1,'first'));
    ticks=(ceil(lim(1)/step):floor(lim(2)/step))*step;
    ticks(abs(ticks)<step*1e-10)=0;
    if numel(ticks)<2, ticks=linspace(lim(1),lim(2),3); end
    if numel(ticks)>maxCount
        ticks=ticks(unique(round(linspace(1,numel(ticks),maxCount))));
    end
end
function savePNG(f,name,cfg)
    sz=getappdata(f,'ExportSize_cm');
    set(f,'Visible','off','Units','centimeters','Position',[1 1 sz], ...
        'PaperUnits','centimeters','PaperSize',sz, ...
        'PaperPosition',[0 0 sz],'PaperPositionMode','manual');
    % Apply the configured Arial font size to every text object.
    set(findall(f,'-property','FontName'),'FontName',cfg.fontName);
    set(findall(f,'-property','FontSize'),'FontSize',cfg.fontSize);
    drawnow;
    print(f,fullfile(cfg.outDir,[name '.png']),'-dpng', ...
        sprintf('-r%d',cfg.resolution));
    if cfg.keepFiguresOpen
        fitPreviewToScreen(f);
        set(f,'Visible','on');
    else
        close(f);
    end
end
function fitPreviewToScreen(f)
    f.Units='pixels';
    oldUnits=get(groot,'Units');
    set(groot,'Units','pixels');
    screen=get(groot,'ScreenSize');
    set(groot,'Units',oldUnits);
    p=f.Position;
    scale=min([1 .90*screen(3)/p(3) .80*screen(4)/p(4)]);
    p(3:4)=p(3:4)*scale;
    p(1:2)=screen(1:2)+(screen(3:4)-p(3:4))/2;
    f.Position=p;
end
