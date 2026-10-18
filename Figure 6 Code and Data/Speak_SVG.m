%% Speech spectrograms: explicit quiet reference and soft spectral suppression
% MATLAB R2020b+; Signal Processing Toolbox (butter, filtfilt, hann).
% Place this script beside the original IM_*.csv files, or edit cfg.dataDir.
% Run the entire script. Original CSV files are read-only.
%
% These are exploratory VISUALIZATION transforms, not validated waveform
% denoising. No audio reconstruction, speech recognition, automatic event
% count, ridge tracking, or denoised SNR/sensitivity calculation is performed.
% "Original" PSD below means uniformly resampled, high-pass-filtered data.
% Approximate intervals were reviewed visually; they are NOT audio labels.
% Their only numerical use is excluding possible events from the quiet
% reference. Suppression is applied continuously to all retained STFT bins.
%
% Default soft thresholds: 6 dB (moderate) and 9 dB (stronger).
% Higher thresholds can erase weak speech components as well as background.
% Both use exactly the same reference, STFT, display range, and colormap.
clear; clc; close all;
%% Configuration
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
cfg.dataDir = scriptDir;                   % EDIT if CSV files are elsewhere
cfg.outDir = fullfile(scriptDir,'speech_denoise_results_matlab');
cfg.recordIDs = [79 83 87 93];
cfg.combinedIDs = [79 83 87 93];
cfg.highpass_Hz = 10;
cfg.highpassOrder = 2;
cfg.waveformBand_Hz = [100 200];           % separate waveform view, as in Speak_SVG.m
cfg.waveformOrder = 2;                    % each HP/LP stage; forward-backward filtering
cfg.window_s = 0.160;
cfg.hop_s = 0.010;
cfg.powerSmoothFrames = 5;                 % symmetric, shrinking endpoints
cfg.eventPadding_s = 0.120;
cfg.edgeExclusion_s = 0.200;
cfg.softThresholds_dB = [6 9];             % edit these two values if desired
cfg.softWidth_dB = 1.5;
cfg.amplitudeGainFloor = 0.03;
cfg.gainSmoothFreqBins = 3;
cfg.gainSmoothTimeFrames = 3;
cfg.displayBand_Hz = [10 400];
cfg.powerColorLimits_dB = [15 48];         % shared by ORIGINAL and processed
cfg.relativeColorLimits_dB = [0 18];
cfg.yTicks_Hz = [10 100 200 300 400];       % at most five ticks
cfg.fontName = 'Arial';
cfg.fontSize = 27;
cfg.exportDPI = 200;
cfg.colormapName = 'parula';
cfg.lineColor = [0 .45 .70];        % Figure 4 filtered waveform / ASD blue
cfg.rawColor = [.78 .78 .78];       % Figure 4 raw waveform gray
cfg.methodVersion = 'explicit_quiet_mean_softgate_v1';
cfg.scope = ['Exploratory PSD visualization; no reconstructed waveform; ' ...
    'not validated denoising and not a quantitative SNR estimator.'];
cfg.eventIntervalStatus = ['Approximate visually inferred activity intervals; ' ...
    'not verified acoustic or audio labels. Used only to exclude noise-reference frames.'];
records = makeRecords();
assert(numel(cfg.softThresholds_dB)==2 && all(isfinite(cfg.softThresholds_dB)), ...
    'Provide exactly two finite cfg.softThresholds_dB values.');
assert(cfg.amplitudeGainFloor>0 && cfg.amplitudeGainFloor<1);
assert(cfg.softWidth_dB>0 && numel(cfg.yTicks_Hz)<=5);
assert(mod(cfg.powerSmoothFrames,2)==1 && cfg.powerSmoothFrames>=1);
assert(mod(cfg.gainSmoothFreqBins,2)==1 && cfg.gainSmoothFreqBins>=1);
assert(mod(cfg.gainSmoothTimeFrames,2)==1 && cfg.gainSmoothTimeFrames>=1);
assert(all(ismember(cfg.recordIDs,[records.id])),'Unknown record ID.');
assert(all(ismember(cfg.combinedIDs,cfg.recordIDs)), ...
    'Each combined ID must also occur in cfg.recordIDs.');
if ~isfolder(cfg.outDir), mkdir(cfg.outDir); end
set(groot,'defaultAxesFontName',cfg.fontName, ...
    'defaultAxesFontSize',cfg.fontSize, ...
    'defaultTextFontName',cfg.fontName,'defaultTextFontSize',cfg.fontSize);
warning('SpeechVisualization:Labels', ...
    ['IM_89/90 have contradictory ordinary/noisy condition labels; they are excluded. ' ...
     'IM_80/81 have unconfirmed task labels and are also excluded.']);
provenance = struct;
provenance.createdUTC = char(datetime('now','TimeZone','UTC', ...
    'Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
provenance.MATLAB = version;
provenance.toolboxes = ver;
provenance.script = [mfilename('fullpath') '.m'];
provenance.settings = cfg;
provenance.records = records;
writeJSON(fullfile(cfg.outDir,'settings_and_provenance.json'),provenance);
if isfile(provenance.script)
    copyfile(provenance.script,fullfile(cfg.outDir,'analysis_script_used.m'));
end
%% Process records separately; reference is estimated within each recording
results = cell(1,numel(cfg.recordIDs));
for k = 1:numel(cfg.recordIDs)
    record = records([records.id]==cfg.recordIDs(k));
    results{k} = processRecord(record,cfg,provenance);
    plotRecord(results{k},cfg);
end
% Each combined figure uses a single color scale and a single common colorbar.
for j = 1:numel(cfg.softThresholds_dB)
    plotCombined(results,cfg,j);
end
fprintf('\nFinished. PNG figures and reproducibility files: %s\n',cfg.outDir);
fprintf(['Compare original and processed PSDs together. A cleaner background ' ...
    'does not establish that all retained features are speech.\n']);
%% Local functions
function records = makeRecords()
    empty = struct('id',[],'label','','events_s',zeros(0,2), ...
        'artifactExclusions_s',zeros(0,2));
    records = repmat(empty,1,7);
    records(1) = struct('id',79,'label','Hello (current reference recording)', ...
        'events_s',[1.20 2.15;2.72 3.50;4.08 4.80], ...
        'artifactExclusions_s',zeros(0,2));
    records(2) = struct('id',83,'label','Hello, lying down', ...
        'events_s',[1.95 2.90;3.58 4.50;4.95 5.98], ...
        'artifactExclusions_s',[6.10 7.30]);
    records(3) = struct('id',85,'label','Hello, lying down (alternative)', ...
        'events_s',[1.25 2.15;2.90 3.75;4.35 5.18], ...
        'artifactExclusions_s',zeros(0,2));
    records(4) = struct('id',87,'label','Hello, reported 83 dB noise', ...
        'events_s',[2.70 3.65;4.30 5.20;5.85 6.85], ...
        'artifactExclusions_s',zeros(0,2));
    records(5) = struct('id',93,'label','Bye', ...
        'events_s',[1.30 1.95;2.62 3.20;4.00 4.60], ...
        'artifactExclusions_s',zeros(0,2));
    records(6) = struct('id',97,'label','Thanks (optional)', ...
        'events_s',[2.25 3.00;3.55 4.25;5.10 5.85], ...
        'artifactExclusions_s',zeros(0,2));
    records(7) = struct('id',100,'label','Thanks (alternative)', ...
        'events_s',[2.20 3.00;3.35 4.10;4.85 5.65], ...
        'artifactExclusions_s',zeros(0,2));
end
function result = processRecord(record,cfg,provenance)
    stem = sprintf('IM_%d',record.id);
    sourceFile = fullfile(cfg.dataDir,[stem '.csv']);
    assert(isfile(sourceFile), ...
        'Cannot find %s. Edit cfg.dataDir to the folder containing CSV files.',sourceFile);
    T = readtable(sourceFile);
    assert(all(ismember({'Time_us','Data28'},T.Properties.VariableNames)), ...
        '%s must contain columns Time_us and Data28.',sourceFile);
    tMicroseconds = double(T.Time_us);
    xRaw = double(T.Data28);
    assert(isvector(tMicroseconds) && isvector(xRaw) && numel(xRaw)>=500, ...
        '%s needs at least 500 samples.',sourceFile);
    tMicroseconds = tMicroseconds(:); xRaw = xRaw(:);
    assert(all(isfinite(tMicroseconds)) && all(isfinite(xRaw)), ...
        '%s contains nonfinite samples. No rows are silently removed.',sourceFile);
    assert(all(diff(tMicroseconds)>0), ...
        '%s timestamps must be strictly increasing; inspect duplicates/rollover.',sourceFile);
    if any(abs(tMicroseconds-round(tMicroseconds))>1e-6)
        warning('SpeechVisualization:TimestampUnits', ...
            '%s Time_us contains noninteger entries; verify microsecond units.',stem);
    end
    tRaw = (tMicroseconds-tMicroseconds(1))*1e-6;
    dt = diff(tRaw); duration_s = tRaw(end);
    nSamples = numel(xRaw);
    Fs = (nSamples-1)/duration_s;
    dt_us = diff(tMicroseconds); % compare in input units to avoid roundoff at threshold
    gapCount = nnz(dt_us>2*median(dt_us));
    if gapCount>0
        warning('SpeechVisualization:TimestampGaps', ...
            '%s has %d timestamp gaps >2 x median; interpolation cannot recover missing data.', ...
            stem,gapCount);
    end
    assert(cfg.highpass_Hz>0 && cfg.highpass_Hz<Fs/2);
    assert(cfg.displayBand_Hz(2)<Fs/2,'Displayed band exceeds Nyquist frequency.');
    assert(all(record.events_s(:,1)>=0 & record.events_s(:,2)<=duration_s), ...
        '%s contains reviewed event intervals outside this recording.',stem);
    tUniform = linspace(0,duration_s,nSamples)';
    xUniform = interp1(tRaw,xRaw-median(xRaw),tUniform,'linear');
    [b,a] = butter(cfg.highpassOrder,cfg.highpass_Hz/(Fs/2),'high');
    xHighpass = filtfilt(b,a,xUniform);
    % Match the original Speak_SVG.m twoStageBandpass exactly: two separate
    % Butterworth stages, each applied forward and backward. This waveform
    % is independent of the unchanged 10 Hz high-pass input to the STFT.
    assert(numel(cfg.waveformBand_Hz)==2 && ...
        cfg.waveformBand_Hz(1)>0 && ...
        cfg.waveformBand_Hz(2)>cfg.waveformBand_Hz(1) && ...
        cfg.waveformBand_Hz(2)<Fs/2,'Invalid waveform display band.');
    [bWaveHigh,aWaveHigh] = butter(cfg.waveformOrder, ...
        cfg.waveformBand_Hz(1)/(Fs/2),'high');
    [bWaveLow,aWaveLow] = butter(cfg.waveformOrder, ...
        cfg.waveformBand_Hz(2)/(Fs/2),'low');
    xWaveformBand = filtfilt(bWaveLow,aWaveLow, ...
        filtfilt(bWaveHigh,aWaveHigh,xUniform));
    windowSamples = max(4,round(cfg.window_s*Fs));
    hopSamples = max(1,round(cfg.hop_s*Fs));
    assert(windowSamples<=nSamples,'STFT window exceeds recording length.');
    nfft = 2^nextpow2(windowSamples);
    [originalPSD,f_Hz,tSTFT_s,frameStarts] = manualSTFT( ...
        xHighpass,Fs,windowSamples,hopSamples,nfft);
    smoothedPSD = movmean(originalPSD,cfg.powerSmoothFrames,2,'Endpoints','shrink');
    paddedEvents = record.events_s+[-cfg.eventPadding_s cfg.eventPadding_s];
    exclusions = [0 cfg.edgeExclusion_s; ...
        duration_s-cfg.edgeExclusion_s duration_s; ...
        paddedEvents; record.artifactExclusions_s];
    [quietIntervals_s,mergedExclusions_s] = intervalComplement(exclusions,duration_s);
    % Every source sample contributing to the temporally smoothed PSD must be
    % inside one quiet interval. This excludes the full Hann window plus the
    % additional +/-2 hops contributed by the default five-frame average.
    smoothRadius = (cfg.powerSmoothFrames-1)/2;
    refSupportHalfWidth_s = ((windowSamples-1)/2+smoothRadius*hopSamples)/Fs;
    referenceMask = false(size(tSTFT_s));
    for q = 1:size(quietIntervals_s,1)
        referenceMask = referenceMask | ...
            (tSTFT_s-refSupportHalfWidth_s>=quietIntervals_s(q,1) & ...
             tSTFT_s+refSupportHalfWidth_s<=quietIntervals_s(q,2));
    end
    assert(nnz(referenceMask)>=20, ...
        '%s has fewer than 20 full-support reference frames. Review quiet intervals.',stem);
    noiseReferencePSD = mean(smoothedPSD(:,referenceMask),2);
    noiseReferencePSD = max(noiseReferencePSD,realmin('double'));
    safeSmoothedPSD = max(smoothedPSD,realmin('double'));
    backgroundRelative_dB = 10*log10(safeSmoothedPSD./noiseReferencePSD);
    % Dimensionless power ratio in dB. This is NOT a separately measured SNR.
    wienerAmplitudeGain = max(cfg.amplitudeGainFloor, ...
        1-noiseReferencePSD./safeSmoothedPSD);
    wienerPSD = originalPSD.*wienerAmplitudeGain.^2;
    softAmplitudeGains = zeros([size(originalPSD) numel(cfg.softThresholds_dB)]);
    softPSD = zeros(size(softAmplitudeGains));
    for j = 1:numel(cfg.softThresholds_dB)
        z = (backgroundRelative_dB-cfg.softThresholds_dB(j))/cfg.softWidth_dB;
        G = cfg.amplitudeGainFloor+(1-cfg.amplitudeGainFloor)./(1+exp(-z));
        G = movmean(G,cfg.gainSmoothFreqBins,1,'Endpoints','shrink');
        G = movmean(G,cfg.gainSmoothTimeFrames,2,'Endpoints','shrink');
        softAmplitudeGains(:,:,j) = G;
        softPSD(:,:,j) = originalPSD.*G.^2;
    end
    diagnostics = struct('Fs_Hz',Fs,'medianInterval_s',median(dt), ...
        'medianBasedFs_Hz',1/median(dt),'sampleCount',nSamples, ...
        'duration_s',duration_s,'gapsAboveTwiceMedian',gapCount, ...
        'maximumInterval_s',max(dt),'windowSamples',windowSamples, ...
        'actualWindow_s',windowSamples/Fs,'hopSamples',hopSamples, ...
        'actualHop_s',hopSamples/Fs,'nfft',nfft, ...
        'referenceFrameCount',nnz(referenceMask), ...
        'referenceSupportHalfWidth_s',refSupportHalfWidth_s);
    sourceInfo = dir(sourceFile);
    result = struct('id',record.id,'label',record.label,'stem',stem, ...
        'f_Hz',f_Hz,'tSTFT_s',tSTFT_s,'originalPSD',originalPSD, ...
        'tRaw',tRaw,'xRaw',xRaw,'tUniform',tUniform,'xHighpass',xHighpass, ...
        'xWaveformBand',xWaveformBand, ...
        'wienerPSD',wienerPSD,'softPSD',softPSD, ...
        'backgroundRelative_dB',backgroundRelative_dB, ...
        'referenceMask',referenceMask,'diagnostics',diagnostics);
    out = fullfile(cfg.outDir,stem);
    if ~isfolder(out), mkdir(out); end
    % All original and processed numerical arrays remain separately available.
    save(fullfile(out,[stem '_spectral_arrays.mat']), ...
        'originalPSD','smoothedPSD','noiseReferencePSD','wienerPSD', ...
        'wienerAmplitudeGain','softPSD','softAmplitudeGains', ...
        'backgroundRelative_dB','f_Hz','tSTFT_s','referenceMask', ...
        'quietIntervals_s','mergedExclusions_s','record','cfg','diagnostics', ...
        'sourceFile','sourceInfo','provenance','tRaw','xRaw','tUniform', ...
        'xUniform','xHighpass','xWaveformBand','b','a', ...
        'bWaveHigh','aWaveHigh','bWaveLow','aWaveLow','frameStarts','-v7.3');
    writetable(table(f_Hz,noiseReferencePSD,'VariableNames', ...
        {'Frequency_Hz','MeanQuietSmoothedPSD_counts2_per_Hz'}), ...
        fullfile(out,'background_reference.csv'));
    writetable(array2table(quietIntervals_s,'VariableNames',{'Start_s','End_s'}), ...
        fullfile(out,'quiet_reference_intervals.csv'));
    writetable(array2table(mergedExclusions_s,'VariableNames',{'Start_s','End_s'}), ...
        fullfile(out,'reference_excluded_intervals.csv'));
    writetable(array2table(record.events_s,'VariableNames',{'Start_s','End_s'}), ...
        fullfile(out,'visually_reviewed_activity_intervals.csv'));
    referenceIndices = find(referenceMask(:));
    referenceTimes = tSTFT_s(referenceMask); referenceTimes = referenceTimes(:);
    writetable(table(referenceIndices,referenceTimes, ...
        referenceTimes-refSupportHalfWidth_s,referenceTimes+refSupportHalfWidth_s, ...
        'VariableNames',{'FrameIndex_1based','Center_s', ...
        'FullSmoothedSupportStart_s','FullSmoothedSupportEnd_s'}), ...
        fullfile(out,'actual_reference_frame_centers.csv'));
    writeJSON(fullfile(out,'record_settings.json'),struct('record',record, ...
        'config',cfg,'diagnostics',diagnostics,'source',sourceInfo));
    fprintf('%s: Fs %.3f Hz, %d reference frames, %.3f s record.\n', ...
        stem,Fs,nnz(referenceMask),duration_s);
end
function [P,f,t,starts] = manualSTFT(x,Fs,L,hop,nfft)
    % Zero-based starts, no endpoint padding. Symmetric Hann, not periodic.
    starts = 0:hop:(numel(x)-L);
    idx = (1:L)'+starts;
    w = hann(L,'symmetric');
    Z = fft(x(idx).*w,nfft,1);
    Z = Z(1:nfft/2+1,:);
    P = abs(Z).^2/(Fs*sum(w.^2));
    P(2:end-1,:) = 2*P(2:end-1,:);
    f = (0:nfft/2)'*Fs/nfft;
    t = (starts+(L-1)/2)/Fs;
end
function [quiet,merged] = intervalComplement(exclusions,duration_s)
    exclusions(:,1) = max(0,min(duration_s,exclusions(:,1)));
    exclusions(:,2) = max(0,min(duration_s,exclusions(:,2)));
    exclusions = exclusions(exclusions(:,2)>exclusions(:,1),:);
    exclusions = sortrows(exclusions,1);
    merged = zeros(0,2);
    for k = 1:size(exclusions,1)
        if isempty(merged) || exclusions(k,1)>merged(end,2)
            merged(end+1,:) = exclusions(k,:); %#ok<AGROW>
        else
            merged(end,2) = max(merged(end,2),exclusions(k,2));
        end
    end
    quiet = zeros(0,2); cursor = 0;
    for k = 1:size(merged,1)
        if merged(k,1)>cursor
            quiet(end+1,:) = [cursor merged(k,1)]; %#ok<AGROW>
        end
        cursor = max(cursor,merged(k,2));
    end
    if cursor<duration_s
        quiet(end+1,:) = [cursor duration_s];
    end
end
function plotRecord(r,cfg)
    out = fullfile(cfg.outDir,r.stem);
    % Separate vertical scales retain the full raw trace without compressing
    % the 100-200 Hz waveform. No arbitrary amplitude multiplier is applied.
    fig = waveformAndSpectrumFigure(r,cfg,true);
    exportgraphics(fig,fullfile(out,[r.stem '_waveform_spectrogram.png']), ...
        'Resolution',cfg.exportDPI); close(fig);
    fig = waveformAndSpectrumFigure(r,cfg,false);
    exportgraphics(fig,fullfile(out,[r.stem '_bandpass_spectrogram.png']), ...
        'Resolution',cfg.exportDPI); close(fig);
    fig = figure('Color','w','Position',[100 100 1250 1000],'Visible','off');
    tl = tiledlayout(fig,3,1,'TileSpacing','compact','Padding','compact');
    names = {sprintf('Before suppression (%.1f Hz high-pass)',cfg.highpass_Hz), ...
        sprintf('Soft suppression, threshold %.1f dB',cfg.softThresholds_dB(1)), ...
        sprintf('Soft suppression, threshold %.1f dB',cfg.softThresholds_dB(2))};
    for k = 1:3
        ax = nexttile(tl);
        if k==1, P = r.originalPSD; else, P = r.softPSD(:,:,k-1); end
        drawPSD(ax,r,P,cfg);
        title(ax,names{k},'FontWeight','normal');
        if k<3, ax.XTickLabel = []; end
    end
    title(tl,sprintf('%s | %s',r.stem,r.label),'Interpreter','none','FontSize',cfg.fontSize);
    ylabel(tl,'Frequency (Hz)','FontName',cfg.fontName,'FontSize',cfg.fontSize);
    xlabel(tl,'Time (s)','FontName',cfg.fontName,'FontSize',cfg.fontSize);
    cb = colorbar(ax); cb.Layout.Tile = 'east';
    % PSD reference remains 1 count^2/Hz; shortened label for the figure.
    cb.Label.String = 'PSD (dB)';
    styleColorbar(cb,cfg,cfg.powerColorLimits_dB);
    exportgraphics(fig,fullfile(out,[r.stem '_original_soft_comparison.png']), ...
        'Resolution',cfg.exportDPI); close(fig);
    fig = figure('Color','w','Position',[100 100 1250 650],'Visible','off');
    tl = tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');
    ax = nexttile(tl); drawPSD(ax,r,r.wienerPSD,cfg);
    title(ax,'Mean quiet-reference Wiener-style suppression','FontWeight','normal');
    ax.XTickLabel = []; cb = colorbar(ax);
    % PSD reference remains 1 count^2/Hz; shortened label for the figure.
    cb.Label.String = 'PSD (dB)';
    styleColorbar(cb,cfg,cfg.powerColorLimits_dB);
    ax = nexttile(tl);
    keep = r.f_Hz>=cfg.displayBand_Hz(1) & r.f_Hz<=cfg.displayBand_Hz(2);
    imagesc(ax,r.tSTFT_s,r.f_Hz(keep),r.backgroundRelative_dB(keep,:));
    styleAxes(ax,r,cfg); caxis(ax,cfg.relativeColorLimits_dB);
    title(ax,'Smoothed power / mean quiet-reference power (not SNR)', ...
        'FontWeight','normal');
    cb = colorbar(ax); cb.Label.String = '10 log_{10}(S/N), dimensionless dB ratio';
    styleColorbar(cb,cfg,cfg.relativeColorLimits_dB);
    ylabel(tl,'Frequency (Hz)','FontName',cfg.fontName,'FontSize',cfg.fontSize);
    xlabel(tl,'Time (s)','FontName',cfg.fontName,'FontSize',cfg.fontSize);
    title(tl,sprintf('%s | Reference diagnostics',r.stem),'Interpreter','none', ...
        'FontSize',cfg.fontSize);
    exportgraphics(fig,fullfile(out,[r.stem '_reference_diagnostics.png']), ...
        'Resolution',cfg.exportDPI); close(fig);
end
function fig = waveformAndSpectrumFigure(r,cfg,includeRaw)
    % Fixed axes positions ensure identical plot widths. The colorbar belongs
    % only to the spectrum. Independent Y scales remain in physical counts.
    if includeRaw
        figureHeight_cm = 20;
        rawPosition = [.12 .755 .67 .135];
        bandPosition = [.12 .455 .67 .205];
        spectrumPosition = [.12 .130 .67 .230];
    else
        figureHeight_cm = 15;
        bandPosition = [.12 .580 .67 .290];
        spectrumPosition = [.12 .170 .67 .290];
    end
    fig = figure('Color','w','Units','centimeters', ...
        'Position',[2 2 28 figureHeight_cm],'Visible','off');
    waveformAxes = gobjects(0);
    if includeRaw
        ar = axes('Parent',fig,'Units','normalized','Position',rawPosition);
        plot(ar,r.tRaw,r.xRaw-median(r.xRaw), ...
            'Color',cfg.rawColor,'LineWidth',.7);
        styleWaveformAxes(ar,r,cfg);
        ylabel(ar,'Counts');
        title(ar,'Raw signal (median-centered)','FontWeight','normal');
        ar.XTickLabel = [];
        waveformAxes(end+1) = ar;
    end
    aw = axes('Parent',fig,'Units','normalized','Position',bandPosition);
    plot(aw,r.tUniform,r.xWaveformBand,'Color',cfg.lineColor,'LineWidth',.7);
    styleWaveformAxes(aw,r,cfg);
    ylabel(aw,'Counts');
    title(aw,sprintf('%g-%g Hz bandpass waveform',cfg.waveformBand_Hz), ...
        'FontWeight','normal');
    aw.XTickLabel = [];
    waveformAxes(end+1) = aw;
    asp = axes('Parent',fig,'Units','normalized','Position',spectrumPosition);
    drawPSD(asp,r,r.softPSD(:,:,2),cfg);
    xlabel(asp,'Time (s)'); ylabel(asp,'Frequency (Hz)');
    title(asp,sprintf('Spectral Threshold %g dB', ...
        cfg.softThresholds_dB(2)),'FontWeight','normal');
    cb = colorbar(asp);
    % PSD reference remains 1 count^2/Hz; shortened label for the figure.
    cb.Label.String = 'PSD (dB)';
    styleColorbar(cb,cfg,cfg.powerColorLimits_dB);
    cb.Units = 'normalized';
    cb.Position = [.820 spectrumPosition(2) .017 spectrumPosition(4)];
    asp.Position = spectrumPosition;
    annotation(fig,'textbox',[.12 .945 .77 .04], ...
        'String',sprintf('%s | %s',r.stem,r.label),'Interpreter','none', ...
        'EdgeColor','none','HorizontalAlignment','center', ...
        'VerticalAlignment','middle','FontName',cfg.fontName,'FontSize',cfg.fontSize);
    drawnow;
    % Reapply limits after colorbar/layout creation, immediately before export.
    % No linkaxes call is needed for a static export with equal time limits.
    for k = 1:numel(waveformAxes)
        ax = waveformAxes(k);
        xlim(ax,[0 r.diagnostics.duration_s]);
        ticks = ax.YTick;
        if numel(ticks)>5
            ax.YTick = ticks(unique(round(linspace(1,numel(ticks),5))));
        end
        ax.YLimMode = 'manual';
        ax.XTick = asp.XTick;
        ax.XTickLabel = [];
    end
    xlim(asp,[0 r.diagnostics.duration_s]);
    ylim(asp,cfg.displayBand_Hz);
    asp.YLimMode = 'manual'; asp.YDir = 'normal';
    yticks(asp,cfg.yTicks_Hz);

    % Measure labels at the selected font size before exporting. Fixed 12%
    % margins can clip 'Counts' when cfg.fontSize is increased (e.g. to 25).
    % Use one shared left margin and right edge for every plot in this figure.
    drawnow;
    allPlotAxes = [waveformAxes(:); asp];
    labelInsets = zeros(numel(allPlotAxes),4);
    for k = 1:numel(allPlotAxes)
        allPlotAxes(k).Units = 'normalized';
        allPlotAxes(k).PositionConstraint = 'innerposition';
        labelInsets(k,:) = allPlotAxes(k).TightInset;
    end
    sharedLeft = max(0.16,max(labelInsets(:,1))+0.025);
    sharedRight = spectrumPosition(1)+spectrumPosition(3);
    assert(sharedRight-sharedLeft>0.25, ...
        'Labels need more room: increase the figure width or reduce cfg.fontSize.');
    for k = 1:numel(allPlotAxes)
        pos = allPlotAxes(k).Position;
        pos(1) = sharedLeft;
        pos(3) = sharedRight-sharedLeft;
        allPlotAxes(k).Position = pos;
    end
    drawnow;
    assert(isequal(asp.YLim,cfg.displayBand_Hz), ...
        'Spectrogram export frequency limits were not retained.');
end
function styleWaveformAxes(ax,r,cfg)
    xlim(ax,[0 r.diagnostics.duration_s]);
    set(ax,'FontName',cfg.fontName,'FontSize',cfg.fontSize,'Box','off', ...
        'TickDir','out','LabelFontSizeMultiplier',1,'TitleFontSizeMultiplier',1);
end
function plotCombined(results,cfg,j)
    fig = figure('Color','w','Position',[100 100 1300 1100],'Visible','off');
    tl = tiledlayout(fig,numel(cfg.combinedIDs),1, ...
        'TileSpacing','compact','Padding','compact');
    for k = 1:numel(cfg.combinedIDs)
        resultIDs = cellfun(@(r) r.id,results);
        r = results{find(resultIDs==cfg.combinedIDs(k),1)};
        ax = nexttile(tl); drawPSD(ax,r,r.softPSD(:,:,j),cfg);
        title(ax,sprintf('%s | %s',r.stem,r.label), ...
            'Interpreter','none','FontWeight','normal');
        % Show each row's own time labels: recordings have different lengths.
    end
    ylabel(tl,'Frequency (Hz)','FontName',cfg.fontName,'FontSize',cfg.fontSize);
    xlabel(tl,'Time (s)','FontName',cfg.fontName,'FontSize',cfg.fontSize);
    title(tl,sprintf('Spectral Threshold %.1f dB', ...
        cfg.softThresholds_dB(j)),'FontSize',cfg.fontSize);
    cb = colorbar(ax); cb.Layout.Tile = 'east';
    % PSD reference remains 1 count^2/Hz; shortened label for the figure.
    cb.Label.String = 'PSD (dB)';
    styleColorbar(cb,cfg,cfg.powerColorLimits_dB);
    thresholdToken = strrep(sprintf('%g',cfg.softThresholds_dB(j)),'.','p');
    filename = sprintf('selected_four_soft_%sdB.png',thresholdToken);
    exportgraphics(fig,fullfile(cfg.outDir,filename),'Resolution',cfg.exportDPI);
    close(fig);
end
function drawPSD(ax,r,P,cfg)
    keep = r.f_Hz>=cfg.displayBand_Hz(1) & r.f_Hz<=cfg.displayBand_Hz(2);
    assert(nnz(keep)>=2,'Fewer than two bins in the displayed frequency band.');
    imagesc(ax,r.tSTFT_s,r.f_Hz(keep), ...
        10*log10(max(P(keep,:),realmin('double'))));
    styleAxes(ax,r,cfg); caxis(ax,cfg.powerColorLimits_dB);
end
function styleAxes(ax,r,cfg)
    axis(ax,'xy'); xlim(ax,[0 r.diagnostics.duration_s]);
    ylim(ax,cfg.displayBand_Hz); yticks(ax,cfg.yTicks_Hz);
    set(ax,'FontName',cfg.fontName,'FontSize',cfg.fontSize, ...
        'TickDir','out','Box','off','Layer','top', ...
        'TitleFontSizeMultiplier',1,'LabelFontSizeMultiplier',1);
    colormap(ax,parula(256));
end
function styleColorbar(cb,cfg,limits)
    cb.FontName = cfg.fontName; cb.FontSize = cfg.fontSize;
    cb.Label.FontName = cfg.fontName; cb.Label.FontSize = cfg.fontSize;
    cb.Ticks = linspace(limits(1),limits(2),4);
end
function writeJSON(filename,value)
    fid = fopen(filename,'w');
    assert(fid~=-1,'Cannot write %s.',filename);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s\n',jsonencode(value));
end
