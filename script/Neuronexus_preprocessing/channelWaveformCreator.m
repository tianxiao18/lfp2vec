sessionNames={'AD_HF01_Session1h', 'AD_HF02_230706_Session2', 'AD_HF02_230722_Session4', 'AD_HF03_230725_Session1', 'AD_HF03_230726_Session2', 'NN_syn_20230601', 'NN_syn_20230607'};
rootDir = '/scratch/th3129/shared/Neuronexus_dataset';
outputDir = '/scratch/th3129/region_decoding/data/Neuronexus/waveforms'

for i=1:length(sessionNames)
    sessionName = sessionNames{i};
    spikes = load(fullfile(rootDir, sessionName, [sessionName '.spikes.cellinfo.mat'])).spikes;
    session = load(fullfile(rootDir, sessionName, [sessionName '.session.mat'])).session;
    
    spikesNew = getWaveformsFromDat(spikes, session);

    matFileName = fullfile(outputDir, [sessionName '_waveforms.mat']);
    save(matFileName, 'spikesNew', '-v7.3');
end