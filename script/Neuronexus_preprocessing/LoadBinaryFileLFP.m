% sessions={'AD_HF01_Session1h', 'AD_HF02_230706_Session2', 'AD_HF02_230722_Session4', 'AD_HF03_230725_Session1', 'AD_HF03_230726_Session2', 'NN_syn_20230601', 'NN_syn_20230607'};
% sessionNames={'AD_HF01_1_lfp', 'AD_HF02_2_lfp', 'AD_HF02_4_lfp', 'AD_HF03_1_lfp', 'AD_HF03_2_lfp', 'NN_syn_01_lfp', 'NN_syn_02_lfp'};
sessions={'AD_HF02_230722_Session4', 'AD_HF03_230725_Session1'}
sessionNames={'AD_HF02_4_lfp', 'AD_HF03_1_lfp'}

for i=1:length(sessions)
    session = sessions{i};
    sessionPath = fullfile('/scratch/th3129/shared/Neuronexus_dataset', session, [session '.lfp']);

    data = LoadBinary(sessionPath, 'frequency', 1250, 'nChannels', 1024, 'start', 2160, 'duration', 300);
    matFileName = [sessionNames{i} '.mat'];
    save(matFileName, 'data', '-v7.3');
end