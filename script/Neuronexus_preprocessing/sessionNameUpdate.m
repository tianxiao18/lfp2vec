% List of session names
sessionNames = {'AD_HF01_Session1h', 'AD_HF02_230706_Session2', 'AD_HF02_230722_Session4', ...
                'AD_HF03_230725_Session1', 'AD_HF03_230726_Session2', 'NN_syn_20230601', 'NN_syn_20230607'};

% Root directory where session files are located
inputDir = '/scratch/th3129/shared/Neuronexus_dataset';

% Generate new base paths dynamically
newBasePaths = fullfile(inputDir, sessionNames);

% Loop through each session
for i = 1:length(sessionNames)
    sessionName = sessionNames{i};
    sessionFilePath = fullfile(inputDir, sessionName, [sessionName '.session.mat']);
    
    % Load the session file
    if isfile(sessionFilePath)
        sessionData = load(sessionFilePath);
        if isfield(sessionData, 'session') && isfield(sessionData.session, 'general') && ...
                isfield(sessionData.session.general, 'basePath')
            
            % Modify the basePath
            sessionData.session.general.basePath = newBasePaths{i};
            
            % Save the modified session back to the same file
            save(sessionFilePath, '-struct', 'sessionData', '-v7.3');
            
            fprintf('Updated basePath for session: %s\n', sessionName);
        else
            fprintf('Skipping %s: Required fields are missing.\n', sessionName);
        end
    else
        fprintf('Session file not found: %s\n', sessionFilePath);
    end
end