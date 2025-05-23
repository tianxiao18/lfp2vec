# region_decoding

In electrophysiology, precise in vivo localization of recording sites in deep brain structures is crucial for consistent targeting in multi-day recordings and accurate deep brain stimulation. However, current approaches present their own challenges: brain atlas-guided probe insertion may be imprecise due to anatomical variability, CT or MRI scan based localization may lack the spatial resolution for subregion structures, and post hoc histology misses longitudinal information in chronic recordings. 

Focusing on mouse recordings, we present a learning-based automatic localizer to identify hippocampus sublayers from high-density extracellular recordings. We demonstrate our method on the following dataset

1. Buzsaki Lab Neuronexus recordings with 1024 channels (by Anna and Misi)

2. Allen Brain Institute Neuropixel Recordings

3. International Brain Lab Neuropixel recordings

The detailed overleaf document can be found here: [Blind Localization](https://www.overleaf.com/project/64fa265a8351c9f6ab9e88c7)



### IBL dataset overview
* Sampling rate: 2500Hz

* Dataset Name: Brainwide map

* Labels: CA1,CA2,CA3,DG,VIS (0,1,2,3,4)

#### Data
We select 19 sessions in IBL dataset that contains CA1,CA2,CA3 regions. To train and test models, we chunk time series data into chunks with 3s/chunk, which now we call it 'trial'. There are maximum 9000 trials in each session. 

* `script/ibl_preprocessing/CA1CA2CA3_eid.npy`: eids of 19 sessions that contains CA123

* `script/ibl_preprocessing/CA1CA2CA3DG_eids.npy`: eids of sessions that contains CA123 and DG

* `script/ibl_preprocessing/selected_eids.npy`: 7 sessions for experiments

* `ibl` contains all ibl data
    * `ibl_insertion.csv`: insertion probe 
    * `{eid}_fs.npy`: sampling rate of raw Ephys data
    * `{eid}_label_lfp.npy`: labels of raw Ephys data (channel x timepoints) 
    * `{eid}_lfp.npy`: raw Ephys data (channel x timepoints)
    * `raw` contains preprocessed LFP raw features
        * trial x (feature dimension + 1 label index)
    * `spectrogram` contains preprocessed spectrogram features
        * `{eid}_data.pickle`: spectrogram feature data (trial x (feature dimension + 1 label index) )
        * `{eid}_raw.pickle`: raw Ephys data (trial x (timepoints + 1 label index))

#### Script
* `downloader.py`: get spike or raw ephy data from ibl server
* `raw_data_creator.py`: create lfp feature from raw ephys data
* `spectrogram_creator.py`: create spectrogram feature from raw ephys data