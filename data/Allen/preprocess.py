import os
import warnings
import numpy as np
from tqdm import tqdm
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
warnings.filterwarnings("ignore")

# Set up paths and acronyms
data_path = "/scratch/mkp6112/LFP/region_decoding/data/Allen/data"
assert os.path.exists(data_path)
manifest_path = os.path.join(data_path, "manifest.json")

# Initialize the data cache
data_cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
sessions_list = data_cache.get_session_table()

# Filter sessions by specific IDs
session_ids = [719161530, 794812542, 778998620, 798911424, 771990200, 771160300, 768515987]
mask = sessions_list.index.isin(session_ids)
filtered_sessions = sessions_list[mask]

# Process each session
for session_id, row in filtered_sessions.iterrows():
    session = data_cache.get_session_data(session_id)
    desc_prefix = f"Session {session_id}"
    pbar = tqdm(session.mean_waveforms.items(), desc=desc_prefix)

    channel_sets = []  # Store channel sets for each unit
    unit_waveforms = []  # Store waveforms for each unit

    # Iterate over units and collect waveforms and channel IDs
    for unit_id, waveforms in pbar:
        channel_ids = set(waveforms.channel_id.values)  # Extract channel IDs
        channel_sets.append(channel_ids)
        unit_waveforms.append(waveforms)

    # Compute union of all channel IDs
    all_channels = set.union(*channel_sets)
    print(f"Session {session_id}: Total channels (union): {len(all_channels)}")

    # Sort channel IDs and create a mapping for indexing
    sorted_channels = sorted(all_channels)

    # Retrieve channel information (probe IDs and CCF coordinates)
    channel_info = {}  # Dictionary to store channel-specific info
    for ch in sorted_channels:
        # Get the probe ID and the ecephys structure acronym
        probe_id = session.channels.probe_id.loc[ch]
        ecephys_structure_acronym = session.channels.ecephys_structure_acronym.loc[ch]
        
        # Get the CCF coordinates
        anterior_posterior_ccf = session.channels.anterior_posterior_ccf_coordinate.loc[ch]
        dorsal_ventral_ccf = session.channels.dorsal_ventral_ccf_coordinate.loc[ch]
        left_right_ccf = session.channels.left_right_ccf_coordinate.loc[ch]
        
        # Store the information in the channel_info dictionary
        channel_info[ch] = {
            "probe_id": probe_id,
            "ecephys_structure_acronym": ecephys_structure_acronym,
            "anterior_posterior_ccf": anterior_posterior_ccf,
            "dorsal_ventral_ccf": dorsal_ventral_ccf,
            "left_right_ccf": left_right_ccf
        }

    # Initialize data for padded waveforms
    n_channels = len(sorted_channels)
    n_timepoints = unit_waveforms[0].shape[1]  # Assuming all units have the same waveform length
    channel_index_map = {ch: i for i, ch in enumerate(sorted_channels)}

    # Pad waveforms to include all channels
    padded_waveforms = []
    for waveforms in unit_waveforms:
        padded_data = np.zeros((n_channels, n_timepoints))  # Initialize zero array
        for i, ch in enumerate(waveforms.channel_id.values):
            if ch in channel_index_map:
                idx = channel_index_map[ch]  # Map channel to its sorted index
                padded_data[idx] = waveforms[i]  # Assign waveform data
        padded_waveforms.append(padded_data)

    # Stack padded waveforms into a single array
    padded_waveforms = np.stack(padded_waveforms)
    print(f"Session {session_id}: Shape of padded_waveforms: {padded_waveforms.shape}")

    # Create output directory for this session
    output_dir = os.path.join(f"data/session_{session_id}")
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "channel_ids.npy"), np.array(sorted_channels))
    np.save(os.path.join(output_dir, "padded_waveforms.npy"), padded_waveforms)
    np.save(os.path.join(output_dir, "channel_info.npy"), channel_info)
