import os
import numpy as np
import matplotlib.pyplot as plt



# All session directories
sessions = [ "session_719161530", "session_768515987", "session_771160300", 
             "session_771990200", "session_778998620", "session_794812542", "session_798911424" ]

# Iterate over each session
for session_dir in sessions:
    data_dir = os.path.join("data", session_dir)
    
    # Load preprocessed results
    padded_waveforms = np.load(os.path.join(data_dir, "padded_waveforms.npy"))
    channel_ids = np.load(os.path.join(data_dir, "channel_ids.npy"))
    channel_info = np.load(os.path.join(data_dir, "channel_info.npy"), allow_pickle=True).item()
    
    # Randomly select 8 units
    selected_units = sorted(np.random.choice(padded_waveforms.shape[0], size=10, replace=False).tolist())

    # Create a dictionary to classify channels by brain area
    brain_area_channels = {}
    for ch, info in channel_info.items():
        # Handle NaN or None cases
        ecephys_structure_acronym = info["ecephys_structure_acronym"]
        if ecephys_structure_acronym is None or ecephys_structure_acronym != ecephys_structure_acronym:
            ecephys_structure_acronym = "[UNKNOWN]"  # Set to "[UNKNOWN]" if NaN or None
        
        if ecephys_structure_acronym not in brain_area_channels:
            brain_area_channels[ecephys_structure_acronym] = []
        brain_area_channels[ecephys_structure_acronym].append(ch)

    # Set up the layout for the plots: width N and height 8 (each brain area 6 channels, 10 units)
    brain_areas = sorted(list(brain_area_channels.keys()))
    fig, axes = plt.subplots(nrows=10, ncols=len(brain_areas), figsize=(50, 18))

    # Plot waveforms for each unit
    for uiid, unit_idx in enumerate(selected_units):
        for area_idx, area in enumerate(brain_areas):
            counter = 0
            for ch_idx, ch in enumerate(brain_area_channels[area]):
                ax = axes[uiid, area_idx]
                if uiid == 0: ax.set_title(f"Area {area}")
                if area_idx == 0: ax.set_ylabel(f"Unit {unit_idx+1}")

                # Find the corresponding channel's waveform data
                ch_idx_in_data = np.where(channel_ids == ch)[0][0]  # Find the corresponding channel index
                waveform = padded_waveforms[unit_idx, ch_idx_in_data, :]  # Extract the waveform for this channel
                if np.min(waveform) == 0 and np.max(waveform) == 0: continue  # Skip if waveform is all zeros
                ax.plot(waveform + counter)  # Plot waveform with a vertical offset
                counter += 1
                if counter == 3: break  # Limit to 3 channels per brain area

    image_path = os.path.join("data", session_dir, "unit_waveforms.png")
    plt.title(session_dir)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close(fig)
    print(f"Saved visualizations for {session_dir} to {image_path}")
