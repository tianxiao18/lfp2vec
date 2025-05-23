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
    
    # Create a dictionary to classify channels by brain area
    brain_area_channels = {}
    for ch, info in channel_info.items():
        # Handle NaN or None cases
        ecephys_structure_acronym = info["ecephys_structure_acronym"]
        if ecephys_structure_acronym is None or ecephys_structure_acronym != ecephys_structure_acronym:
            ecephys_structure_acronym = "[UNK]"  # Set to "[UNK]" if NaN or None
        
        if ecephys_structure_acronym not in brain_area_channels:
            brain_area_channels[ecephys_structure_acronym] = []
        brain_area_channels[ecephys_structure_acronym].append(ch)

    # Set up the layout for the plots: width N and height number of channels (10 units randomly selected)
    brain_areas = sorted(list(brain_area_channels.keys()))
    
    # Set up the plot
    fig, axes = plt.subplots(nrows=10, ncols=len(brain_areas), figsize=(50, 18))

    # Plot waveforms for each brain area and channel
    for area_idx, area in enumerate(brain_areas):
        # Get the channels for this brain area
        channels = brain_area_channels[area][:10]
        
        # For each channel, plot the waveforms of randomly selected units
        for ch_idx, ch in enumerate(channels):
            ax = axes[ch_idx, area_idx]
            if ch_idx == 0: ax.set_title(f"Area {area}")
            ax.set_ylabel(f"Channel {ch}")

            counter = 0
            # Find the corresponding waveform data for each unit
            for unit_idx in range(padded_waveforms.shape[0]):
                ch_idx_in_data = np.where(channel_ids == ch)[0][0]  # Find the corresponding channel index
                waveform = padded_waveforms[unit_idx, ch_idx_in_data, :]  # Extract the waveform for this channel
                if np.min(waveform) == 0 and np.max(waveform) == 0: continue  # Skip if waveform is all zeros
                ax.plot(waveform + counter)  # Plot the waveform for the unit
                counter += 1
                if counter == 3: break
            
    # Adjust layout and save the figure
    image_path = os.path.join("data", session_dir, "channel_waveforms.png")
    plt.title(session_dir)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close(fig)
    print(f"Saved visualizations for {session_dir} to {image_path}")
