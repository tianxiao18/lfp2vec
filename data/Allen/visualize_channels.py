import os
import numpy as np
import matplotlib.pyplot as plt



# All session directories
sessions = [ "session_719161530", "session_768515987", "session_771160300", 
             "session_771990200", "session_778998620", "session_794812542", "session_798911424" ]

# Iterate over each session
for session_dir in sessions:
    # Load channel_info
    channel_info_path = os.path.join("data", session_dir, "channel_info.npy")
    if not os.path.exists(channel_info_path):
        print(f"Warning: {channel_info_path} not found.")
        continue

    channel_info = np.load(channel_info_path, allow_pickle=True).item()

    # Extract CCF coordinates and structure acronyms from channel_info
    anterior_posterior_ccf = []
    dorsal_ventral_ccf = []
    left_right_ccf = []
    ecephys_structure_acronym = []

    for ch, info in channel_info.items():
        anterior_posterior_ccf.append(info["anterior_posterior_ccf"])
        dorsal_ventral_ccf.append(info["dorsal_ventral_ccf"])
        left_right_ccf.append(info["left_right_ccf"])
        acronym = info["ecephys_structure_acronym"]
        if acronym is None or acronym != acronym: acronym = "[UNKNOWN]"  # Handle NaN or None acronyms
        ecephys_structure_acronym.append(acronym)

    # Convert data to NumPy arrays
    anterior_posterior_ccf = np.array(anterior_posterior_ccf)
    dorsal_ventral_ccf = np.array(dorsal_ventral_ccf)
    left_right_ccf = np.array(left_right_ccf)

    # Create color mapping (using a wider range of colors)
    unique_acronyms = sorted(list(set(ecephys_structure_acronym)))
    
    # Use a color map with a larger range of colors
    color_map = plt.get_cmap("tab20", len(unique_acronyms))  # Select more colors
    acronym_to_color = {acronym: color_map(i / len(unique_acronyms)) for i, acronym in enumerate(unique_acronyms)}
    if "APN" in acronym_to_color: acronym_to_color["APN"] = (0.1, 0.2, 0.8, 1.0)  # Modify color for APN

    # Count the number of channels for each brain region and show this in the legend
    acronym_counts = {acronym: ecephys_structure_acronym.count(acronym) for acronym in unique_acronyms}

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Assign color to each channel based on its structure acronym
    for i, acronym in enumerate(ecephys_structure_acronym):
        color = acronym_to_color[acronym]
        ax.scatter(left_right_ccf[i], dorsal_ventral_ccf[i], anterior_posterior_ccf[i], color=color, s=30)

    # Set axis labels
    ax.set_xlabel("Left-Right CCF Coordinate")
    ax.set_ylabel("Dorsal-Ventral CCF Coordinate")
    ax.set_zlabel("Anterior-Posterior CCF Coordinate")

    # Display the legend (with structure acronym and number of channels)
    handles = []
    labels = []
    for acronym in unique_acronyms:
        color = acronym_to_color[acronym]
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
        labels.append(f"{acronym} ({acronym_counts[acronym]})")  # Display brain region name and corresponding channels count
    
    ax.legend(handles, labels, title="Structure Acronym")

    # Save the image without displaying
    image_path = os.path.join("data", session_dir, "channel_positions_&_acronym.png")
    plt.title(session_dir)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close(fig)
    print(f"Saved visualizations for {session_dir} to {image_path}")
