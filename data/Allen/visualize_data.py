import numpy as np
import pandas as pd
import vedo

from tqdm import tqdm
from brainrender import Scene, settings
from brainrender.actors import Points
from brainrender.video import VideoMaker

settings.SHOW_AXES = False  # No axes

# ibl_df = pd.read_csv("../IBL/joined.csv")
ibl_df = pd.read_csv("../IBL/ibl_insertion2.csv")
ibl_unique_probes = ibl_df['probe_id'].unique()

allen_df = pd.read_csv("data/joined.csv")
allen_unique_probes = allen_df['probe_id'].unique()

session_probes = ['719161530', '794812542', '778998620', '798911424', '771990200', '771160300', '768515987',
                  '15763234-d21e-491f-a01b-1238eb96d389', '1a507308-c63a-4e02-8f32-3239a07dc578',
                  '4a45c8ba-db6f-4f11-9403-56e06a33dfa4', '56956777-dca5-468c-87cb-78150432cc57',
                  '5b49aca6-a6f4-4075-931a-617ad64c219c', '73918ae1-e4fd-4c18-b132-00cb555b1ad2',
                  'b39752db-abdb-47ab-ae78-e8608bbf50ed']

vedo.settings.default_backend = 'vtk'
popup_scene = Scene(atlas_name='allen_mouse_50um', title='Probe distribution - Allen + IBL')

popup_scene.hemisphere = 'right'

ca1 = popup_scene.add_brain_region('CA1', color='blue', alpha=0.2)
ca2 = popup_scene.add_brain_region('CA2', color='red', alpha=0.2)
ca3 = popup_scene.add_brain_region('CA3', color='green', alpha=0.2)
dg = popup_scene.add_brain_region('DG', color='yellow', alpha=0.2)

vis1 = popup_scene.add_brain_region('VIS', color='gray', alpha=0.2)
vis2 = popup_scene.add_brain_region('VISal', color='gray', alpha=0.2)
vis3 = popup_scene.add_brain_region('VISam', color='gray', alpha=0.2)
vis4 = popup_scene.add_brain_region('VISl', color='gray', alpha=0.2)
vis5 = popup_scene.add_brain_region('VISli', color='gray', alpha=0.2)
vis6 = popup_scene.add_brain_region('VISp', color='gray', alpha=0.2)
vis7 = popup_scene.add_brain_region('VISpm', color='gray', alpha=0.2)
vis8 = popup_scene.add_brain_region('VISrl', color='gray', alpha=0.2)

# Color the rest of the regions in different colors
apn = popup_scene.add_brain_region('APN', color='purple', alpha=0.2)
lp = popup_scene.add_brain_region('LP', color='orange', alpha=0.2)
mb = popup_scene.add_brain_region('MB', color='pink', alpha=0.2)
th = popup_scene.add_brain_region('TH', color='cyan', alpha=0.2)
lgd = popup_scene.add_brain_region('LGd', color='brown', alpha=0.2)
pros = popup_scene.add_brain_region('ProS', color='lightblue', alpha=0.2)
pol = popup_scene.add_brain_region('POL', color='lightgreen', alpha=0.2)
ppt = popup_scene.add_brain_region('PPT', color='lightyellow', alpha=0.2)
op = popup_scene.add_brain_region('OP', color='lightpink', alpha=0.2)
nott = popup_scene.add_brain_region('NOT', color='lightgray', alpha=0.2)
hpf = popup_scene.add_brain_region('HPF', color='lightpurple', alpha=0.2)
sub = popup_scene.add_brain_region('SUB', color='lightorange', alpha=0.2)
zi = popup_scene.add_brain_region('ZI', color='lightviolet', alpha=0.2)
lgv = popup_scene.add_brain_region('LGv', color='lightbrown', alpha=0.2)
sgn = popup_scene.add_brain_region('SGN', color='lightred', alpha=0.2)
scig = popup_scene.add_brain_region('SCig', color='lightblue', alpha=0.2)
mgm = popup_scene.add_brain_region('MGm', color='lightgreen', alpha=0.2)
mgv = popup_scene.add_brain_region('MGv', color='lightyellow', alpha=0.2)
vpm = popup_scene.add_brain_region('VPM', color='lightpink', alpha=0.2)

# Add labels
popup_scene.add_label(ca1, "CA1")
popup_scene.add_label(ca2, "CA2")
popup_scene.add_label(ca3, "CA3")
popup_scene.add_label(dg, "DG")
popup_scene.add_label(vis1, "VIS")
popup_scene.add_label(vis2, "VISal")
popup_scene.add_label(vis3, "VISam")
popup_scene.add_label(vis4, "VISl")
popup_scene.add_label(vis5, "VISli")
popup_scene.add_label(vis6, "VISp")
popup_scene.add_label(vis7, "VISpm")
popup_scene.add_label(vis8, "VISrl")
popup_scene.add_label(apn, "APN")
popup_scene.add_label(lp, "LP")
popup_scene.add_label(mb, "MB")
popup_scene.add_label(th, "TH")
popup_scene.add_label(lgd, "LGd")
popup_scene.add_label(pros, "ProS")
popup_scene.add_label(pol, "POL")
popup_scene.add_label(ppt, "PPT")
popup_scene.add_label(op, "OP")
popup_scene.add_label(nott, "NOT")
popup_scene.add_label(hpf, "HPF")
popup_scene.add_label(sub, "SUB")
popup_scene.add_label(zi, "ZI")
popup_scene.add_label(lgv, "LGv")
popup_scene.add_label(sgn, "SGN")
popup_scene.add_label(scig, "SCig")
popup_scene.add_label(mgm, "MGm")
popup_scene.add_label(mgv, "MGv")
popup_scene.add_label(vpm, "VPM")

min_ap, max_ap = 100000, -100000

for probe in tqdm(allen_unique_probes):
    probe_data = allen_df.loc[allen_df['probe_id'] == probe]
    if probe_data.empty:
        continue
    ccf_coords = probe_data[['anterior_posterior_ccf_coordinate',
                             'dorsal_ventral_ccf_coordinate',
                             'left_right_ccf_coordinate']].values
    if ccf_coords[:, 0].min() < min_ap:
        min_ap = ccf_coords[:, 0].min()
    if ccf_coords[:, 0].max() > max_ap:
        max_ap = ccf_coords[:, 0].max()
    color = 'red' if str(np.array(probe_data['session_id'])[0]) in session_probes else 'cyan'
    radius = 33 if str(np.array(probe_data['session_id'])[0]) in session_probes else 10
    points = Points(ccf_coords, radius=radius, colors=color)
    popup_scene.add(points)

for probe in tqdm(ibl_unique_probes):
    probe_data = ibl_df.loc[ibl_df['id'] == probe]
    if probe_data.empty:
        continue
    ccf_coords = probe_data[['anterior_posterior_ccf_coordinate',
                             'dorsal_ventral_ccf_coordinate',
                             'left_right_ccf_coordinate']].values
    # Mirror the coordinates
    midpoint = 5700
    ccf_coords[:, 2] = 2 * midpoint - ccf_coords[:, 2]
    if ccf_coords[:, 0].min() < min_ap:
        min_ap = ccf_coords[:, 0].min()
    if ccf_coords[:, 0].max() > max_ap:
        max_ap = ccf_coords[:, 0].max()

    points = Points(ccf_coords, radius=33, colors='black')
    popup_scene.add(points)

# Comment this slices to see the full brain
plane_min = popup_scene.atlas.get_plane(pos=(min_ap, 0, 0), norm=(1, 0, 0), color='red', alpha=0.5)
plane_max = popup_scene.atlas.get_plane(pos=(max_ap, 0, 0), norm=(-1, 0, 0), color='blue', alpha=0.5)
popup_scene.slice(plane_min)
popup_scene.slice(plane_max)

# Min AP: 6350.0, Max AP: 9875.0

# Render the scene
popup_scene.render()
