# WARNING: This script will download a lot of data (~350 GB). Make sure you have enough space on your disk.

import os
import shutil
import warnings

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

warnings.filterwarnings('ignore')


# Downloading the data
print(os.getcwd())

hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG'}
manifest_path = 'data'
if not os.path.exists(manifest_path):
    os.makedirs(manifest_path)

manifest_path = os.path.join(manifest_path, "manifest.json")
data_cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions_list = data_cache.get_session_table()
mask = sessions_list['ecephys_structure_acronyms'].apply(lambda acronyms: bool(hc_acronyms.intersection(set(acronyms))))

# Uncomment this if you only want to download 2 sessions (773418906, 797828357) (~30 GB)
# mask = sessions_list.index.isin([773418906, 781842082])


filtered_sessions = sessions_list[mask]
print(f"Total number of sessions: {len(filtered_sessions)}")

for session_id, row in filtered_sessions.iterrows():
    truncated_file = True
    directory = os.path.join(manifest_path + '/session_' + str(session_id))
    print(session_id)
    while truncated_file:
        session = data_cache.get_session_data(session_id)
        try:
            print(f"|---{session.specimen_name}")
            truncated_file = False
        except OSError:
            shutil.rmtree(directory)
            print("|---|---|---Truncated spikes file, re-downloading")

    for probe_id, probe in session.probes.iterrows():

        print(f"|---|---{probe.description}")
        truncated_lfp = True

        while truncated_lfp:
            try:
                lfp = session.get_lfp(probe_id)
                truncated_lfp = False
            except OSError:
                fname = directory + '/probe_' + str(probe_id) + '_lfp.nwb'
                os.remove(fname)
                print("|---|---|---Truncated LFP file, re-downloading")
            except ValueError:
                print("|---|---|---LFP file not found.")
                truncated_lfp = False
