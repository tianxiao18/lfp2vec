import warnings
import gc
import argparse

from script.utils import *

warnings.filterwarnings('ignore')

print("Job start")
print(os.getcwd())



def parse_arguments():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Parse arguments for LFP data processing.")

    # Add arguments
    parser.add_argument(
        '--chunk_size', 
        type=int, 
        required=True, 
        default=3,
        help="Size of each data chunk to be processed."
    )

    parser.add_argument(
        '--start_time', 
        type=float, 
        required=True, 
        default=30,
        help="Start time for the analysis in seconds."
    )

    parser.add_argument(
        '--n_chunks', 
        type=int, 
        required=True, 
        default=1800,
        help="Number of chunks to be processed."
    )

    parser.add_argument(
        '--start_eid', 
        type=int, 
        required=False, 
        default=0,
        help="Start event ID for data processing."
    )

    parser.add_argument(
        '--end_eid', 
        type=int, 
        required=False, 
        default=19,
        help="End event ID for data processing."
    )

    # Parse the arguments
    args = parser.parse_args()

    return args

def load_session(eid):
    return np.load(f'../../data/ibl/{eid}_lfp.npy'), np.load(f'../../data/ibl/{eid}_fs.npy'), np.load(f'../../data/ibl/{eid}_label_lfp.npy', allow_pickle=True)

def raw_data_creator(chunk_size, start_time, n_chunks, start_eid=0, end_eid=19):
    """
    output: X,y,trial_number,channel_number
    """
    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'VIS'}
    acronyms_arr = np.array(sorted(list(hc_acronyms)))
    acronyms_arr_num = np.arange(len(acronyms_arr))
    acr_dict = {acr: i for i, acr in enumerate(acronyms_arr)}
    print(acr_dict)



    ## load data
    filtered_sessions = np.load('selected_eids.npy')
    
    print(f"Total number of sessions: {len(filtered_sessions)}")

    # Model execution

    output_path = "../../data/ibl/raw"
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    for eid in filtered_sessions[start_eid:end_eid]:
        ### if exist in ../../data/ibl/raw, then skip
        session_data, sampling_rate, labels = load_session(eid)
        ## calculate the time in seconds in the session_data
        time = len(session_data[1]) / sampling_rate
        print(f"Session {eid} has {time} seconds of data")
        region_counts = [0, 0, 0, 0, 0]
        data_list = []

        for region in hc_acronyms:
            # chan_ids = np.where(labels == region)[0]
            chan_ids = [i for i, label in enumerate(labels) if region in label]
            regional_lfp_data = session_data[chan_ids]
            lfp_channel_index = chan_ids
            print(f"Region {region} has {len(chan_ids)} channels")

            for i, (chan_id, channel_data) in enumerate(zip(chan_ids, regional_lfp_data)):
                time_mul = chunk_size
                window_size = int(time_mul * sampling_rate)
                start = int(start_time * sampling_rate)
                end = int(start + window_size)
                count = 0 # trial index as the number of chunks
                while count < n_chunks and end < len(channel_data):
                    row = channel_data[start:end]
                    if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                        base_rms_mini = calculate_rms(row)
                        pow_whole_mini, rms_whole_mini = calculate_power(row, sampling_rate)
                        pow_delta_mini, rms_delta_mini = calculate_power(row, sampling_rate, (0, 4))
                        pow_theta_mini, rms_theta_mini = calculate_power(row, sampling_rate, (4, 8))
                        pow_alpha_mini, rms_alpha_mini = calculate_power(row, sampling_rate, (8, 12))
                        pow_beta_mini, rms_beta_mini = calculate_power(row, sampling_rate, (12, 30))
                        pow_gamma_mini, rms_gamma_mini = calculate_power(row, sampling_rate, (30, -1))
                        probe = 0
                        # Append data to the list
                        data_list.append(
                            ([
                            base_rms_mini, pow_whole_mini,
                            pow_delta_mini,
                            pow_theta_mini, pow_alpha_mini, pow_beta_mini, pow_gamma_mini, rms_whole_mini,
                            rms_delta_mini,
                            rms_theta_mini, rms_alpha_mini, rms_beta_mini, rms_gamma_mini], acr_dict[region], count, chan_id)
                        )

                        start += window_size
                        end += window_size
                        count += 1

                        region_counts[acr_dict[region]] += 1
                print(f"{count} of chunks processed for channel {chan_id} in region {region}")

        gc.collect()
        print(f"Data size: {len(data_list)}")
        print(region_counts)

        # Convert the list to a DataFrame
        with open(os.path.join(output_path, f"{eid}_data.pickle"), 'wb') as file:
                print(f"Length of data_list: {len(data_list)}")
                pickle.dump(data_list, file)
    gc.collect()

# raw_csv_creator(chunk_size, start_time, n_chunks)

if __name__ == "__main__":
    args = parse_arguments()
    chunk_size, start_time, n_chunks, start_eid, end_eid = args.chunk_size, args.start_time, args.n_chunks, args.start_eid, args.end_eid

    raw_data_creator(chunk_size, start_time, n_chunks, start_eid, end_eid)
