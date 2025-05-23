import scipy.io as spio
import mat73
import scipy.io.matlab as spiomat
import numpy as np
import scipy
import matplotlib.pyplot as plt

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    try:
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return _check_keys(data)
    except:
        return mat73.loadmat(filename)


class DotDict(dict):
    # def __getattr__(self, name):
    #     return self[name]
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value
    
    def __repr__(self) -> str:
        return str(list(self.keys()))
    
    def __getstate__(self):
        # Return what you want to pickle
        return self.__dict__

    def __setstate__(self, state):
        # Restore state from the unpickled state
        self.__dict__.update(state)
    

def loadmat_full(filename,structname=None):
    if structname is None:
        mat = loadmat(filename)
    else:
        mat = loadmat(filename)[structname]
    mat = DotDict(mat)
    return mat

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        # if isinstance(dict[key], spiomat.mio5_params.mat_struct):
        if isinstance(dict[key], scipy.io.matlab.mat_struct):
            dict[key] = _todict(dict[key])
        
        elif isinstance(dict[key], np.ndarray):
            
            dict_key_res = np.zeros_like(dict[key])
            # with np.nditer([dict[key],dict_key_res],op_flags=[['readonly'], ['readwrite']]) as it:
            for ind,x in np.ndenumerate(dict_key_res): 
                orig_val = dict[key][ind]
                
                if isinstance(orig_val,scipy.io.matlab.mat_struct):
                    dict_key_res[ind] = _todict(orig_val)
                else:
                    dict_key_res[ind] = orig_val

            dict[key] = dict_key_res
        
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        # if isinstance(elem, spiomat.mio5_params.mat_struct):
        if isinstance(elem, scipy.io.matlab.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem,np.ndarray) and len(elem) >= 1: # used for the multi maze case; then there might be a cell array of struct that is not correctly unwrapped
            # if isinstance(elem[0], spiomat.mio5_params.mat_struct):
            if isinstance(elem, scipy.io.matlab.mat_struct):
                dict[strg] = np.array([_todict(e) for e in elem],dtype=object)
            else:
                dict[strg]  = elem
        else:
            dict[strg]  = elem
    return dict

def visualize_max_waveform_channel(maxWaveformCh, rows=128, cols=8):
    # binary_spike_matrix = np.zeros(rows * cols)
    # for i in maxWaveformCh:
    #     binary_spike_matrix[i] += 1
    # binary_spike_matrix = binary_spike_matrix.reshape(rows, cols)
    # print(binary_spike_matrix.shape)
    # print(np.max(binary_spike_matrix))

    # fig, ax = plt.subplots(1, cols, figsize=(12, 12))
    # for i in range(cols):
    #     heatmap = np.expand_dims(binary_spike_matrix[:, i], axis=1)
    #     im = ax[i].imshow(heatmap, aspect='auto', vmin=0, origin='lower')

    # plt.tight_layout()
    # cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8)
    # cbar.set_label('Spike Count')

    x = maxWaveformCh % rows
    y = maxWaveformCh // rows
    fig, ax = plt.subplots()
    ax.scatter(y, x)
    ax.invert_yaxis()
    plt.savefig('results/spike_heatmap.png')    


def main():
    file_path = "/scratch/th3129/shared/Neuronexus_dataset/AD_HF01_Session1h/AD_HF01_Session1h.spikes.cellinfo.mat"
    # file_path = "/scratch/th3129/shared/Neuronexus_dataset/NN_syn_20230601/NN_syn_20230601.spikes.cellinfo.mat"
    # mat = loadmat_full(file_path, structname='cell_metrics')
    # print(mat.keys())
    mat = loadmat_full(file_path, structname='spikes')
    visualize_max_waveform_channel(mat['maxWaveformCh'])


if __name__ == "__main__":
    main()