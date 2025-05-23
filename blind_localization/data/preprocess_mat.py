import path_setup
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