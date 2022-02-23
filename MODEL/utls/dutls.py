
import os
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MMData(Dataset):
    
    def __init__(self, path_data, wind_values, data_title,
                 sar_values        = 'GMF',
                 dtype             = torch.float32,
                 convert_to_tensor = True):
        
        '''
        For semplicity:
            
            - X == SAR images (GMF)
            - Y == UPA recordings
            - W == Wind field patches (SAR ECMWF)
            - U == Wind speed values (ECMWF or INSITU)
        '''
        
        # SAR_GMF    = pickle.load( open(os.path.join(path_data, 'SAR_GMF_{}_{}.pkl'.format(wind_values, data_title)), 'rb' ) )
        # SAR_NRCS   = pickle.load( open(os.path.join(path_data, 'SAR_NRCS_{}_{}.pkl'.format(wind_values, data_title)), 'rb') )
        # SAR_ECMWF  = pickle.load( open(os.path.join(path_data, 'SAR_ECMWF_{}_{}.pkl'.format(wind_values, data_title)), 'rb') )
        UPA        = pickle.load( open(os.path.join(path_data, 'UPA_{}_{}.pkl'.format(wind_values, data_title)), 'rb') )
        WIND_label = pickle.load( open(os.path.join(path_data, 'WIND_label_{}_{}.pkl'.format(wind_values, data_title)), 'rb') )
        
        # if sar_values == 'GMF':
        #     SAR = SAR_GMF
        # elif sar_values == 'NRCS':
        #     SAR = SAR_NRCS
        # #end
        # self.X = SAR['data']
        self.Y = UPA['data']
        # self.W = SAR_ECMWF['data']
        self.U = WIND_label['data']
        
        self.which_wind = WIND_label['which']
        
        self.preprocess_params = {
                # 'x' : SAR['nparms'],
                'y' : UPA['nparms'],
                # 'w' : SAR_ECMWF['nparms'],
                'u' : WIND_label['nparms']
            }
        
        # assert self.X.__len__() == self.Y.__len__()
        # assert self.X.__len__() == self.W.__len__()
        # assert self.X.__len__() == self.U.__len__()
        
        self.nsamples = self.Y.__len__()
        self.dtype    = dtype
        
        if convert_to_tensor: self.to_tensor()
        
    #end
    
    def __len__(self):
        
        return self.nsamples
    #end
    
    def __getitem__(self, idx):
        
        # return self.X[idx], self.Y[idx], self.W[idx], self.U[idx]
        return self.Y[idx], self.Y[idx], self.U[idx], self.U[idx]
    #end
    
    def get_modality_data_size(self, data = None, asdict = False):
        
        N = {
            # 'x' : np.int32( np.prod(self.X[0].shape[-2:]) ),
            'y' : np.int32(self.Y[0].shape[1]),
            # 'w' : np.int32( np.prod(self.W[0].shape[-2:]) ),
            'u' : np.int32(1)
        }
        
        if data is None:
            if asdict:
                return N
            else:
                return N['x'], N['y'], N['w'], N['u']
            #end
        else:
            return N[data]
        #end
    #end
    
    def to_tensor(self):
        
        for i in range(self.nsamples):
            
            # self.X[i]     = torch.Tensor(self.X[i]).type(self.dtype)
            self.Y[i]     = torch.Tensor(self.Y[i]).type(self.dtype)
            # self.W[i]     = torch.Tensor(self.W[i]).type(self.dtype)
            self.U[i]     = torch.Tensor(self.U[i]).type(self.dtype)
        #end
    #end
    
    def undo_preprocess(self, data_preprocessed, tag):
        
        v_min, v_max = self.preprocess_params[tag]
        data = (v_max - v_min) * data_preprocessed + v_min
        return data
    #end
    
#end


class SMData(Dataset):
    
    def __init__(self, path_data, wind_values, data_title,
                 dtype = torch.float32,
                 convert_to_tensor = True):
        
        UPA        = pickle.load( open(os.path.join(path_data, f'UPA_{data_title}.pkl'), 'rb') )
        WIND_situ  = pickle.load( open(os.path.join(path_data, f'WIND_label_SITU_{data_title}.pkl'), 'rb') )
        WIND_ecmwf = pickle.load( open(os.path.join(path_data, f'WIND_label_ECMWF_{data_title}.pkl'), 'rb') )
        
        self.UPA = UPA['data']
        self.WIND_situ = WIND_situ['data']
        self.WIND_ecmwf = WIND_ecmwf['data']
        
        self.which_wind = WIND_situ['which']
        
        self.preprocess_params = {
                'upa' : UPA['nparms'],
                'wind_situ' : WIND_situ['nparms'],
                'wind_ecmwf' : WIND_ecmwf['nparms']
        }
        
        self.nsamples = self.UPA.__len__()
        self.dtype    = dtype
        
        if convert_to_tensor: self.to_tensor()
    #end
    
    def __len__(self):
        
        return self.nsamples
    #end
    
    def __getitem__(self, idx):
        
        return self.UPA[idx], self.WIND_ecmwf[idx], self.WIND_situ[idx]
    #end
    
    def get_modality_data_size(self, data = None, asdict = False):
        
        N = {
            'upa' : np.int32(self.UPA[0].shape[1]),
            'wind_situ' : np.int32(1),
            'wind_ecmwf' : np.int32(1),
        }
        
        if data is None:
            if asdict:
                return N
            else:
                return N['upa'], N['wind_ecmwf'], N['wind_situ']
            #end
        else:
            return N[data]
        #end
    #end
    
    def to_tensor(self):
        
        for i in range(self.nsamples):
            
            self.UPA[i] = torch.Tensor(self.UPA[i]).type(self.dtype)
            self.WIND_situ[i] = torch.Tensor(self.WIND_situ[i]).type(self.dtype)
            self.WIND_ecmwf[i] = torch.Tensor(self.WIND_ecmwf[i]).type(self.dtype)
        #end
    #end
    
    def undo_preprocess(self, data_preprocessed, tag):
        
        v_min, v_max = self.preprocess_params[tag]
        data = (v_max - v_min) * data_preprocessed + v_min
        return data
    #end
#end


class TISMData(Dataset):
    
    def __init__(self, path_data, dtype = torch.float32):
        
        UPA  = pickle.load( open( os.path.join(path_data, 'UPA_SITU__Taylor_et_al_2020.pkl'), 'rb' ) )
        WIND = pickle.load( open( os.path.join(path_data, 'WIND_label_SITU__Taylor_et_al_2020.pkl'), 'rb' ) )
        
        self.Y = UPA['data']
        self.U = WIND['data']
        self.dtype = dtype
        
        self.preprocess_params = {
                'y' : UPA['nparms'],
                'u' : WIND['nparms']
            }
        
        assert self.Y.__len__() == self.U.__len__()
        
        self.nsamples = self.Y.__len__()
        
        self.to_tensor()
    #end
    
    def __len__(self):
        return self.nsamples
    #end
    
    def __getitem__(self, idx):
        return self.Y[idx], self.U[idx]
    #end
    
    def get_modality_data_size(self, data = None, asdict = False):
        
        N = {
            'y' : np.int32(self.Y[0].shape[0]),
            'u' : np.int32(1)
        }
        
        if data is None:
            if asdict:
                return N
            else:
                return N['y'], N['u']
            #end
        else:
            return N[data]
        #end
    #end
    
    def to_tensor(self):
        
        for i in range(self.nsamples):
            
            self.Y[i]     = torch.Tensor(self.Y[i]).type(self.dtype)
            self.U[i]     = torch.Tensor([self.U[i]]).type(self.dtype)
        #end
    #end
#end

def assess_shapes_equality(sample):
    
    # transitive property, check only agains sample[0]
    assert sample[0].shape[0] == sample[1].shape[0], 'SHAPES MISMATCH'
    assert sample[0].shape[0] == sample[2].shape[0], 'SHAPES MISMATCH'
    assert sample[0].shape[0] == sample[3].shape[0], 'SHAPES MISMATCH'
    return sample[0].shape[0]
#end





