
import os
import sys
import pickle

sys.path.append( os.path.join( os.getcwd(), 'utls' ) )
sys.path.append( os.path.join( os.getcwd(), 'mmodels' ) )

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from dotenv import load_dotenv
import json
import glob

from tutls import NormLoss
from dutls import SMData
from ms4dvar import LitModel, AutoEncoder

load_dotenv(os.path.join(os.getcwd(), 'config.env'))
PATH_DATA = os.getenv('PATH_DATA')
# MODEL = 'MM_GS10it_loadckpt_5it'
MODEL = 'MM_GS10it_loadckpt_5it_ECMWFzero'
PATH_MODELS = f'/home/administrateur/Desktop/4DVarNet-exports/learning_curves/ltvs_ECMWF/{MODEL}'

with open(os.path.join(PATH_MODELS, 'HYPERPARAMS.json'), 'r') as filestream:
    CPARAMS = json.load(filestream)
filestream.close()

FORMAT_SIZE = 24
CPARAMS.update({'FORMAT_SIZE' : FORMAT_SIZE})


test_set = SMData(os.path.join(PATH_DATA, 'test'), 'SITU', '2011')
test_loader = DataLoader(test_set, batch_size = test_set.__len__(), shuffle = False)

N_UPA = test_set.get_modality_data_size('upa')
N_ECMWF = test_set.get_modality_data_size('wind_ecmwf')
N_SITU = test_set.get_modality_data_size('wind_situ')
N_DATA = N_UPA + N_ECMWF + N_SITU

encoder = torch.nn.Sequential(
    nn.Conv1d(N_DATA, 128, kernel_size = 3, padding = 'same'),
    nn.Dropout(CPARAMS['DROPOUT']),
    nn.LeakyReLU(0.1),
    nn.Conv1d(128, CPARAMS['LATENT_DIM'], kernel_size = 3, padding = 'same'),
    nn.Dropout(CPARAMS['DROPOUT'])
)
decoder = torch.nn.Sequential(
    nn.Conv1d(CPARAMS['LATENT_DIM'], 128, kernel_size = 3, padding = 'same'),
    nn.Dropout(CPARAMS['DROPOUT']),
    nn.LeakyReLU(0.1),
    nn.Conv1d(128, N_DATA, kernel_size = 3, padding = 'same'),
    nn.Dropout(CPARAMS['DROPOUT']),
    nn.LeakyReLU(0.1)
)

Phi = AutoEncoder(encoder, decoder)

if CPARAMS['MM_ECMWF']:
    shape_data_Obs = (CPARAMS['BATCH_SIZE'], N_UPA + N_ECMWF, FORMAT_SIZE)
    shape_data_Solver = [N_UPA + N_ECMWF + N_SITU, FORMAT_SIZE]
else:
    shape_data_Obs = (CPARAMS['BATCH_SIZE'], N_UPA, FORMAT_SIZE)
    shape_data_Solver = [N_UPA + N_SITU, FORMAT_SIZE]
#end

shape_data_modalities = {'upa' : N_UPA, 'ecmwf' : N_ECMWF, 'situ' : N_SITU}

shape_data = (CPARAMS['BATCH_SIZE'], N_UPA + N_ECMWF, FORMAT_SIZE)

u_recos = []
u_datas = []
if MODEL == 'MM_GS10it_loadckpt_5it_ECMWFzero':
    MODEL_NAME = '4DVAR_AE_MM_gs5it_lckpt_gs10_zero'
if MODEL == 'MM_GS10it_loadckpt_5it':
    MODEL_NAME = '4DVAR_AE_MM_gs5it_lckpt_gs10'
#end

lrange = (7, 10)
for run in range(7, 10):   # (0,3) , (3, 7), (7, 10)
    
    lit_model = LitModel(
        Phi,
        shape_data_modalities,
        test_set.preprocess_params,
        CPARAMS
    )
    
    CKPT_NAME = glob.glob(os.path.join(PATH_MODELS, f'{run}-' + MODEL_NAME + '-epoch=*.ckpt'))[0]
    print('\n\nCHECKPOINT (TEST) : ' + CKPT_NAME + '\n\n')
    checkpoint_model = open(CKPT_NAME, 'rb')
    lit_model_state_dict = torch.load(checkpoint_model, map_location = torch.device('cpu'))['state_dict']
    lit_model.load_state_dict(lit_model_state_dict)
    
    profiler_kwargs = {'max_epochs' : CPARAMS['EPOCHS'], 'log_every_n_steps' : 1}
    trainer = pl.Trainer(**profiler_kwargs)
    
    lit_model.eval()
    lit_model.Phi.eval()
    lit_model.model.eval()
    trainer.test(lit_model, test_loader)
    
    u_data = lit_model.samples_to_save[0]['u_data'].cpu().detach().numpy()
    u_reco = lit_model.samples_to_save[0]['u_reco'].cpu().detach().numpy()
    
    pred_error_metric = NormLoss((u_data - u_reco), mask = None, divide = True, rmse = True)
    mask_central = torch.zeros((u_data.shape)); mask_central[:, FORMAT_SIZE // 2, :] = 1
    pred_error_metric_central = NormLoss((u_data - u_reco), mask = mask_central, divide = True, rmse = True)
    r2_metric = r2_score(u_data.reshape(-1,1), u_reco.reshape(-1,1))
    
    print('R² score = {:.4f}'.format(r2_metric))
    print('RMSE     = {:.4f}'.format(pred_error_metric))
    
    u_datas.append(u_data)
    u_recos.append(u_reco)
    print(u_recos[-1].mean())
#end

if lrange == (0, 3):  batch = 0
if lrange == (3, 7):  batch = 1
if lrange == (7, 10): batch = 2

pickle.dump( [u_datas, u_recos], open(os.path.join(os.getcwd(), 
                                      'reconstructions', 
                                     f'reco_{MODEL}_{batch}.pkl'), 'wb') )