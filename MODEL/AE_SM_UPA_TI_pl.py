
print('\n\n')
print('###############################################')
print('AE SM TI UPA')
print('###############################################')
print('\n\n')

import os
import sys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), 'config.env'))

sys.path.append( os.path.join( os.getcwd(), 'utls' ) )
sys.path.append( os.path.join( os.getcwd(), '4dvar' ) )

import pickle
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import pathlib
import json

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import r2_score

from tutls import NormLoss, xavier_weights_initialization
from dutls import MMData, TISMData
from gutls import plot_UPA, plot_WS, plot_WS_scatter


class AutoEncoder(nn.Module):
    
    def __init__(self, encoder, decoder):
        
        super(AutoEncoder, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.apply(xavier_weights_initialization)
    #end
    
    def forward(self, data):
        
        latent = self.encoder(data)
        reco = self.decoder(latent)
        return reco
    #end
#end


class LitModel(pl.LightningModule):
    
    def __init__(self, network, preprocess_params):
        
        super(LitModel, self).__init__()
        
        self.network = network
        self.hparams.automatic_optimization = True
        self.preprocess_params = preprocess_params
        self.test_rmse = np.float64(0.)
        self.samples_to_save = list()
    #end
    
    def configure_optimizers(self):
        
        optimizers = optim.Adam( [{'params'       : self.network.parameters(),
                                   'lr'           : MODEL_LR,
                                   'weight_decay' : MODEL_WD}]
                                )
        return optimizers
    #end
    
    def compute_loss(self, batch, phase = 'train'):
        
        '''Reshape the input data'''
        batch_size = batch[0].shape[0]
        if TAYLOR_DS:
            data_UPA = batch[0].reshape(batch_size, N)
            data_ws  = batch[1].reshape(batch_size, Nu)
        else:
            data_UPA = batch[1].reshape(batch_size * FORMAT_SIZE, N)
            data_ws  = batch[3].reshape(batch_size * FORMAT_SIZE, Nu)
        #end
        
        '''Produce the masks according to the data sparsity patterns'''
        mask_UPA = torch.zeros_like(data_UPA)
        mask_UPA[data_UPA.isnan().logical_not()] = 1.
        mask_UPA[data_UPA == 0] = 0
        data_UPA[data_UPA.isnan()] = 0.
        
        mask_ws = torch.zeros_like(data_ws)
        mask_ws[data_ws.isnan().logical_not()] = 1.
        mask_ws[data_ws == 0] = 0
        data_ws[data_ws.isnan()] = 0.
        
        data_input = torch.cat( (data_UPA, 0. * data_ws), dim = -1 )
        outputs = self.network(data_input)
        
        if TAYLOR_DS:
            reco_UPA = outputs[:, :N]
            reco_ws  = outputs[:, -Nu:]
        else:
            reco_UPA = outputs[:, :N].reshape(batch_size, FORMAT_SIZE, N)
            reco_ws  = outputs[:, -Nu:].reshape(batch_size, FORMAT_SIZE, Nu)
            
            data_UPA = data_UPA.reshape(batch_size, FORMAT_SIZE, N)
            mask_UPA = mask_UPA.reshape(batch_size, FORMAT_SIZE, N)
            data_ws  = data_ws.reshape(batch_size, FORMAT_SIZE, Nu)
            mask_ws  = mask_ws.reshape(batch_size, FORMAT_SIZE, Nu)
        #end
        
        if phase == 'test':
            
            data_UPA = self.undo_preprocess(data_UPA, self.preprocess_params['y'])
            reco_UPA = self.undo_preprocess(reco_UPA, self.preprocess_params['y'])
            data_ws  = self.undo_preprocess(data_ws,  self.preprocess_params['u'])
            reco_ws  = self.undo_preprocess(reco_ws,  self.preprocess_params['u'])
            outputs = { 'y_data' : data_UPA, 'y_reco' : reco_UPA,
                        'u_data' : data_ws,  'u_reco' : reco_ws
            }
            
            self.samples_to_save.append( outputs )
        #end
        
        loss_data = NormLoss((data_UPA - reco_UPA), mask = mask_UPA, divide = True)
        loss_ws   = NormLoss((data_ws - reco_ws),   mask = mask_ws,  divide = True)
        loss = WEIGHT_DATA * loss_data + WEIGHT_PRED * loss_ws
        
        return dict({'loss' : loss, 'loss_reco' : loss_data, 'loss_pred' : loss_ws}), outputs
    #end
    
    def training_step(self, batch, batch_idx):
        
        metrics, outs = self.compute_loss(batch)
        loss = metrics['loss']
        loss_reco = metrics['loss_reco']
        loss_pred = metrics['loss_pred']
        
        self.log('loss', loss,           on_step = True, on_epoch = True, prog_bar = True)
        self.log('loss_reco', loss_reco, on_step = True, on_epoch = True, prog_bar = True)
        self.log('loss_pred', loss_pred, on_step = True, on_epoch = True, prog_bar = True)
        
        return loss
    #end
    
    def test_step(self, batch, batch_idx):
        
        if batch_idx >= 1:
            raise ValueError('Batch index greater that 0 : Reformat DataLoader')
        #end
        
        with torch.no_grad():
            metrics, outs = self.compute_loss(batch, phase = 'test')
            
            metrics['loss']      = np.sqrt(metrics['loss'].item())
            metrics['loss_reco'] = np.sqrt(metrics['loss_reco'].item())
            metrics['loss_pred'] = np.sqrt(metrics['loss_pred'].item())
            
            self.log('loss_test',      metrics['loss'].item())
            self.log('loss_reco_test', metrics['loss_reco'].item())
            self.log('loss_pred_test', metrics['loss_pred'].item())
        #end
        
        return metrics, outs
    #end
    
    def undo_preprocess(self, data, params):
        
        return (params[1] - params[0]) * data + params[0]
    #end
#end



###############################################################################
# MAIN
###############################################################################


# CONSTANTS
WIND_VALUES = 'SITU'
DATA_TITLE  = '2011'
PLOTS       = False
RUNS        = 10
COLOCATED   = False
TAYLOR_DS   = True
TRAIN       = True
TEST        = True

FORMAT_SIZE = 24
MODEL_NAME  = 'AE_SM_UPA_TI_TaylorDS' if TAYLOR_DS else 'AE_SM_UPA_TI'
PATH_DATA   = os.getenv('PATH_DATA')
PATH_MODEL  = os.getenv('PATH_MODEL')

# HPARAMS
if TAYLOR_DS: # NE PAS TOUCHER !
    EPOCHS      = 200
    BATCH_SIZE  = 128
    LATENT_DIM  = 20
    DROPOUT     = 0.
    WEIGHT_DATA = 0.25
    WEIGHT_PRED = 1.
    MODEL_LR    = 1e-3
    MODEL_WD    = 1e-6
    WINDMAP_LR  = 1e-3
    WINDMAP_WD  = 1e-6
else:
    EPOCHS      = 200
    BATCH_SIZE  = 128
    LATENT_DIM  = 20
    DROPOUT     = 0.
    WEIGHT_DATA = 0.25
    WEIGHT_PRED = 1.
    MODEL_LR    = 1e-3
    MODEL_WD    = 1e-6
    WINDMAP_LR  = 1e-3
    WINDMAP_WD  = 1e-6
#end


'''
Initialize the performance metrics data structures
'''
windspeed_rmses = {
        'only_SAR'  : {'u' : np.zeros(RUNS), 'u_x' : np.zeros(RUNS), 'u_y' : np.zeros(RUNS)},
        'only_UPA'  : {'u' : np.zeros(RUNS), 'u_x' : np.zeros(RUNS), 'u_y' : np.zeros(RUNS), 'aggr' : 0.},
        'colocated' : {'u' : np.zeros(RUNS), 'u_x' : np.zeros(RUNS), 'u_y' : np.zeros(RUNS)},
}
predictions = list()


for run in range(RUNS):
    print('\n------\nRun {}'.format(run))
    
    if TAYLOR_DS:
        PATH_DATA = os.path.join( pathlib.Path(PATH_DATA).parent, 'Taylor_et_al_2020' )
        train_set = TISMData(os.path.join(PATH_DATA, 'train'))
        test_set  = TISMData(os.path.join(PATH_DATA, 'test'))
    else:
        train_set = MMData(os.path.join(PATH_DATA, 'train'), WIND_VALUES, '2011')
        test_set  = MMData(os.path.join(PATH_DATA, 'test_only_UPA'), WIND_VALUES, '2011')
    #end
    
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8)
    test_loader  = DataLoader(test_set, batch_size = test_set.__len__(), shuffle = False, num_workers = 8)
    
    N  = train_set.get_modality_data_size('y')
    Nu = train_set.get_modality_data_size('u')
    
    ''' MODEL TRAIN '''
    if TRAIN:
        
        if TAYLOR_DS: # NE PAS TOUCHER !
            encoder = torch.nn.Sequential(
                    nn.Linear(N + Nu, 128), 
                    nn.Dropout(DROPOUT), 
                    nn.LeakyReLU(0.1),
                    nn.Linear(128, LATENT_DIM), 
                    nn.Dropout(DROPOUT),
            )
            decoder = torch.nn.Sequential(
                    nn.Linear(LATENT_DIM, 32), 
                    nn.Dropout(DROPOUT), 
                    nn.LeakyReLU(0.1),
                    nn.Linear(32, N + Nu), 
                    nn.Dropout(DROPOUT), 
                    nn.LeakyReLU(0.1)
            )
        else:
            encoder = torch.nn.Sequential(
                    nn.Linear(N + Nu, 128), 
                    nn.Dropout(DROPOUT), 
                    nn.LeakyReLU(0.1),
                    nn.Linear(128, LATENT_DIM), 
                    nn.Dropout(DROPOUT),
            )
            decoder = torch.nn.Sequential(
                    nn.Linear(LATENT_DIM, 128), 
                    nn.Dropout(DROPOUT), 
                    nn.LeakyReLU(0.1),
                    nn.Linear(128, N + Nu), 
                    nn.Dropout(DROPOUT), 
                    nn.LeakyReLU(0.1)
            )
        #end
        
        network = AutoEncoder(
                encoder = encoder,
                decoder = decoder       
        )
        
        profiler_kwargs = {'max_epochs' : EPOCHS, 'log_every_n_steps' : 1}
        
        lit_model = LitModel(network, preprocess_params = test_set.preprocess_params)
        trainer = pl.Trainer(**profiler_kwargs, progress_bar_refresh_rate = 10)
        trainer.fit(lit_model, train_loader)
        
        if RUNS == 1:
            torch.save({'trainer' : trainer, 'model' : lit_model, 
                        'name' : MODEL_NAME, 'saved_at_time' : datetime.now()},
                       open(os.path.join(PATH_MODEL, '{}.pkl'.format(MODEL_NAME)), 'wb'))
        #end
    #end
    
    ''' MODEL TEST '''
    if TEST:
        
        if RUNS == 1:
            saved_model = torch.load( open(os.path.join(PATH_MODEL, '{}.pkl'.format(MODEL_NAME)), 'rb') )
            trainer = saved_model['trainer']
            lit_model = saved_model['model']
            saved_at = saved_model['saved_at_time']
            name = saved_model['name']
            print('\nModel : {}, saved at {}'.format(name, saved_at))
        #end
        
        lit_model.eval()
        trainer.test(lit_model, test_loader)
        
        if RUNS == 1 and PLOTS:
            plot_UPA(lit_model.samples_to_save)
            if not TAYLOR_DS: plot_WS(lit_model.samples_to_save)
            plot_WS_scatter(lit_model.samples_to_save, 'y')
        #end
        
        u_data = lit_model.samples_to_save[0]['u_data'].detach().numpy()
        u_reco = lit_model.samples_to_save[0]['u_reco'].detach().numpy()
        
        pred_error_metric = NormLoss((u_data - u_reco), mask = None, divide = True, rmse = True)
        r2_metric = r2_score(u_data.reshape(-1,1), u_reco.reshape(-1,1))
        
        print('R² score = {:.4f}'.format(r2_metric))
        print('RMSE     = {:.4f}'.format(pred_error_metric))
        windspeed_rmses['only_UPA']['u'][run] = pred_error_metric
        
        predictions.append( u_reco )
    #end
#end

preds = torch.Tensor( np.median( np.array(predictions), axis = 0 ) )
wdata = torch.Tensor( u_data )
windspeed_baggr = NormLoss((preds - wdata), mask = None, divide = True, rmse = True)
windspeed_rmses['only_UPA']['aggr'] = windspeed_baggr


''' SERIALIZE THE HYPERPARAMETERS IN A JSON FILE, with the respective perf metric '''
hyperparams = {
    'EPOCHS'      : EPOCHS,
    'BATCH_SIZE'  : BATCH_SIZE,
    'LATENT_DIM'  : LATENT_DIM,
    'DROPOUT'     : DROPOUT,
    'WEIGHT_DATA' : WEIGHT_DATA,
    'WEIGHT_PRED' : WEIGHT_PRED,
    'MODEL_LR'    : MODEL_LR,
    'MODEL_WD'    : MODEL_WD,
    'WINDMAP_LR'  : WINDMAP_LR,
    'WINDMAP_WD'  : WINDMAP_WD,
    'PRED_ERROR'  : pred_error_metric.item(),
    'R_SQUARED'   : r2_metric.item()
}
with open(os.path.join(PATH_MODEL, 'HYPERPARAMS.json'), 'w') as filestream:
    json.dump(hyperparams, filestream, indent = 4)
#end
filestream.close()

pickle.dump(windspeed_rmses, open(os.path.join(os.getcwd(), 'Evaluation', '{}.pkl'.format(MODEL_NAME)), 'wb'))

with open( os.path.join(os.getcwd(), 'Evaluation', '{}.txt'.format(MODEL_NAME)), 'w' ) as f:
    f.write('Minimum    ; {:.4f}\n'.format(windspeed_rmses['only_UPA']['u'].min()))
    f.write('Mean ± std ; {:.4f} ± {:.4f}\n'.format(windspeed_rmses['only_UPA']['u'].mean(),
                                                  windspeed_rmses['only_UPA']['u'].std()))
    f.write('Median     ; {:.4f}\n'.format(windspeed_baggr))
f.close()

