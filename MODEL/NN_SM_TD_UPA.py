
print('\n\n')
print('###############################################')
print('NN SM TD UPA')
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
import json

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import r2_score

from tutls import NormLoss, RegressionNetwork
from dutls import MMData
from gutls import plot_WS, plot_WS_scatter

if torch.cuda.is_available():
    device = 'cuda'
    gpus = -1
else:
    device = 'cpu'
    gpus = 0
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
        data_UPA = batch[1].reshape(batch_size, FORMAT_SIZE, N).transpose(1,2)
        data_ws  = batch[3].reshape(batch_size, FORMAT_SIZE, Nu).transpose(1,2)
        
        '''Produce the masks according to the data sparsity patterns'''
        mask_UPA = torch.zeros_like(data_UPA)
        mask_UPA[data_UPA.isnan().logical_not()] = 1.
        mask_UPA[data_UPA == 0] = 0
        data_UPA[data_UPA.isnan()] = 0.
        
        if DATA_AUGMT and phase == 'train':
            '''Delete an arbitrary number of data, from 1 to 12 items per time step'''
            for i in range(batch_size):
                timesteps_indices = torch.randint(0, FORMAT_SIZE, ( torch.randint(1, 12, (1,)) ,))
                mask_UPA[i, timesteps_indices, :] = 0
                data_UPA[i, timesteps_indices, :] = 0
            #end
        #end
        
        mask_ws = torch.zeros_like(data_ws)
        mask_ws[data_ws.isnan().logical_not()] = 1.
        mask_ws[data_ws == 0] = 0
        data_ws[data_ws.isnan()] = 0.
        
        data_input = torch.cat( (data_UPA, 0. * data_ws), dim = 1 )
        outputs = self.network(data_input)
        reco_ws = outputs[:, -Nu:, :]
        
        data_ws = data_ws.transpose(1,2)
        reco_ws = reco_ws.transpose(1,2)
        mask_ws = mask_ws.transpose(1,2)
        
        if phase == 'test':
            
            data_ws  = self.undo_preprocess(data_ws,  self.preprocess_params['u'])
            reco_ws  = self.undo_preprocess(reco_ws,  self.preprocess_params['u'])
            outputs = { 'u_data' : data_ws,  'u_reco' : reco_ws }
            
            self.samples_to_save.append( outputs )
        #end
        
        loss_ws = NormLoss((data_ws - reco_ws),   mask = mask_ws,  divide = True)
        loss = WEIGHT_PRED * loss_ws
        
        return dict({'loss' : loss}), outputs
    #end
    
    def training_step(self, batch, batch_idx):
        
        metrics, outs = self.compute_loss(batch)
        loss = metrics['loss']
        
        self.log('loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        
        return loss
    #end
    
    def test_step(self, batch, batch_idx):
        
        if batch_idx >= 1:
            raise ValueError('Batch index greater that 0 : Reformat DataLoader')
        #end
        
        with torch.no_grad():
            metrics, outs = self.compute_loss(batch, phase = 'test')
            
            metrics['loss'] = np.sqrt(metrics['loss'].item())
            
            self.log('loss_test', metrics['loss'].item())
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
TRAIN       = True
TEST        = True

FORMAT_SIZE = 24
MODEL_NAME  = 'NN_SM_UPA_TD'
PATH_DATA   = os.getenv('PATH_DATA')
PATH_MODEL  = os.getenv('PATH_MODEL')

# HPARAMS
EPOCHS      = 200
BATCH_SIZE  = 32
DATA_AUGMT  = False
DROPOUT     = 0.
WEIGHT_DATA = 0.
WEIGHT_PRED = 1.
MODEL_LR    = 5e-4
MODEL_WD    = 1e-6


'''
Initialize the performance metrics data structures
'''
windspeed_rmses = {
        'only_SAR'  : {'u' : np.zeros(RUNS), 'u_x' : np.zeros(RUNS), 'u_y' : np.zeros(RUNS)},
        'only_UPA'  : {'u' : np.zeros(RUNS), 'u_x' : np.zeros(RUNS), 'u_y' : np.zeros(RUNS)},
        'colocated' : {'u' : np.zeros(RUNS), 'u_x' : np.zeros(RUNS), 'u_y' : np.zeros(RUNS)},
}
predictions = list()

for run in range(RUNS):
    print('Run {}'.format(run))
    
    train_set = MMData(os.path.join(PATH_DATA, 'train'), WIND_VALUES, '2011')
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8)
    
    test_set = MMData(os.path.join(PATH_DATA, 'test_only_UPA'), WIND_VALUES, '2011')
    test_loader = DataLoader(test_set, batch_size = test_set.__len__(), shuffle = False, num_workers = 8)
    
    N = train_set.get_modality_data_size('y')
    Nu = train_set.get_modality_data_size('u')
    
    ''' MODEL TRAIN '''
    if TRAIN:
        network = RegressionNetwork(
                nn.Sequential(
                    nn.Conv1d(N + Nu, 128, kernel_size = 3, padding = 'same'),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(128, N + Nu, kernel_size = 3, padding = 'same')
                )
        )
        
        profiler_kwargs = {'max_epochs' : EPOCHS, 'log_every_n_steps' : 1}
        
        lit_model = LitModel(network, preprocess_params = test_set.preprocess_params)
        trainer = pl.Trainer(**profiler_kwargs)
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
            plot_WS(lit_model.samples_to_save)
            plot_WS_scatter(lit_model.samples_to_save, 'y')
        #end
        
        u_data = lit_model.samples_to_save[0]['u_data'].cpu().detach().numpy()
        u_reco = lit_model.samples_to_save[0]['u_reco'].cpu().detach().numpy()
        
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
    'DATA_AUGMT'  : DATA_AUGMT,
    'DROPOUT'     : DROPOUT,
    'WEIGHT_PRED' : WEIGHT_PRED,
    'MODEL_LR'    : MODEL_LR,
    'MODEL_WD'    : MODEL_WD,
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