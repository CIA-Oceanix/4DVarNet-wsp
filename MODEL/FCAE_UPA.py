
print('\n\n')
print('###############################################')
print('FC-AE SM UPA')
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
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import json

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import r2_score

from tutls import NormLoss, xavier_weights_initialization
from dutls import SMData
from gutls import plot_UPA, plot_WS, plot_WS_scatter

if torch.cuda.is_available():
    device = 'cuda'
    gpus = -1
else:
    device = 'cpu'
    gpus = 0
#end


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
        
        if TIMEDEP == 'TI':
            data_UPA = batch[0].reshape(batch_size * FORMAT_SIZE, N)
            data_ws  = batch[2].reshape(batch_size * FORMAT_SIZE, Nu)
        elif TIMEDEP == 'TD':
            data_UPA = batch[0].reshape(batch_size, FORMAT_SIZE, N)
            data_ws  = batch[2].reshape(batch_size, FORMAT_SIZE, Nu)
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
        
        if TIMEDEP == 'TI':
            
            reco_UPA = outputs[:, :N].reshape(batch_size, FORMAT_SIZE, N)
            reco_ws  = outputs[:, -Nu:].reshape(batch_size, FORMAT_SIZE, Nu)
            
            data_UPA = data_UPA.reshape(batch_size, FORMAT_SIZE, N)
            mask_UPA = mask_UPA.reshape(batch_size, FORMAT_SIZE, N)
            data_ws  = data_ws.reshape(batch_size, FORMAT_SIZE, Nu)
            mask_ws  = mask_ws.reshape(batch_size, FORMAT_SIZE, Nu)
            
        elif TIMEDEP == 'TD':
            
            reco_UPA = outputs[:, :, :N]
            reco_ws  = outputs[:, :, -Nu:]
        #end
                        
        if phase == 'test':
            
            data_UPA = self.undo_preprocess(data_UPA, self.preprocess_params['upa'])
            reco_UPA = self.undo_preprocess(reco_UPA, self.preprocess_params['upa'])
            data_ws  = self.undo_preprocess(data_ws,  self.preprocess_params['wind_situ'])
            reco_ws  = self.undo_preprocess(reco_ws,  self.preprocess_params['wind_situ'])
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
    
    def validation_step(self, batch, batch_idx):
        
        metrics, outs = self.compute_loss(batch)
        val_loss = metrics['loss_pred']
        
        self.log('val_loss', val_loss, on_step = False, on_epoch = True, prog_bar = False)        
        return val_loss
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
with open(os.path.join(os.getcwd(), 'cparams-fcae.json'), 'r') as filestream:
    CPARAMS = json.load(filestream)
filestream.close()

WIND_VALUES = CPARAMS['WIND_VALUES']
DATA_TITLE  = CPARAMS['DATA_TITLE']
TIMEDEP     = CPARAMS['TIMEDEP']

RUNS        = CPARAMS['RUNS']
EPOCHS      = CPARAMS['EPOCHS']
TRAIN       = CPARAMS['TRAIN']
TEST        = CPARAMS['TEST']
PLOTS       = CPARAMS['PLOTS']

BATCH_SIZE  = CPARAMS['BATCH_SIZE']
LATENT_DIM  = CPARAMS['LATENT_DIM']

DROPOUT     = CPARAMS['DROPOUT']
WEIGHT_DATA = CPARAMS['WEIGHT_DATA']
WEIGHT_PRED = CPARAMS['WEIGHT_PRED']
MODEL_LR    = CPARAMS['MODEL_LR']
MODEL_WD    = CPARAMS['MODEL_WD']

FORMAT_SIZE = 24
MODEL_NAME  = f'AE_SM_UPA_{TIMEDEP}'
PATH_DATA   = os.getenv('PATH_DATA')
PATH_MODEL  = os.getenv('PATH_MODEL')

PATH_MODEL = os.path.join(PATH_MODEL, MODEL_NAME)
if not os.path.exists(PATH_MODEL): os.mkdir(PATH_MODEL)


# Introduction
print('Experiment:')
print('----------------------------------------------------------------------')
print(f'Time dependence                   : {TIMEDEP}')
print(f'Runs                              : {RUNS}')
print(f'Path Target                       : {PATH_MODEL}')
print(f'Model                             : {MODEL_NAME}')
print('----------------------------------------------------------------------')


'''
Initialize the performance metrics data structures
'''
windspeed_rmses = {'u' : np.zeros(RUNS), 'aggr' : np.float32(0.)}
predictions = list()


for run in range(RUNS):
    print('\n------\nRun {}'.format(run))
    
    train_set = SMData(os.path.join(PATH_DATA, 'train'), WIND_VALUES, '2011')
    test_set  = SMData(os.path.join(PATH_DATA, 'test'), WIND_VALUES, '2011')
    val_set   = SMData(os.path.join(PATH_DATA, 'val'), WIND_VALUES, '2011')
        
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)#, num_workers = 8)
    val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False)#, num_workers = 8)
    test_loader  = DataLoader(test_set, batch_size = test_set.__len__(), shuffle = False)#, num_workers = 8)
    
    N  = train_set.get_modality_data_size('upa')
    Nu = train_set.get_modality_data_size('wind_situ')
    
    ''' MODEL TRAIN '''
    if TRAIN:
        
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
        
        network = AutoEncoder(
                encoder = encoder,
                decoder = decoder       
        )
        
        profiler_kwargs = {'max_epochs' : EPOCHS, 'log_every_n_steps' : 1, 'gpus' : gpus}
        
        lit_model = LitModel(network, preprocess_params = test_set.preprocess_params)
        trainer = pl.Trainer(**profiler_kwargs)
        trainer.fit(lit_model, train_loader, val_loader)
        
        # torch.save({'trainer' : trainer, 'model' : lit_model, 
        #             'name' : MODEL_NAME, 'saved_at_time' : datetime.now()},
        #            open(os.path.join(PATH_MODEL, '{}.pkl'.format(MODEL_NAME)), 'wb'))
    #end
    
    ''' MODEL TEST '''
    if TEST:
        
        # saved_model = torch.load( open(os.path.join(PATH_MODEL, '{}.pkl'.format(MODEL_NAME)), 'rb') )
        # trainer = saved_model['trainer']
        # lit_model = saved_model['model']
        # saved_at = saved_model['saved_at_time']
        # name = saved_model['name']
        # print('\nModel : {}, saved at {}'.format(name, saved_at))
        
        lit_model.eval()
        trainer.test(lit_model, test_loader)
        
        if RUNS == 1 and PLOTS:
            plot_UPA(lit_model.samples_to_save)
            plot_WS(lit_model.samples_to_save)
            plot_WS_scatter(lit_model.samples_to_save, 'y')
        #end
        
        u_data = lit_model.samples_to_save[0]['u_data'].cpu().detach().numpy()
        u_reco = lit_model.samples_to_save[0]['u_reco'].cpu().detach().numpy()
        
        pred_error_metric = NormLoss((u_data - u_reco), mask = None, divide = True, rmse = True)
        r2_metric = r2_score(u_data.reshape(-1,1), u_reco.reshape(-1,1))
        
        print('R² score = {:.4f}'.format(r2_metric))
        print('RMSE     = {:.4f}'.format(pred_error_metric))
        windspeed_rmses['u'][run] = pred_error_metric
        
        predictions.append( u_reco )
    #end
#end

preds = torch.Tensor( np.median( np.array(predictions), axis = 0 ) )
wdata = torch.Tensor( u_data )
windspeed_baggr = NormLoss((preds - wdata), mask = None, divide = True, rmse = True)
windspeed_rmses['aggr'] = windspeed_baggr


with open(os.path.join(PATH_MODEL, 'HYPERPARAMS.json'), 'w') as filestream:
    json.dump(CPARAMS, filestream, indent = 4)
#end
filestream.close()

pickle.dump(windspeed_rmses, open(os.path.join(os.getcwd(), 'Evaluation', '{}.pkl'.format(MODEL_NAME)), 'wb'))

with open( os.path.join(os.getcwd(), 'Evaluation', '{}.txt'.format(MODEL_NAME)), 'w' ) as f:
    f.write('Minimum    ; {:.4f}\n'.format(windspeed_rmses['u'].min()))
    f.write('Mean ± std ; {:.4f} ± {:.4f}\n'.format(windspeed_rmses['u'].mean(),
                                                  windspeed_rmses['u'].std()))
    f.write('Median     ; {:.4f}\n'.format(windspeed_baggr))
f.close()

