
# CHECKPOINT

print('\n\n')
print('###############################################')
print('                4DVAR SM TD UPA                ')
print('###############################################')
print('\n\n')

import os
import sys
import glob
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
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.metrics import r2_score

from tutls import L2NormLoss, NormLoss, xavier_weights_initialization
from dutls import SMData
from gutls import plot_UPA, plot_WS, plot_WS_scatter
import solver as NN_4DVar

if torch.cuda.is_available():
    device = 'cuda'
    gpus = -1
else:
    device = 'cpu'
    gpus = 0
#end



class AutoEncoder(nn.Module):
    
    def __init__(self, encoder, decoder, ):
        
        super(AutoEncoder, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.encoder.apply(xavier_weights_initialization)
    #end
    
    def forward(self, data):
        
        latent = self.encoder(data)
        reco = self.decoder(latent)
        return reco
    #end
#end


class ConvNet(nn.Module):
    
    def __init__(self, net):
        
        super(ConvNet, self).__init__()
        
        self.net = net
        self.net.apply(xavier_weights_initialization)
    #end
    
    def forward(self, data):
        
        reco = self.net(data)
        return reco
    #end
#end



class Model_H(torch.nn.Module):
    
    def __init__(self, shape_data, dim = 1):
        
        super(Model_H, self).__init__()
        self.dim_obs = 1
        self.dim_obs_channel = np.array([shape_data[0], dim])
    #end
    
    def forward(self, x, y, mask):
        
        dyout = (x - y).mul(mask)
        return dyout
    #end
#end


class LitModel(pl.LightningModule):
    
    def __init__(self, Phi, shapeData, preprocess_params):
        
        super(LitModel, self).__init__()
    
        
        self.Phi = Phi
        self.hparams.dim_grad_solver = DIM_LSTM
        self.hparams.dropout = DROPOUT
        self.hparams.n_solver_iter = N_SOL_ITER
        self.hparams.n_fourdvar_iter = N_4DV_ITER  # con n = 1 torniamo alla versione plain
        self.hparams.automatic_optimization = True
        self.preprocess_params = preprocess_params
        self.test_rmse = np.float64(0.)
        self.samples_to_save = list()
        self.train_losses = np.zeros(EPOCHS)
        self.val_losses = np.zeros(EPOCHS)
        
        if MM_ECMWF:
            mod_shape_data = [Nupa + Necmwf + Nsitu, FORMAT_SIZE]
        else:
            mod_shape_data = [Nupa + Nsitu, FORMAT_SIZE]
        #end
        
        self.model = NN_4DVar.Solver_Grad_4DVarNN(  # Instantiation of the LSTM solver
            
            Phi,                              # Dynamical prior   
            Model_H(shapeData),               # Observation operator
            NN_4DVar.model_GradUpdateLSTM(    # m_Grad
                mod_shape_data,               # m_Grad : Shape Data
                False,                           # m_Grad : Use Periodic Bnd
                self.hparams.dim_grad_solver,    # m_Grad : Dim LSTM
                self.hparams.dropout,            # m_Grad : Dropout
                False                            # m_Grad : Stochastic
            ),
            L2NormLoss(),                     # Norm obserbvations
            L2NormLoss(),                     # Norm prior
            shapeData,                        # Shape data
            self.hparams.n_solver_iter        # Solver iterations
            )
    #end
    
    def forward(self, input_data, iterations = None, phase = 'train'):
        
        state_init = None
        
        for i in range(self.hparams.n_fourdvar_iter):
            
            loss, outs = self.compute_loss(input_data, phase = phase, state_init = state_init)
            state_init = outs.detach()
        #end
        
        return loss, outs
    #end
    
    def configure_optimizers(self):
        
        optimizers = optim.Adam([ 
                                  {'params'       : self.model.model_Grad.parameters(),
                                   'lr'           : SOLVER_LR,
                                   'weight_decay' : SOLVER_WD},
                                  {'params'       : self.model.Phi.parameters(),
                                   'lr'           : PHI_LR,
                                   'weight_decay' : PHI_WD}
                                ])
        return optimizers
    #end
    
    def get_init_state(self, batch, state):
        
        if state is not None:
            return state
        #end
        
        batch_size = batch[0].shape[0]
        data_UPA = batch[0].reshape(batch_size, FORMAT_SIZE, Nupa)
        data_we  = batch[1].reshape(batch_size, FORMAT_SIZE, Necmwf)
        data_ws  = batch[2].reshape(batch_size, FORMAT_SIZE, Nsitu)
        data_UPA = data_UPA.transpose(1, 2)
        data_we  = data_we.transpose(1, 2)
        data_ws  = data_ws.transpose(1, 2)
        
        data_UPA[data_UPA.isnan()] = 0.
        data_ws[data_ws.isnan()] = 0.
        data_we[data_we.isnan()] = 0.
        
        if not MM_ECMWF:
            state = torch.cat( (data_UPA, 0. * data_ws), dim = 1)
        else:
            state = torch.cat( (data_UPA, data_we, 0. * data_ws), dim = 1 )
        #end
        
        return state
    #end
    
    def compute_loss(self, batch, phase = 'train', state_init = None):
        
        '''Reshape the input data'''
        batch_size = batch[0].shape[0]
        data_UPA = batch[0].reshape(batch_size, FORMAT_SIZE, Nupa).clone()
        data_we  = batch[1].reshape(batch_size, FORMAT_SIZE, Necmwf).clone()
        data_ws  = batch[2].reshape(batch_size, FORMAT_SIZE, Nsitu).clone()
        data_UPA = data_UPA.transpose(1, 2)
        data_we  = data_we.transpose(1, 2)
        data_ws  = data_ws.transpose(1, 2)
        
        '''Produce the masks according to the data sparsity patterns'''
        mask_UPA = torch.zeros_like(data_UPA)
        mask_UPA[data_UPA.isnan().logical_not()] = 1.
        mask_UPA[data_UPA == 0] = 0
        data_UPA[data_UPA.isnan()] = 0.
        
        mask_we = torch.zeros_like(data_we)
        mask_we[data_we.isnan().logical_not()] = 1.
        mask_we[data_we == 0] = 0
        data_we[data_we.isnan()] = 0.
        
        mask_ws = torch.zeros_like(data_ws)
        mask_ws[data_ws.isnan().logical_not()] = 1.
        mask_ws[data_ws == 0] = 0
        data_ws[data_ws.isnan()] = 0.
        
        '''Aggregate UPA and wind speed data in a single tensor
           This is done with an horizontal concatenation'''
        if not MM_ECMWF:
            data_input = torch.cat( (data_UPA, 0. * data_ws), dim = 1 )
            mask_input = torch.cat( (mask_UPA, mask_ws), dim = 1 )
        else:
            data_input = torch.cat( (data_UPA, data_we, 0. * data_ws), dim = 1 )
            mask_input = torch.cat( (mask_UPA, mask_we, mask_ws), dim = 1 )
        #end
        
        inputs_init = self.get_init_state(batch, state_init)
        
        with torch.set_grad_enabled(True):
            
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad = True)
            
            if FIXED_POINT:
                outputs = self.Phi(data_input) # Fixed point
            else:
                outputs, hidden, cell, normgrad = self.model(inputs_init, data_input, mask_input)                
            #end
            
            '''Split UPA and windspeed reconstructions and predictions and 
               reinstate them in the ``(batch_size, time_series_length, num_features)`` format'''
            reco_UPA = outputs[:, :Nupa, :]
            reco_we  = outputs[:, Nupa : Nupa + Necmwf, :]
            reco_ws  = outputs[:, -Nsitu:, :]
            
            data_UPA = data_UPA.transpose(2, 1)
            reco_UPA = reco_UPA.transpose(2, 1) 
            mask_UPA = mask_UPA.transpose(2, 1) 
            
            data_we = data_we.transpose(2, 1)
            reco_we = reco_we.transpose(2, 1)
            mask_we = mask_we.transpose(2, 1)
            
            data_ws = data_ws.transpose(2, 1)
            reco_ws = reco_ws.transpose(2, 1)
            mask_ws = mask_ws.transpose(2, 1)
            
            if phase == 'test' or phase == 'val':
                
                '''If test, then denormalize the data and append them in a list
                   so to plot them in the end'''
                data_UPA = self.undo_preprocess(data_UPA, self.preprocess_params['upa'])
                reco_UPA = self.undo_preprocess(reco_UPA, self.preprocess_params['upa'])
                data_we  = self.undo_preprocess(data_we,  self.preprocess_params['wind_ecmwf'])
                reco_we  = self.undo_preprocess(reco_we,  self.preprocess_params['wind_ecmwf'])
                data_ws  = self.undo_preprocess(data_ws,  self.preprocess_params['wind_situ'])
                reco_ws  = self.undo_preprocess(reco_ws,  self.preprocess_params['wind_situ'])
                
                '''Recreate the outputs variable'''
                outputs = { 'y_data' : data_UPA, 'y_reco' : reco_UPA,
                            'u_data' : data_ws,  'u_reco' : reco_ws,
                            'w_data' : data_we,  'w_reco' : reco_we
                }
                
                if phase == 'test':
                    self.samples_to_save.append( outputs )
                #end
            #end
            
            '''Loss computation. Note the use of the masks and see the ``NormLoss`` documentation in
               the devoted module'''
            loss_data = NormLoss((data_UPA - reco_UPA), mask = mask_UPA, divide = True, dformat = 'mtn')
            loss_ws = NormLoss((data_ws - reco_ws), mask = mask_ws, divide = True, dformat = 'mtn')
            
            if MM_ECMWF:
                loss_we = NormLoss((data_we - reco_we), mask = mask_we, divide = True, dformat = 'mtn')
                loss = WEIGHT_DATA * loss_data + WEIGHT_WE * loss_we + WEIGHT_PRED * loss_ws
                return dict({'loss' : loss, 'loss_reco' : loss_data, 
                             'loss_we' : loss_we, 'loss_pred' : loss_ws}), outputs
            else:
                loss = WEIGHT_DATA * loss_data + WEIGHT_PRED * loss_ws
                return dict({'loss' : loss, 'loss_reco' : loss_data, 
                             'loss_pred' : loss_ws}), outputs
                #end
            #end                
        #end
    #end
    
    def training_step(self, batch, batch_idx):
        
        metrics, outs = self.compute_loss(batch)
        loss = metrics['loss']
        loss_reco = metrics['loss_reco']
        loss_pred = metrics['loss_pred']
        
        self.log('loss', loss,           on_step = True, on_epoch = True, prog_bar = True)
        self.log('loss_reco', loss_reco, on_step = True, on_epoch = True, prog_bar = True)
        self.log('loss_pred', loss_pred, on_step = True, on_epoch = True, prog_bar = True)
        
        if MM_ECMWF:
            loss_we   = metrics['loss_we']
            self.log('loss_we',   loss_we,   on_step = True, on_epoch = True, prog_bar = True)
        #end
        
        return loss
    #end
    
    def training_epoch_end(self, outputs):
        
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        self.train_losses[self.current_epoch] = loss
        if device == 'cpu':
            print(f'Epoch {self.current_epoch} : Loss = {loss:.4f}')
        #end
    #end
    
    def validation_step(self, batch, batch_idx):
        
        metrics, outs = self.compute_loss(batch)
        val_loss = metrics['loss']
        self.log('val_loss', val_loss)
        
        return val_loss
    #end
    
    def validation_epoch_end(self, outputs):
        
        loss = torch.stack([out for out in outputs]).mean()
        self.val_losses[self.current_epoch] = loss
    #end
    
    def test_step(self, batch, batch_idx):
        
        if batch_idx >= 1:
            raise ValueError('Batch index greater than 0 : Reformat DataLoader')
        #end
        
        with torch.no_grad():
            metrics, outs = self.compute_loss(batch, phase = 'test')
            
            metrics['loss']      = np.sqrt(metrics['loss'].item())
            metrics['loss_reco'] = np.sqrt(metrics['loss_reco'].item())            
            metrics['loss_pred'] = np.sqrt(metrics['loss_pred'].item())
            
            self.log('loss_test',      metrics['loss'].item())
            self.log('loss_reco_test', metrics['loss_reco'].item())            
            self.log('loss_pred_test', metrics['loss_pred'].item())
            
            if MM_ECMWF:
                metrics['loss_we']   = np.sqrt(metrics['loss_we'].item())
                self.log('loss_we_test',   metrics['loss_we'].item())
            #end
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
MM_ECMWF    = False
PLOTS       = False
RUNS        = 10
COLOCATED   = False
TRAIN       = True
TEST        = True
FIXED_POINT = False
LOAD_CKPT   = False
PRIOR       = 'AE'

FORMAT_SIZE = 24
MODEL_NAME  = '4DVAR'
PATH_DATA   = os.getenv('PATH_DATA')
PATH_MODEL  = os.getenv('PATH_MODEL')

# HPARAMS
EPOCHS      = 200
BATCH_SIZE  = 32
LATENT_DIM  = 20
DIM_LSTM    = 100
N_SOL_ITER  = 5
N_4DV_ITER  = 1
DROPOUT     = 0.
WEIGHT_DATA = 0.5
WEIGHT_PRED = 1.5
WEIGHT_WE   = 1
SOLVER_LR   = 1e-3
SOLVER_WD   = 1e-5
PHI_LR      = 1e-3
PHI_WD      = 1e-5
NSOL_IT_REF = 5

print(f'Prior       : {PRIOR}')
print(f'Fixed point : {FIXED_POINT}\n\n')
MODEL_NAME  = f'{MODEL_NAME}_{PRIOR}'

if MM_ECMWF:
    MODEL_NAME = f'{MODEL_NAME}_MM'
else:
    MODEL_NAME = f'{MODEL_NAME}_SM'
#end

if FIXED_POINT:
    MODEL_NAME = f'{MODEL_NAME}_fp1it'
else:
    MODEL_NAME = f'{MODEL_NAME}_gs{NSOL_IT_REF}it'
#end

if LOAD_CKPT:
    MODEL_SOURCE = MODEL_NAME
    MODEL_NAME = f'{MODEL_NAME}_lckpt'
else:
    MODEL_SOURCE = MODEL_NAME
#end

PATH_SOURCE = os.path.join(PATH_MODEL, MODEL_SOURCE)   
PATH_MODEL = os.path.join(PATH_MODEL, MODEL_NAME)
if not os.path.exists(PATH_SOURCE) and LOAD_CKPT: os.mkdir(PATH_SOURCE)
if not os.path.exists(PATH_MODEL): os.mkdir(PATH_MODEL)

'''
Initialize the performance metrics data structures
'''
windspeed_rmses = {
        'only_SAR'  : {'u' : np.zeros(RUNS), 'u_x' : np.zeros(RUNS), 'u_y' : np.zeros(RUNS)},
        'only_UPA'  : {'u' : np.zeros(RUNS), 'u_c' : np.zeros(RUNS)},
        'colocated' : {'u' : np.zeros(RUNS), 'u_x' : np.zeros(RUNS), 'u_y' : np.zeros(RUNS)},
}
predictions = list()
performance_metrics = {
        'train_loss' : np.zeros((EPOCHS, RUNS)),
        'val_loss'   : np.zeros((EPOCHS, RUNS))
}


for run in range(RUNS):
    print('Run {}'.format(run))
    
    train_set = SMData(os.path.join(PATH_DATA, 'train'), WIND_VALUES, '2011')
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8)
    
    val_set = SMData(os.path.join(PATH_DATA, 'val'), WIND_VALUES, '2011')
    val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 8)
    
    test_set = SMData(os.path.join(PATH_DATA, 'test'), WIND_VALUES, '2011')
    test_loader = DataLoader(test_set, batch_size = test_set.__len__(), shuffle = False, num_workers = 8)
    
    Nupa = train_set.get_modality_data_size('upa')
    Necmwf = train_set.get_modality_data_size('wind_ecmwf')
    Nsitu = train_set.get_modality_data_size('wind_situ')
    
    ''' MODEL TRAIN '''
    if TRAIN:
        
        if MM_ECMWF:
            N_data = Nupa + Necmwf + Nsitu
        else:
            N_data = Nupa + Nsitu
        #end
        
        if PRIOR == 'AE':
            
            encoder = torch.nn.Sequential(
                nn.Conv1d(N_data, 128, kernel_size = 3, padding = 'same'),
                nn.Dropout(DROPOUT),
                nn.LeakyReLU(0.1),
                nn.Conv1d(128, LATENT_DIM, kernel_size = 3, padding = 'same'),
                nn.Dropout(DROPOUT)
            )
            decoder = torch.nn.Sequential(
                nn.Conv1d(LATENT_DIM, 128, kernel_size = 3, padding = 'same'),
                nn.Dropout(DROPOUT),
                nn.LeakyReLU(0.1),
                nn.Conv1d(128, N_data, kernel_size = 3, padding = 'same'),
                nn.Dropout(DROPOUT),
                nn.LeakyReLU(0.1)
            )
            
            Phi = AutoEncoder(encoder, decoder)
            
        elif PRIOR == 'CN':
            
            Phi = ConvNet(
                nn.Sequential(
                    nn.Conv1d(N_data, 128, kernel_size = 3, padding = 'same'),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(128, N_data, kernel_size = 3, padding = 'same')
                )
            )
        #end
        
        if MM_ECMWF:
            shape_data = (BATCH_SIZE, Nupa + Necmwf, FORMAT_SIZE)
        else:
            shape_data = (BATCH_SIZE, Nupa, FORMAT_SIZE)
        #end
        
        lit_model = LitModel( Phi, shapeData = shape_data,
                              preprocess_params = test_set.preprocess_params
        )
        
        if LOAD_CKPT:
            
            CKPT_NAME = glob.glob(os.path.join(PATH_SOURCE, f'{run}-' + MODEL_SOURCE + '-epoch=*.ckpt'))[0]
            checkpoint_model = open(CKPT_NAME, 'rb')
            print(CKPT_NAME)
            lit_model_state_dict = torch.load(checkpoint_model)['state_dict']
            lit_model.load_state_dict(lit_model_state_dict)
            
            name_append = '_second'
        else:
            name_append = ''
        #end
        
        profiler_kwargs = {'max_epochs' : EPOCHS, 'log_every_n_steps' : 1, 'gpus' : gpus}
        
        model_checkpoint = ModelCheckpoint(
                monitor = 'val_loss',
                dirpath = PATH_MODEL,
                filename = f'{run}-' + MODEL_NAME + '-{epoch:02d}' + name_append,
                save_top_k = 1,
                mode = 'min'
        )
        trainer = pl.Trainer(**profiler_kwargs, callbacks = [model_checkpoint])
        
        trainer.fit(lit_model, train_loader, val_loader)
        performance_metrics['train_loss'][:, run] = lit_model.train_losses
        performance_metrics['val_loss'][:, run] = lit_model.val_losses
        
        # torch.save({'trainer' : trainer, 'model' : lit_model, 
        #             'name' : MODEL_NAME, 'saved_at_time' : datetime.now()},
        #             open(os.path.join(PATH_MODEL, f'{MODEL_NAME}.pkl'), 'wb')
        # )
    #end
    
    ''' MODEL TEST '''
    if TEST:
        
        # saved_model = torch.load( open(os.path.join(PATH_MODEL, f'{MODEL_NAME}.pkl'), 'rb') )
        # trainer = saved_model['trainer']
        # lit_model = saved_model['model']
        # saved_at = saved_model['saved_at_time']
        # name = saved_model['name']
        # print(f'\nModel : {name}, saved at {saved_at}')
        CKPT_NAME = glob.glob(os.path.join(PATH_MODEL, f'{run}-' + MODEL_NAME + '-epoch=*.ckpt'))[0]
        print(CKPT_NAME)
        checkpoint_model = open(CKPT_NAME, 'rb')
        lit_model_state_dict = torch.load(checkpoint_model)['state_dict']
        lit_model.load_state_dict(lit_model_state_dict)
        
        lit_model.eval()
        lit_model.Phi.eval()
        lit_model.model.eval()
        trainer.test(lit_model, test_loader)
        
        if RUNS == 1 and PLOTS:
            plot_UPA(lit_model.samples_to_save)
            plot_WS(lit_model.samples_to_save)
            plot_WS_scatter(lit_model.samples_to_save, 'y')
        #end
        
        u_data = lit_model.samples_to_save[0]['u_data'].cpu().detach().numpy()
        u_reco = lit_model.samples_to_save[0]['u_reco'].cpu().detach().numpy()
        
        pred_error_metric = NormLoss((u_data - u_reco), mask = None, divide = True, rmse = True)
        mask_central = torch.zeros((u_data.shape)); mask_central[:, FORMAT_SIZE // 2, :] = 1
        pred_error_metric_central = NormLoss((u_data - u_reco), mask = mask_central, divide = True, rmse = True)
        r2_metric = r2_score(u_data.reshape(-1,1), u_reco.reshape(-1,1))
        
        print('R² score = {:.4f}'.format(r2_metric))
        print('RMSE     = {:.4f}'.format(pred_error_metric))
        windspeed_rmses['only_UPA']['u'][run] = pred_error_metric
        windspeed_rmses['only_UPA']['u_c'][run] = pred_error_metric_central
        
        predictions.append( u_reco )
    #end
#end


''' Median of the prediction to produce the voted-by-models prediction '''
preds = torch.Tensor( np.median( np.array(predictions), axis = 0 ) )
wdata = torch.Tensor( u_data )
windspeed_baggr = NormLoss((preds - wdata), mask = None, divide = True, rmse = True)
windspeed_rmses['only_UPA']['aggr'] = windspeed_baggr


''' SERIALIZE THE HYPERPARAMETERS IN A JSON FILE, with the respective perf metric '''
hyperparams = {
    'EPOCHS'      : EPOCHS,
    'BATCH_SIZE'  : BATCH_SIZE,
    'LATENT_DIM'  : LATENT_DIM,
    'DIM_LSTM'    : DIM_LSTM,
    'N_SOL_ITER'  : N_SOL_ITER,
    'N_4DV_ITER'  : N_4DV_ITER,
    'DROPOUT'     : DROPOUT,
    'WEIGHT_DATA' : WEIGHT_DATA,
    'WEIGHT_PRED' : WEIGHT_PRED,
    'SOLVER_LR'   : SOLVER_LR,
    'SOLVER_WD'   : SOLVER_WD,
    'PHI_LR'      : PHI_LR,
    'PHI_WD'      : PHI_WD,
    'PRED_ERROR'  : pred_error_metric.item(),
    'R_SQUARED'   : r2_metric.item()
}
with open(os.path.join(PATH_MODEL, 'HYPERPARAMS.json'), 'w') as filestream:
    json.dump(hyperparams, filestream, indent = 4)
#end
filestream.close()

with open(os.path.join(os.getcwd(), 'Evaluation', f'{MODEL_NAME}_wsm.pkl'), 'wb') as filename:
    pickle.dump(windspeed_rmses, filename)
filename.close()

with open(os.path.join(os.getcwd(), 'Evaluation', f'{MODEL_NAME}_perfmetrics.pkl'), 'wb') as filestream:
    pickle.dump(performance_metrics, filestream)
filestream.close()

with open( os.path.join(os.getcwd(), 'Evaluation', f'{MODEL_NAME}.txt'), 'w' ) as f:
    f.write('Minimum          ; {:.4f}\n'.format(windspeed_rmses['only_UPA']['u'].min()))
    f.write('(all) Mean ± std ; {:.4f} ± {:.4f}\n'.format(windspeed_rmses['only_UPA']['u'].mean(),
                                                  windspeed_rmses['only_UPA']['u'].std()))
    f.write('(cen) Mean ± std ; {:.4f} ± {:.4f}\n'.format(windspeed_rmses['only_UPA']['u_c'].mean(),
                                                  windspeed_rmses['only_UPA']['u_c'].std()))
    f.write('Median           ; {:.4f}\n'.format(windspeed_baggr))
f.close()
