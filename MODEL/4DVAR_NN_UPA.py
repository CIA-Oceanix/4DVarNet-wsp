
print('\n\n')
print('###############################################')
print('4DVAR SM TD UPA')
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
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import r2_score

from tutls import L2NormLoss, NormLoss, xavier_weights_initialization
from dutls import MMData
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
        self.__trained = False
        self.loss_val = np.float64(0.)
        
        self.model = NN_4DVar.Solver_Grad_4DVarNN(  # Instantiation of the LSTM solver
            
            Phi,                              # Dynamical prior   
            Model_H(shapeData),               # Observation operator
            NN_4DVar.model_GradUpdateLSTM(    # m_Grad
                [N + Nu, FORMAT_SIZE],           # m_Grad : Shape Data
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
    
    def set_trained(self):
        self.__trained = True
    #end
    
    def is_trained(self):
        return self.__trained
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
        data_UPA = batch[1].reshape(batch_size, FORMAT_SIZE, N)
        data_ws  = batch[3].reshape(batch_size, FORMAT_SIZE, Nu)
        data_UPA = data_UPA.transpose(1, 2)
        data_ws  = data_ws.transpose(1, 2)
        
        state = torch.cat( (data_UPA, 0. * data_ws), dim = 1 )
        return state
    #end
    
    def compute_loss(self, batch, phase = 'train', state_init = None):
        
        '''Reshape the input data'''
        batch_size = batch[0].shape[0]
        data_UPA = batch[1].reshape(batch_size, FORMAT_SIZE, N).clone()
        data_ws  = batch[3].reshape(batch_size, FORMAT_SIZE, Nu).clone()
        data_UPA = data_UPA.transpose(1, 2)
        data_ws  = data_ws.transpose(1, 2)
        
        '''Produce the masks according to the data sparsity patterns'''
        mask_UPA = torch.zeros_like(data_UPA)
        mask_UPA[data_UPA.isnan().logical_not()] = 1.
        mask_UPA[data_UPA == 0] = 0
        data_UPA[data_UPA.isnan()] = 0.
        
        mask_ws = torch.zeros_like(data_ws)
        mask_ws[data_ws.isnan().logical_not()] = 1.
        mask_ws[data_ws == 0] = 0
        data_ws[data_ws.isnan()] = 0.
        
        '''Aggregate UPA and wind speed data in a single tensor
           This is done with an horizontal concatenation'''
        data_input = torch.cat( (data_UPA, 0. * data_ws), dim = 1 )
        mask_input = torch.cat( (mask_UPA, mask_ws), dim = 1 )
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
            reco_UPA = outputs[:, :N, :]
            reco_ws  = outputs[:, -Nu:, :]
            
            data_UPA = data_UPA.transpose(2, 1); data_ws = data_ws.transpose(2, 1)
            reco_UPA = reco_UPA.transpose(2, 1); reco_ws = reco_ws.transpose(2, 1)
            mask_UPA = mask_UPA.transpose(2, 1); mask_ws = mask_ws.transpose(2, 1)
            
            if phase == 'test' or phase == 'val':
                
                '''If test, then denormalize the data and append them in a list
                   so to plot them in the end'''
                data_UPA = self.undo_preprocess(data_UPA, self.preprocess_params['y'])
                reco_UPA = self.undo_preprocess(reco_UPA, self.preprocess_params['y'])
                data_ws  = self.undo_preprocess(data_ws,  self.preprocess_params['u'])
                reco_ws  = self.undo_preprocess(reco_ws,  self.preprocess_params['u'])
                
                '''Recreate the outputs variable'''
                outputs = { 'y_data' : data_UPA, 'y_reco' : reco_UPA,
                            'u_data' : data_ws,  'u_reco' : reco_ws
                }
                
                if phase == 'test':
                    self.samples_to_save.append( outputs )
                #end
            #end
            
            '''Loss computation. Note the use of the masks and see the ``NormLoss`` documentation in
               the devoted module'''
            loss_data = NormLoss((data_UPA - reco_UPA), mask = mask_UPA, divide = True, dformat = 'mtn')
            loss_ws = NormLoss((data_ws - reco_ws), mask = mask_ws, divide = True, dformat = 'mtn')
            loss = WEIGHT_DATA * loss_data + WEIGHT_PRED * loss_ws
        #end
        
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

# Two decisions that influence the program flow
FIXED_POINT = False

# Minor decision
SAVE_OUTS  = True
SAVE_TRAIN = False
LOAD_TEST  = False
SAVE_CKPT  = False
LOAD_CKPT  = True
WHAT_CKPT  = 'last'

if SAVE_CKPT and LOAD_CKPT:
    raise ValueError('Not save and load checkpoints at the same time')
#end

if SAVE_CKPT:
    
    if FIXED_POINT:
        raise ValueError('No fixed point with checkpoints streams')
    #end
    
    if WHAT_CKPT == 'best':
        SAVE_TOP_K_CKPT = 1
        SAVE_LAST_CKPT = None
    elif WHAT_CKPT == 'last':
        SAVE_TOP_K_CKPT = 0
        SAVE_LAST_CKPT = True
    #end
#end    

FORMAT_SIZE = 24
MODEL_NAME  = '4DVAR_SM_UPA_TD'
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
SOLVER_LR   = 1e-3
SOLVER_WD   = 1e-5
PHI_LR      = 1e-3
PHI_WD      = 1e-5
PRIOR       = 'AE'


print('Prior        : {}'.format(PRIOR))
print('Fixed point  : {}\n\n'.format(FIXED_POINT))
MODEL_NAME  = '{}_{}'.format(MODEL_NAME, PRIOR)
if FIXED_POINT:
    MODEL_NAME = '{}_fp1it'.format(MODEL_NAME)
else:
    MODEL_NAME = '{}_gs{}it'.format(MODEL_NAME, N_SOL_ITER)
#end
PATH_MODEL_SOURCE = os.path.join(PATH_MODEL, MODEL_NAME)

if LOAD_CKPT:
    if WHAT_CKPT == 'best':
        MODEL_NAME = '{}_best'.format(MODEL_NAME)
    elif WHAT_CKPT == 'last':
        MODEL_NAME = '{}_last'.format(MODEL_NAME)
    #end
#end

PATH_MODEL = os.path.join(PATH_MODEL, MODEL_NAME)
if not os.path.exists(PATH_MODEL): os.mkdir(PATH_MODEL)

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
    
    val_set = MMData(os.path.join(PATH_DATA, 'val'), WIND_VALUES, '2011')
    val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 8)
    
    N = train_set.get_modality_data_size('y')
    Nu = train_set.get_modality_data_size('u')
    
    ''' MODEL TRAIN '''
    if TRAIN:
        
        if PRIOR == 'AE':
            
            encoder = torch.nn.Sequential(
                nn.Conv1d(N + Nu, 128, kernel_size = 3, padding = 'same'),
                nn.Dropout(DROPOUT),
                nn.LeakyReLU(0.1),
                nn.Conv1d(128, LATENT_DIM, kernel_size = 3, padding = 'same'),
                nn.Dropout(DROPOUT)
            )
            decoder = torch.nn.Sequential(
                nn.Conv1d(LATENT_DIM, 128, kernel_size = 3, padding = 'same'),
                nn.Dropout(DROPOUT),
                nn.LeakyReLU(0.1),
                nn.Conv1d(128, N + Nu, kernel_size = 3, padding = 'same'),
                nn.Dropout(DROPOUT),
                nn.LeakyReLU(0.1)
            )
            
            Phi = AutoEncoder(encoder, decoder)
            
        elif PRIOR == 'CN':
            
            Phi = ConvNet(
                nn.Sequential(
                    nn.Conv1d(N + Nu, 128, kernel_size = 3, padding = 'same'),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(128, N + Nu, kernel_size = 3, padding = 'same') 
                )
            )
        #end
        
        profiler_kwargs = {'max_epochs' : EPOCHS, 'log_every_n_steps' : 1, 'gpus' : gpus}
        
        if LOAD_CKPT:
            
            if WHAT_CKPT == 'best':
                model_file = open(os.path.join(PATH_MODEL_SOURCE, 'checkpoints', 'model.ckpt'), 'rb')            
            elif WHAT_CKPT == 'last':
                model_file = open(os.path.join(PATH_MODEL_SOURCE, 'checkpoints', 'last.ckpt'), 'rb')
            #end
            
            model_state_dict = torch.load(model_file)['state_dict']
            lit_model = LitModel( Phi, shapeData = (BATCH_SIZE, N, FORMAT_SIZE),
                                  preprocess_params = test_set.preprocess_params
            )
            lit_model.load_state_dict(model_state_dict)
            trainer = pl.Trainer(**profiler_kwargs)
            trainer.fit(lit_model, train_loader, val_loader)
            
        else:
            
            if SAVE_CKPT:
                
                lit_model = LitModel( Phi, shapeData = (BATCH_SIZE, N, FORMAT_SIZE),
                                      preprocess_params = test_set.preprocess_params
                )
                
                model_checkpoint = ModelCheckpoint(
                    monitor = 'val_loss',
                    dirpath = os.path.join(PATH_MODEL, 'checkpoints'),
                    filename = 'model',
                    save_top_k = SAVE_TOP_K_CKPT,
                    save_last = SAVE_LAST_CKPT,
                    mode = 'min'
                )
                
                trainer = pl.Trainer(**profiler_kwargs, callbacks = [model_checkpoint])
                trainer.fit(lit_model, train_loader, val_loader)
                
            
            else:
                    
                lit_model = LitModel( Phi, shapeData = (BATCH_SIZE, N, FORMAT_SIZE),
                                      preprocess_params = test_set.preprocess_params
                )
                
                trainer = pl.Trainer(**profiler_kwargs)
                trainer.fit(lit_model, train_loader, val_loader)
            #end
        #end
        
        if SAVE_TRAIN:
            model_file = open(os.path.join(PATH_MODEL, '{}.pkl'.format(MODEL_NAME)), 'wb')
            torch.save({
                'model' : lit_model, 'trainer' : trainer,
                'name' : MODEL_NAME, 'saved_at' : datetime.now()},
                model_file
            )
        #end
    #end
    
    ''' MODEL TEST '''
    if TEST:
        
        if LOAD_TEST:
            model_file = open(os.path.join(PATH_MODEL, '{}.pkl'.format(MODEL_NAME)), 'rb')
            saved_model = torch.load( model_file )
            trainer = saved_model['trainer']
            lit_model = saved_model['model']
            saved_at = saved_model['saved_at']
            name = saved_model['name']
            print('\nModel : {}, saved at {}\n'.format(name, saved_at))
        #end
        
        lit_model.eval()
        lit_model.Phi.eval()
        lit_model.model.eval()
        trainer.test(lit_model, test_loader)
        
        if PLOTS:
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
        windspeed_rmses['only_UPA']['u'][run] = pred_error_metric
        
        predictions.append( u_reco )
    #end
#end


''' Median of the prediction to produce the voted-by-models prediction '''
preds = torch.Tensor( np.median( np.array(predictions), axis = 0 ) )
wdata = torch.Tensor( u_data )
windspeed_baggr = NormLoss((preds - wdata), mask = None, divide = True, rmse = True)
windspeed_rmses['only_UPA']['aggr'] = windspeed_baggr


if SAVE_OUTS:
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
    
    pickle.dump(windspeed_rmses, open(os.path.join(os.getcwd(), 'Evaluation', '{}.pkl'.format(MODEL_NAME)), 'wb'))
    
    with open( os.path.join(os.getcwd(), 'Evaluation', '{}.txt'.format(MODEL_NAME)), 'w' ) as f:
        f.write('Minimum    ; {:.4f}\n'.format(windspeed_rmses['only_UPA']['u'].min()))
        f.write('Mean ± std ; {:.4f} ± {:.4f}\n'.format(windspeed_rmses['only_UPA']['u'].mean(),
                                                      windspeed_rmses['only_UPA']['u'].std()))
        f.write('Median     ; {:.4f}\n'.format(windspeed_baggr))
    f.close()
#end
