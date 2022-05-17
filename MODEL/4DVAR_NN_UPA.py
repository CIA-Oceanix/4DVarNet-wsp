
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
from numpy.random import default_rng
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

from tutls import L2NormLoss, NormLoss, CrossEntropyLoss, xavier_weights_initialization
from dutls import SMData
from gutls import plot_UPA, plot_WS, plot_WS_scatter
import solver as NN_4DVar

if torch.cuda.is_available():
    device = 'cuda'
    gpus = -1
    NUM_WORKERS = 64
else:
    device = 'cpu'
    gpus = 0
    NUM_WORKERS = 8
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

class AutoEncoder_regression(nn.Module):
    
    def __init__(self, N_data, latent_dim):
        super(AutoEncoder_regression, self).__init__()
        
        self.econv1 = nn.Conv1d(N_data, 128, kernel_size = 3, padding = 'same')
        self.enl1   = nn.LeakyReLU(0.1)
        self.econv2 = nn.Conv1d(128, latent_dim, kernel_size = 3, padding = 'same')
        
        self.dconv1 = nn.Conv1d(latent_dim, 128, kernel_size = 3, padding = 'same')
        self.dnl1   = nn.LeakyReLU(0.1)
        self.dconv2 = nn.Conv1d(128, N_data, kernel_size = 3, padding = 'same')
        self.dnl2   = nn.LeakyReLU(0.1)
    #end
    
    def forward(self, data):
        
        h = self.enl1( self.econv1(data) )
        h = self.econv2(h)
        h = self.dnl1( self.dconv1(h) )
        out = self.dnl2( self.dconv2(h) )
        
        return out
    #end
#end


class AutoEncoder_classification(nn.Module):
    
    def __init__(self, N_data, latent_dim, N_situ):
        super(AutoEncoder_classification, self).__init__()
        
        self.econv1 = nn.Conv1d(N_data, 128, kernel_size = 3, padding = 'same')
        self.enl1   = nn.LeakyReLU(0.1)
        self.econv2 = nn.Conv1d(128, latent_dim, kernel_size = 3, padding = 'same')
        
        self.dconv1      = nn.Conv1d(latent_dim, 128, kernel_size = 3, padding = 'same')
        self.dnl1        = nn.LeakyReLU(0.1)
        self.dconv2_data = nn.Conv1d(128, N_data - N_situ, kernel_size = 3, padding = 'same')
        self.dnl2_data   = nn.LeakyReLU(0.1)
        self.dconv2_ws   = nn.Conv1d(128, N_situ, kernel_size = 3, padding = 'same')
        self.dnl2_ws     = nn.LogSoftmax(dim = 2)
    #end
    
    def forward(self, data):
        
        h = self.enl1( self.econv1(data) )
        h = self.econv2(h)
        h = self.dnl1( self.dconv1(h) )
        reco_data = self.dnl2_data( self.dconv2_data(h) )
        reco_ws = self.dconv2_ws(h)
        # reco_ws = self.dnl2_ws(reco_ws)
        
        out = torch.cat((reco_data, reco_ws), dim = 1)
        return out
    #end
#end



class UNet(nn.Module):
    
    def __init__(self):
        pass
    #end
    
    def forward(self):
        pass
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
        self.hparams.n_fourdvar_iter = N_4DV_ITER
        self.hparams.automatic_optimization = True
        self.preprocess_params = preprocess_params
        self.test_rmse = np.float64(0.)
        self.samples_to_save = list()
        self.train_losses = np.zeros(EPOCHS)
        self.val_losses = np.zeros(EPOCHS)
        
        if MM_ECMWF:
            mod_shape_data = [N_UPA + N_ECMWF + N_SITU, FORMAT_SIZE]
        else:
            mod_shape_data = [N_UPA + N_SITU, FORMAT_SIZE]
        #end
        
        self.model = NN_4DVar.Solver_Grad_4DVarNN(  # Instantiation of the LSTM solver
            
            Phi,                              # Dynamical prior   
            Model_H(shapeData),               # Observation operator
            NN_4DVar.model_GradUpdateLSTM(    # m_Grad
                mod_shape_data,                  # m_Grad : Shape Data
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
        data_UPA   = batch[0].reshape(batch_size, FORMAT_SIZE, N_UPA).clone()
        data_we    = batch[1].reshape(batch_size, FORMAT_SIZE, N_ECMWF).clone()
        data_ws    = batch[2].reshape(batch_size, FORMAT_SIZE, N_SITU).clone()
        data_UPA   = data_UPA.transpose(1, 2)
        data_we    = data_we.transpose(1, 2)
        data_ws    = data_ws.transpose(1, 2)
        
        data_UPA[data_UPA.isnan()] = 0.
        data_ws[data_ws.isnan()] = 0.
        data_we[data_we.isnan()] = 0.
        
        if not MM_ECMWF:
            state = torch.cat( (data_UPA, 0. * data_ws), dim = 1 )
        else:
            state = torch.cat( (data_UPA, data_we, 0. * data_ws), dim = 1 )
        #end
        
        return state
    #end
    
    def compute_loss(self, batch, phase = 'train', state_init = None):
        
        '''Reshape the input data'''
        '''
        NOTE : If the task is classification, the dutls.SMData class 
        takes charge of doing the Beaufort class quantization and the
        one-hot encoding of real wind speed values
        '''
        batch_size = batch[0].shape[0]
        data_UPA   = batch[0].reshape(batch_size, FORMAT_SIZE, N_UPA).clone()
        data_we    = batch[1].reshape(batch_size, FORMAT_SIZE, N_ECMWF).clone()
        data_ws    = batch[2].reshape(batch_size, FORMAT_SIZE, N_SITU).clone()
        data_UPA   = data_UPA.transpose(1, 2)
        data_we    = data_we.transpose(1, 2)
        data_ws    = data_ws.transpose(1, 2)
        
        
        ''' Possible change: set data_we to NaN, so to make masks zeros! '''
        if phase == 'test' and TEST_ECMWF is not None:
            ''' If MM_ECMWF is true but TEST_ECMWF is false,
                then the mulit-modal model is used, with no
                modification to ECMWF wind speed '''
            
            if TEST_ECMWF == 'zero':
                print("zero")
                data_we = torch.zeros_like(data_we)
                
            elif TEST_ECMWF == 'dmean':
                print("dmean")
                mean_we = torch.zeros_like(data_we)
                for i in range(data_we.shape[0]):
                    mean_we[i,:,:] = data_we[i].mean()
                #end
                data_we = mean_we
            #end
        #end
        
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
        
        ''' Missing data : produce random masks '''
        if IS_TRAIN is not None:
            
            for i in range(batch_size):
                
                num_indices = default_rng().choice(
                    FORMAT_SIZE,
                    size = np.int32(IS_TRAIN * FORMAT_SIZE),
                    replace = False
                )
                rand_idx = list( num_indices )
                mask_UPA[i, :, rand_idx] = 0.
            #end
        #end
        
        ''' k-steps-ahead prediction : produce proper masks '''
        if KSA_TRAIN is not None:
            
            mask_UPA[:, :, np.int32(KSA_TRAIN):] = 0.
            mask_we[:, :, np.int32(KSA_TRAIN):] = 0.
        #end
        
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
        
        if not MM_ECMWF:
            inputs_init = inputs_init * torch.cat( (mask_UPA, 
                                                    torch.ones_like(data_ws)), 
                                                  dim = 1 )
        else:
            inputs_init = inputs_init * torch.cat( (mask_UPA, 
                                                    torch.ones_like(data_we),
                                                    torch.ones_like(data_ws)), 
                                                  dim = 1 )
        #end
        
        with torch.set_grad_enabled(True):
            
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad = True)
            
            if FIXED_POINT:
                outputs = self.Phi(data_input)
            else:
                outputs, hidden, cell, normgrad = self.model(inputs_init, data_input, mask_input)
            #end
            
            '''Split UPA and windspeed reconstructions and predictions and 
               reinstate them in the ``(batch_size, time_series_length, num_features)`` format'''
            reco_UPA = outputs[:, :N_UPA, :]
            reco_we  = outputs[:, N_UPA : N_UPA + N_ECMWF, :]
            reco_ws  = outputs[:, -N_SITU:, :]
            
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
                
                ''' If test, then denormalize the data and append them in a list
                   so to plot them in the end '''
                data_UPA = self.undo_preprocess(data_UPA, self.preprocess_params['upa'])
                reco_UPA = self.undo_preprocess(reco_UPA, self.preprocess_params['upa'])
                data_we  = self.undo_preprocess(data_we,  self.preprocess_params['wind_ecmwf'])
                reco_we  = self.undo_preprocess(reco_we,  self.preprocess_params['wind_ecmwf'])
                
                if TASK == 'reco':
                    ''' If the task is classification, then it makes no sense
                        to denormalize wind classes probabilities '''
                    data_ws  = self.undo_preprocess(data_ws,  self.preprocess_params['wind_situ'])
                    reco_ws  = self.undo_preprocess(reco_ws,  self.preprocess_params['wind_situ'])
                #end
                
                ''' Recreate the outputs variable '''
                outputs = { 'y_data' : data_UPA, 'y_reco' : reco_UPA,
                            'u_data' : data_ws,  'u_reco' : reco_ws,
                            'w_data' : data_we,  'w_reco' : reco_we
                }
                
                if phase == 'test':
                    self.samples_to_save.append( outputs )
                #end
            #end
            
            if TASK == 'reco':
                ''' Loss computation. Note the use of the masks and see the ``NormLoss`` documentation in
                   the devoted module '''
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
            
            elif TASK == 'class':
                # loss = MSE(UPA and ECMWF) + BCE(wind speed classes)
                
                loss_data = NormLoss((data_UPA - reco_UPA), mask = mask_UPA, divide = True, dformat = 'mtn')
                loss_ws = CrossEntropyLoss(data_ws, reco_ws, mask = mask_ws, divide = True, dformat = 'mtn') 
                
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

# ! IMPLICIT NONE

# CONSTANTS
with open(os.path.join(os.getcwd(), 'cparams.json'), 'r') as filestream:
    CPARAMS = json.load(filestream)
filestream.close()

WIND_VALUES = CPARAMS['WIND_VALUES']
DATA_TITLE  = CPARAMS['DATA_TITLE']

RUNS        = CPARAMS['RUNS']
EPOCHS      = CPARAMS['EPOCHS']
TRAIN       = CPARAMS['TRAIN']
TEST        = CPARAMS['TEST']
PLOTS       = CPARAMS['PLOTS']
MM_ECMWF    = CPARAMS['MM_ECMWF']
TEST_ECMWF  = CPARAMS['TEST_ECMWF']
IS_TRAIN    = CPARAMS['IS_TRAIN']
KSA_TRAIN   = CPARAMS['KSA_TRAIN']
TASK        = CPARAMS['TASK']

FIXED_POINT = CPARAMS['FIXED_POINT']
LOAD_CKPT   = CPARAMS['LOAD_CKPT']
PRIOR       = CPARAMS['PRIOR']

N_SOL_ITER  = CPARAMS['N_SOL_ITER']
NSOL_IT_REF = CPARAMS['NSOL_IT_REF']
N_4DV_ITER  = CPARAMS['N_4DV_ITER']

BATCH_SIZE  = CPARAMS['BATCH_SIZE']
LATENT_DIM  = CPARAMS['LATENT_DIM']
DIM_LSTM    = CPARAMS['DIM_LSTM']

DROPOUT     = CPARAMS['DROPOUT']
WEIGHT_DATA = CPARAMS['WEIGHT_DATA']
WEIGHT_PRED = CPARAMS['WEIGHT_PRED']
WEIGHT_WE   = CPARAMS['WEIGHT_WE']
SOLVER_LR   = CPARAMS['SOLVER_LR']
SOLVER_WD   = CPARAMS['SOLVER_WD']
PHI_LR      = CPARAMS['PHI_LR']
PHI_WD      = CPARAMS['PHI_WD']

FORMAT_SIZE = 24
MODEL_NAME  = '4DVAR'
PATH_DATA   = os.getenv('PATH_DATA')
PATH_MODEL  = os.getenv('PATH_MODEL')

CPARAMS.update({'FORMAT_SIZE' : FORMAT_SIZE})

MODEL_NAME  = f'{MODEL_NAME}_{PRIOR}'

if MM_ECMWF:
    MODEL_NAME = f'{MODEL_NAME}_MM'
else:
    MODEL_NAME = f'{MODEL_NAME}_SM'
#end

if TASK == 'reco':
    MODEL_NAME = f'{MODEL_NAME}_reco'
elif TASK == 'class':
    MODEL_NAME = f'{MODEL_NAME}_class'
#end

if FIXED_POINT:
    MODEL_NAME = f'{MODEL_NAME}_fp1it'
else:
    MODEL_NAME = f'{MODEL_NAME}_gs{NSOL_IT_REF}it'
#end

if LOAD_CKPT:
    MODEL_SOURCE = MODEL_NAME
    MODEL_NAME = f'{MODEL_NAME}_lckpt_gs{N_SOL_ITER}'
else:
    MODEL_SOURCE = MODEL_NAME
#end

if TEST_ECMWF is not None:
    
    if TEST_ECMWF != 'zero' and TEST_ECMWF != 'dmean':
        raise ValueError('ECMWF modification does not match available possibilities')
    #end
#end

if TEST_ECMWF is not None:
    MODEL_NAME = f'{MODEL_NAME}_{TEST_ECMWF}'
#end

if IS_TRAIN is not None:
    is_percentage = str(IS_TRAIN).replace('.', 'd')
    MODEL_NAME = f'{MODEL_NAME}_is{is_percentage}'
#end

if KSA_TRAIN is not None:
    ksa_steps = str(KSA_TRAIN) + 'hours'
    MODEL_NAME = f'{MODEL_NAME}_ksa{ksa_steps}'
#end

PATH_SOURCE = os.path.join(PATH_MODEL, MODEL_SOURCE)
PATH_MODEL = os.path.join(PATH_MODEL, MODEL_NAME)
if not os.path.exists(PATH_SOURCE) and LOAD_CKPT: os.mkdir(PATH_SOURCE)
if not os.path.exists(PATH_MODEL): os.mkdir(PATH_MODEL)


# Introduction
print('Experiment:')
print('----------------------------------------------------------------------')
print(f'Prior                             : {PRIOR}')
print(f'Fixed point                       : {FIXED_POINT}')
print(f'Runs                              : {RUNS}')
print(f'Inlcude ECMWF                     : {MM_ECMWF}')
print(f'Test mode ECMWF                   : {TEST_ECMWF}')
print(f'Path Source                       : {PATH_SOURCE}')
print(f'Path Target                       : {PATH_MODEL}')
print(f'Model                             : {MODEL_NAME}')
print(f'Load from checkpoint              : {LOAD_CKPT} (if False, SOURCE == TARGET)')
print(f'Task                              : {TASK}')
if not FIXED_POINT:
    print(f'N iterations 4DVarNet             : {N_SOL_ITER}')
    print(f'N iterations 4DVarNet (reference) : {NSOL_IT_REF}')
print('----------------------------------------------------------------------')

'''
Initialize the performance metrics data structures
'''
windspeed_rmses = {
    'only_SAR'  : {'u' : np.zeros(RUNS), 'u_x' : np.zeros(RUNS), 'u_y' : np.zeros(RUNS)},
    'only_UPA'  : {'u' : np.zeros(RUNS), 'u_c' : np.zeros(RUNS)},
    'colocated' : {'u' : np.zeros(RUNS), 'u_x' : np.zeros(RUNS), 'u_y' : np.zeros(RUNS)},
}
windspeed_accrs = {
    'classwise' : 0.,
    'total'     : 0.
}
predictions = list()
r2_scores = list()
performance_metrics = {
        'train_loss' : np.zeros((EPOCHS, RUNS)),
        'val_loss'   : np.zeros((EPOCHS, RUNS))
}
classwise_accuracies = list()
total_accuracies = list()

u_datas = list()


for run in range(RUNS):
    print('\n\n----------------------------------------------------------------------')
    print(f'Run {run}\n')
    
    train_set = SMData(os.path.join(PATH_DATA, 'train'), WIND_VALUES, '2011', TASK)
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)#, num_workers = NUM_WORKERS)
    
    val_set = SMData(os.path.join(PATH_DATA, 'val'), WIND_VALUES, '2011', TASK)
    val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False)#, num_workers = NUM_WORKERS)
    
    test_set = SMData(os.path.join(PATH_DATA, 'test'), WIND_VALUES, '2011', TASK)
    test_loader = DataLoader(test_set, batch_size = test_set.__len__(), shuffle = False)#, num_workers = NUM_WORKERS)
    
    N_UPA = train_set.get_modality_data_size('upa')
    N_ECMWF = train_set.get_modality_data_size('wind_ecmwf')
    N_SITU = train_set.get_modality_data_size('wind_situ')
    
    ''' MODEL TRAIN '''
    if TRAIN:
        
        if MM_ECMWF:
            
            N_DATA = N_UPA + N_ECMWF + N_SITU
        else:
            
            N_DATA = N_UPA + N_SITU
        #end
        
        if PRIOR == 'AE':
            
            if 1 == 0:
                encoder = torch.nn.Sequential(
                    nn.Conv1d(N_DATA, 128, kernel_size = 3, padding = 'same'),
                    nn.Dropout(DROPOUT),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(128, LATENT_DIM, kernel_size = 3, padding = 'same'),
                    nn.Dropout(DROPOUT)
                )
                decoder = torch.nn.Sequential(
                    nn.Conv1d(LATENT_DIM, 128, kernel_size = 3, padding = 'same'),
                    nn.Dropout(DROPOUT),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(128, N_DATA, kernel_size = 3, padding = 'same'),
                    nn.Dropout(DROPOUT),
                    nn.LeakyReLU(0.1)
                )
                
                Phi = AutoEncoder(encoder, decoder)
            #end
            
            if TASK == 'reco':
                Phi = AutoEncoder_regression(N_DATA, LATENT_DIM)                
            elif TASK == 'class':
                Phi = AutoEncoder_classification(N_DATA, LATENT_DIM, N_SITU)
            #end
                        
        elif PRIOR == 'CN':
            
            Phi = ConvNet(
                nn.Sequential(
                    nn.Conv1d(N_DATA, 128, kernel_size = 3, padding = 'same'),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(128, N_DATA, kernel_size = 3, padding = 'same')
                )
            )
        #end
        
        if MM_ECMWF:
            
            shape_data = (BATCH_SIZE, N_UPA + N_ECMWF, FORMAT_SIZE)
        else:
            
            shape_data = (BATCH_SIZE, N_UPA, FORMAT_SIZE)
        #end
        
        lit_model = LitModel( Phi, shapeData = shape_data,
                              preprocess_params = test_set.preprocess_params
        )
        
        if LOAD_CKPT:
            
            CKPT_NAME = glob.glob(os.path.join(PATH_SOURCE, f'{run}-' + MODEL_SOURCE + '-epoch=*.ckpt'))[0]
            checkpoint_model = open(CKPT_NAME, 'rb')
            print('\n\nCHECKPOINT (LOAD) : ' + CKPT_NAME + '\n\n')
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
        
    #end
    
    ''' MODEL TEST '''
    if TEST:
        
        CKPT_NAME = glob.glob(os.path.join(PATH_MODEL, f'{run}-' + MODEL_NAME + '-epoch=*.ckpt'))[0]
        print('\n\nCHECKPOINT (TEST) : ' + CKPT_NAME + '\n\n')
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
        
        u_datas.append(u_data)
                
        if TASK == 'class':
            
            # u_pred = torch.nn.functional.softmax(u_pred, dim = 2).bernoulli()
            # calcola accuracy e accuracy per classe
            # report : media accuracy, non RMSE e R²
            
            u_pred = torch.nn.functional.softmax(torch.Tensor(u_reco), dim = 2).bernoulli()
            
            # classwise accuracy
            classwise_accuracy = np.zeros(N_SITU)
            for n in range(N_SITU):
                ud = torch.Tensor(u_data)
                up = torch.Tensor(u_pred)
                classwise_accuracy[n] = (ud[:,:,n] == up[:,:,n]).sum() / (u_data.shape[0] * u_data.shape[1])
            #end
            classwise_accuracies.append(classwise_accuracy)
            
            # total accuracy
            u_data_cat = torch.argmax(torch.Tensor(u_data), dim = 2).flatten()
            u_pred_cat = torch.argmax(torch.Tensor(u_pred), dim = 2).flatten()
            total_accuracy = (u_data_cat == u_pred_cat).sum() / u_data_cat.shape[0]
            total_accuracies.append(total_accuracy)
            
            print(f'Total accuracy     = {total_accuracy:.4f}')
            print('Classwise accuracy = ', classwise_accuracy)
            
            # gather predictions
            predictions.append(u_pred)
            
        elif TASK == 'reco':
            
            pred_error_metric = NormLoss((u_data - u_reco), mask = None, divide = True, rmse = True)
            mask_central = torch.zeros((u_data.shape))
            mask_central[:, FORMAT_SIZE // 2, :] = 1
            pred_error_metric_central = NormLoss((u_data - u_reco), mask = mask_central, divide = True, rmse = True)
            r2_metric = r2_score(u_data.reshape(-1,1), u_reco.reshape(-1,1))
            
            print('R² score = {:.4f}'.format(r2_metric))
            print('RMSE     = {:.4f}'.format(pred_error_metric))
            windspeed_rmses['only_UPA']['u'][run] = pred_error_metric
            windspeed_rmses['only_UPA']['u_c'][run] = pred_error_metric_central
            r2_scores.append(r2_metric)
            predictions.append( u_reco )
        #end
        
    #end
#end

if TASK == 'reco':
    ''' Median of the prediction to produce the voted-by-models prediction '''
    preds = torch.Tensor( np.median( np.array(predictions), axis = 0 ) )
    wdata = torch.Tensor( u_data )
    windspeed_baggr = NormLoss((preds - wdata), mask = None, divide = True, rmse = True)
    windspeed_rmses['only_UPA']['aggr'] = windspeed_baggr

elif TASK == 'class':
    ''' Unlike regression, where we compute the median, for classification
        we need to define how the RUNS models vote the suitable class '''
    windspeed_accrs['classwise'] = np.array(classwise_accuracies).mean(axis = 0)
    windspeed_accrs['total'] = np.array(total_accuracies).mean()
    
    # models vote for most likely classes
    wdata = np.argmax(np.array( u_data ), axis = 2).flatten()
    for k in range(predictions.__len__()):
        predictions[k] = np.argmax(np.array( predictions[k] ), axis = 2).flatten()
    #end
    
    predictions = np.array(predictions).T
    pred_counts = np.zeros((predictions.shape[0], N_SITU))
    
    for i in range(u_data.shape[0]):
        for n in range(N_SITU):
            pred_counts[i,n] = np.count_nonzero(predictions[i] == n)
        #end
    #end
    
    preds = np.argmax(predictions, axis = 1)
    windspeed_baggr = (wdata == preds).sum() / wdata.shape[0]
    windspeed_accrs['aggr'] = windspeed_baggr
    
#end


''' SERIALIZE THE HYPERPARAMETERS IN A JSON FILE '''
with open(os.path.join(PATH_MODEL, 'HYPERPARAMS.json'), 'w') as filename:
    json.dump(CPARAMS, filename, indent = 4)
#end
filename.close()

''' SERIALIZE WIND SPEED VALUES '''
with open(os.path.join(PATH_MODEL, 'wind_data_medianreco.pkl'), 'wb') as filename:
    pickle.dump({'u_data'          : wdata, 
                 'u_pred'          : preds, 
                 'u_pred_ensemble' : predictions}, 
                filename)
filename.close()

''' SERIALIZE TRAINING METRICS '''
with open(os.path.join(os.getcwd(), 'Evaluation', f'{MODEL_NAME}_perfmetrics.pkl'), 'wb') as filename:
    pickle.dump(performance_metrics, filename)
filename.close()

if TASK == 'reco':
    
    ''' SERIALIZE PERFORMANCE METRICS '''
    with open(os.path.join(os.getcwd(), 'Evaluation', f'{MODEL_NAME}_wsm.pkl'), 'wb') as filename:
        pickle.dump(windspeed_rmses, filename)
    filename.close()
    
    ''' WRITE TO TEXT SYNTHESIS REPORT OF PERFORMANCE '''
    with open( os.path.join(os.getcwd(), 'Evaluation', f'{MODEL_NAME}.txt'), 'w' ) as filename:
        filename.write('Minimum          ; {:.4f}\n'.format(windspeed_rmses['only_UPA']['u'].min()))
        filename.write('(all) Mean ± std ; {:.4f} ± {:.4f}\n'.format(windspeed_rmses['only_UPA']['u'].mean(),
                                                      windspeed_rmses['only_UPA']['u'].std()))
        filename.write('(cen) Mean ± std ; {:.4f} ± {:.4f}\n'.format(windspeed_rmses['only_UPA']['u_c'].mean(),
                                                      windspeed_rmses['only_UPA']['u_c'].std()))
        filename.write('Median           ; {:.4f}\n'.format(windspeed_baggr))
        filename.write('R² score         ; {:.2f}\n'.format(np.array(r2_scores).mean()))
    filename.close()

elif TASK == 'class':
    
    ''' SERIALIZE PERFORMANCE METRICS '''
    with open(os.path.join(os.getcwd(), 'Evaluation', f'{MODEL_NAME}_wsm.pkl'), 'wb') as filename:
        pickle.dump(windspeed_accrs, filename)
    filename.close()
    
    ''' WRITE TO TEXT SYNTHESIS REPORT OF PERFORMANCE '''
    with open( os.path.join(os.getcwd(), 'Evaluation', f'{MODEL_NAME}.txt'), 'w' ) as filename:
        filename.write('Mean accuracy      ; {:.4f}\n'.format(windspeed_accrs['total']))
        filename.write('Median             ; {:.4f}\n'.format(windspeed_baggr))
    filename.close()
#end
