#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020
@author: rfablet
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CorrelateNoise(torch.nn.Module):
    def __init__(self, shape_data, dim_cn):
        super(CorrelateNoise, self).__init__()
        self.conv1 = torch.nn.Conv2d(shape_data, dim_cn, (3, 3), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(dim_cn, 2 * dim_cn, (3, 3), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim_cn, shape_data, (3, 3), padding=1, bias=False)
    #end
    
    def forward(self, w):
        w = self.conv1(F.relu(w)).to(device)
        w = self.conv2(F.relu(w)).to(device)
        w = self.conv3(w).to(device)
        return w
    #end
#end

class RegularizeVariance(torch.nn.Module):
    def __init__(self, shape_data, dim_rv):
        super(RegularizeVariance, self).__init__()
        self.conv1 = torch.nn.Conv2d(shape_data, dim_rv, (3, 3), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(dim_rv, 2 * dim_rv, (3, 3), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim_rv, shape_data, (3, 3), padding=1, bias=False)
    #end    
    
    def forward(self, v):
        v = self.conv1(F.relu(v)).to(device)
        v = self.conv2(F.relu(v)).to(device)
        v = self.conv3(v).to(device)
        return v
    #end
#end

def compute_WeightedLoss(x2,w):
    x2_msk = x2[:, w==1, ...]
    x2_num = ~x2_msk.isnan() & ~x2_msk.isinf()
    loss2 = F.mse_loss(x2_msk[x2_num], torch.zeros_like(x2_msk[x2_num]))
    loss2 = loss2 *  w.sum()
    return loss2
#end

# Modules for the definition of the norms for
# the observation and prior model
class Model_WeightedL2Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL2Norm, self).__init__()
    #end
    
    def forward(self, x, w, eps = 0.):
        
        loss_ = torch.nansum( x**2, dim = 3)
        loss_ = torch.nansum( loss_, dim = 2)
        loss_ = torch.nansum( loss_, dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )
        
        return loss_
    #end
#end

class Model_WeightedL1Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL1Norm, self).__init__()
    #end
    
    def forward(self,x,w,eps):
        
        loss_ = torch.nansum( torch.sqrt( eps**2 + x**2 ) , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_
    #end
#end

class Model_WeightedLorenzNorm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedLorenzNorm, self).__init__()
    #end
    
    def forward(self,x,w,eps):
        
        loss_ = torch.nansum( torch.log( 1. + eps**2 * x**2 ) , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )
        
        return loss_
    #end
#end

class Model_WeightedGMcLNorm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL1Norm, self).__init__()
    #end
    
    def forward(self,x,w,eps):
        
        loss_ = torch.nansum( 1.0 - torch.exp( - eps**2 * x**2 ) , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )
        
        return loss_
    #end
#end

def compute_WeightedL2Norm1D(x2,w):
    loss_ = torch.nansum(x2**2 , dim = 2)
    loss_ = torch.nansum( loss_ , dim = 0)
    loss_ = torch.nansum( loss_ * w )
    loss_ = loss_ / (torch.sum(~torch.isnan(x2)) / x2.shape[1] )
    
    return loss_
#end


# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class ConvLSTM2d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3, stochastic=False):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)
        self.stochastic = stochastic
        #self.correlate_noise = CorrelateNoise(input_size, 10)
        #self.regularize_variance = RegularizeVariance(input_size, 10)
    #end
    
    def forward(self, input_, prev_state):
        
        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]
        if self.stochastic == True:
            z = torch.randn(input_.shape).to(device)
            z = self.correlate_noise(z)
            z = (z-torch.mean(z))/torch.std(z)
            #z = torch.mul(self.regularize_variance(z),self.correlate_noise(z))
        #end
        
        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(device),
                torch.autograd.Variable(torch.zeros(state_size)).to(device)
            )
        #end
        
        # prev_state has two components
        prev_hidden, prev_cell = prev_state
        
        # data size is [batch, channel, height, width]
        if self.stochastic == False:
            stacked_inputs = torch.cat((input_, prev_hidden), 1)
        else:
            stacked_inputs = torch.cat((torch.add(input_, z), prev_hidden), 1)
        #end
        
        gates = self.Gates(stacked_inputs)
        
        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        
        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        
        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)
        
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        
        return hidden, cell
    #end
#end

class ConvLSTM1d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3):
        super(ConvLSTM1d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv1d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)
        #end
    #end
    
    def forward(self, input_, prev_state):
        
        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]
        
        # generate empty prev_state, if None is provided
        if prev_state is None:
            '''
            ATTENTION : the variable ``spatial_size`` has the value of
            24, which is not spatial but rather the temporal dimension.
            Could it be an issue?
            Is it the dimension on which the convolutional operators act,
            regardless of the fact they are 1d or 2d?
            '''
            # batch_size   : m
            # hidden_size  : dim state of LSTM (user defined)
            # spatial_size : FORMAT_SIZE
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(device),
                torch.autograd.Variable(torch.zeros(state_size)).to(device)
            )
        #end
        
        # prev_state has two components
        # input_     : [m, N, T]
        # prev_state : ([m, dim_LSTM, T], [m, dim_LSTM, T])
        prev_hidden, prev_cell = prev_state
        
        # data size is [batch, channel, height, width]
        # stacked_inputs : [m, N + dim_LSTM, T]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
        
        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        
        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        
        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)
        
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        
        # hidden : [m, dim_LSTM, T]
        # cell   : [m, dim_LSTM, T]
        return hidden, cell
    #end
#end

class model_GradUpdateLSTM(torch.nn.Module):
    
    def __init__(self, ShapeData, periodicBnd = False, DimLSTM = 0, rateDropout = 0., stochastic = False):
        super(model_GradUpdateLSTM, self).__init__()
        
        with torch.no_grad():
            
            self.shape = ShapeData
            if DimLSTM == 0:
                self.dim_state = 5 * self.shape[0]
            else:
                self.dim_state = DimLSTM
            #end
            
            self.PeriodicBnd = periodicBnd
            if( (self.PeriodicBnd == True) & (len(self.shape) == 2) ):
                print('No periodic boundary available for FxTime (eg, L63) tensors. Forced to False')
                self.PeriodicBnd = False
            #end
        #end
        
        self.convLayer = self._make_ConvGrad()
        K = torch.Tensor([0.1]).view(1,1,1,1)
        self.convLayer.weight = torch.nn.Parameter(K)
        
        self.dropout = torch.nn.Dropout(rateDropout)
        self.stochastic = stochastic
        
        if len(self.shape) == 2: ## 1D Data
            self.lstm = ConvLSTM1d(self.shape[0], self.dim_state, 3)
        elif len(self.shape) == 3: ## 2D Data
            self.lstm = ConvLSTM2d(self.shape[0], self.dim_state, 3, stochastic = self.stochastic)
        #end
    #end
    
    def _make_ConvGrad(self):
        
        layers = []
        if len(self.shape) == 2: ## 1D Data
            layers.append(torch.nn.Conv1d(self.dim_state, self.shape[0], 1, padding = 0, bias = False))
        elif len(self.shape) == 3: ## 2D Data
            layers.append(torch.nn.Conv2d(self.dim_state, self.shape[0], (1,1), padding = 0, bias = False))
        #end
        
        return torch.nn.Sequential(*layers)
    #end
    
    def _make_LSTMGrad(self):
        
        layers = []
        if len(self.shape) == 2: ## 1D Data
            layers.append(ConvLSTM1d(self.shape[0], self.dim_state, 3))
        elif len(self.shape) == 3: ## 2D Data
            layers.append(ConvLSTM2d(self.shape[0], self.dim_state, 3, stochastic = self.stochastic))
        #end
        
        return torch.nn.Sequential(*layers)
    #end
    
    def forward(self, hidden, cell, grad, gradnorm = 1.0):
        
        # compute gradient
        # hidden : [m, 2, T]
        # cell   : [m, 2, T]
        # grad   : [m, N, T]
        grad = grad / gradnorm
        grad = self.dropout( grad )
        
        if self.PeriodicBnd == True :
            dB = 7
            grad_ = torch.cat( (grad[:,:,grad.size(2)-dB:,:], grad, grad[:,:,0:dB,:]) ,dim=2 )
            if hidden is None:
                hidden_,cell_ = self.lstm(grad_, None)
            else:
                hidden_ = torch.cat( (hidden[:,:,grad.size(2)-dB:,:], hidden, hidden[:,:,0:dB,:]), dim=2 )
                cell_ = torch.cat( (cell[:,:,grad.size(2)-dB:,:], cell, cell[:,:,0:dB,:]), dim=2 )
                hidden_,cell_ = self.lstm(grad_, [hidden_, cell_])
            #end
            
            hidden = hidden_[:,:,dB:grad.size(2)+dB,:]
            cell = cell_[:,:,dB:grad.size(2)+dB,:]
        else:
            if hidden is None:
                
                # grad   : [m, N, T]
                # hidden : []
                # cell   : []
                hidden, cell = self.lstm(grad, None)
            else:
                hidden, cell = self.lstm(grad, [hidden, cell])
            #end
        #end
        
        grad = self.dropout( hidden ) # grad : [m, dim_LSTM, T]
        grad = self.convLayer( grad ) # grad : [m, N, T]
        
        return grad, hidden, cell
    #end
#end


# New module for the definition/computation of the variational cost
class Model_Var_Cost(nn.Module):
    
    def __init__(self, m_NormObs, m_NormPhi, ShapeData,
                 dim_obs = 1, dim_obs_channel = 0, dim_state = 0,
                 learnable_params = False,
                 alphaObs = 1., alphaReg = 1.):
        super(Model_Var_Cost, self).__init__()
        
        self.dim_obs_channel = dim_obs_channel
        self.dim_obs = dim_obs
        
        if dim_state > 0 :
            self.dim_state = dim_state
        else:
            self.dim_state = ShapeData[0]
        #end
        
        self.normObs   = m_NormObs
        self.normPrior = m_NormPhi
        
        # parameters for variational cost
        if learnable_params:
            self.alphaObs = torch.nn.Parameter(torch.Tensor(1. * np.ones((self.dim_obs,1)))).to(device)
            self.alphaReg = torch.nn.Parameter(torch.Tensor([1.])).to(device)
            if self.dim_obs_channel[0] == 0 :
                self.WObs = torch.nn.Parameter(torch.Tensor(np.ones((self.dim_obs,ShapeData[0])))).to(device)
                self.dim_obs_channel = ShapeData[0] * np.ones((self.dim_obs,))
            else:
                self.WObs = torch.nn.Parameter(torch.Tensor(np.ones((self.dim_obs,np.max(self.dim_obs_channel))))).to(device)
            #end
            self.WReg = torch.nn.Parameter(torch.Tensor(np.ones(self.dim_state,)))
            self.epsObs = torch.nn.Parameter(0.1 * torch.Tensor(np.ones((self.dim_obs,))))
            self.epsReg = torch.nn.Parameter(torch.Tensor([0.1]))
        else:
            self.alphaObs = torch.Tensor([alphaObs]).to(device)
            self.alphaReg = torch.Tensor([alphaReg]).to(device)
            if self.dim_obs_channel[0] == 0 :
                self.WObs = torch.Tensor(np.ones((self.dim_obs,ShapeData[0])))
                self.dim_obs_channel = ShapeData[0] * np.ones((self.dim_obs,))
            else:
                self.WObs = torch.Tensor(np.ones((self.dim_obs, np.max(self.dim_obs_channel))))
            #end
            self.WReg = torch.Tensor(np.ones(self.dim_state,))
            self.epsObs = 0.1 * torch.Tensor(np.ones((self.dim_obs,)))
            self.epsReg = torch.Tensor([0.1])
        #end
    #end
    
    def forward(self, data_fidelty, regularization, mask_obs):
        
        # data_fidelty   : [m, N, T]
        # regularization : [m, N, T]
        # mask_obs       : [m, N, T]
        loss = self.alphaReg.pow(2) * self.normPrior(regularization, mask = None)
        
        if self.dim_obs == 1:
            loss += self.alphaObs[0].pow(2) * self.normObs(data_fidelty, mask_obs)
        else:
            for kk in range(0,self.dim_obs):
                loss += self.alphaObs[kk].pow(2) * self.normObs(data_fidelty[kk], mask = mask_obs)
            #end
        #end
        
        return loss
    #end
#end


class Model_Distance_Pdist(nn.Module):
    
    def __init__(self, m_MeasureDistMetric, ShapeData, n_bins_histogram,
                 dim_obs = 1, dim_obs_channel = 1, dim_state = 1,):
        super(Model_Distance_Pdist, self).__init__()
        
        self.m_MeasureDistMetric = m_MeasureDistMetric
        self.n_bins_histogram = n_bins_histogram
        
        self.dim_obs_channel = dim_obs_channel
        self.dim_obs = dim_obs
        
        if dim_state > 0 :
            self.dim_state = dim_state
        else:
            self.dim_state = ShapeData[0]
        #end
        
        pass
    #end
    
    def forward(self, guess, target):
        
        var_cost_cat = self.m_MeasureDistMetric(guess, target)
        return var_cost_cat
    #end
#end

# 4DVarNN Solver class using automatic differentiation for the computation of gradient of the variational cost
# input modules: operator phi_r, gradient-based update model m_Grad
# modules for the definition of the norm of the observation and prior terms given as input parameters 
# (default norm (None) refers to the L2 norm)
# updated inner modles to account for the variational model module
class Solver_Grad_4DVarNN(nn.Module):
    
    def __init__(self , prior, mod_H, m_Grad, m_NormObs, m_NormPhi, ShapeData, n_iter_grad,
                 stochastic = False, varcost_learnable_params = False,
                 probabilistic = False, pddmeasure = None, n_bins_histogram = None):
        super(Solver_Grad_4DVarNN, self).__init__()
        
        self.Phi = prior
        self.probabilistic = probabilistic
        self.n_bins_histogram = n_bins_histogram
        
        if probabilistic and pddmeasure is None:
            raise ValueError('probabilistic requires a probability distribution distance measure')
        #end
        
        if m_NormObs == None:
            m_NormObs =  Model_WeightedL2Norm()
        if m_NormPhi == None:    
            m_NormPhi = Model_WeightedL2Norm()
        #end
        
        self.model_H = mod_H
        self.model_Grad = m_Grad
        self.model_VarCost = Model_Var_Cost(m_NormObs, m_NormPhi, ShapeData, 
                                            mod_H.dim_obs, mod_H.dim_obs_channel,
                                            learnable_params = varcost_learnable_params)
        self.model_PDDMeasure = Model_Distance_Pdist(pddmeasure, ShapeData, n_bins_histogram)
        
        self.stochastic = stochastic
        
        with torch.no_grad():
            self.n_grad = int(n_iter_grad)
        #end
    #end
    
    def forward(self, x, yobs, mask):
        
        # x    : [m, N, T]
        # yobs : [m, N, T]
        # mask : [m, N, T]
        return self.solve(x_0 = x, obs = yobs, mask = mask)
    #end
    
    def solve(self, x_0, obs, mask):
        
        # x_0 : [m, N, T]
        # x_k : [m, N, T]
        x_k = torch.mul(x_0, 1.)
        hidden = None
        cell = None 
        normgrad_ = 0.
        
        for _ in range(self.n_grad):
            
            x_k_plus_1, hidden, cell, normgrad_ = self.solver_step(x_k, obs, mask, hidden, cell, normgrad_)
            x_k = torch.mul(x_k_plus_1,1.)
        #end
        
        return x_k_plus_1, hidden, cell, normgrad_
    #end
    
    def solver_step(self, x_k, obs, mask, hidden, cell, normgrad = 0.):
        
        # x_k  : [m, N, T]
        # obs  : [m, N, T]
        # mask : [m, N, T]
        var_cost, var_cost_grad = self.var_cost(x_k, obs, mask)
        
        if normgrad == 0. :
            normgrad_= torch.sqrt( torch.mean( var_cost_grad**2 + 0.))
        else:
            normgrad_= normgrad
        #end
        
        grad, hidden, cell = self.model_Grad(hidden, cell, var_cost_grad, normgrad_)
        grad *= 1./ self.n_grad
        x_k_plus_1 = x_k - grad
        
        return x_k_plus_1, hidden, cell, normgrad_
    #end
    
    def var_cost(self, x, yobs, mask):
        
        # x    : [m, N, T]
        # yobs : [m, N, T]
        # mask : [m, N, T]
        data_fidelty = self.model_H(x, yobs, mask)
        regularization = x - self.Phi(x)
        
        # data_fidelty   : [m, N, T]
        # regularization : [m, N, T]
        var_cost = self.model_VarCost(data_fidelty, regularization, mask)
        
        if self.probabilistic:
            x_hist = x[:, -self.n_bins_histogram:, :]
            phi_x_hist = self.Phi(x)[:, -self.n_bins_histogram:, :]
            
            var_cost_cat = self.model_PDDMeasure(x_hist, phi_x_hist)
            var_cost += var_cost_cat
        #end
        
        var_cost_grad = torch.autograd.grad(var_cost, x, create_graph = True)[0]
        
        return var_cost, var_cost_grad
    #end
#end


