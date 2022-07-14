
import torch
from torch import nn


#------------------------------------------------------------------------------
# M O D E L S
#------------------------------------------------------------------------------

# Parameters init

def xavier_weights_initialization(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    #end
#end



# -----------------------------------------------------------------------------
# L O S S E S
# -----------------------------------------------------------------------------

# NORM LOSS AND MISSING VALUES

def NormLoss(item, mask, rmse = False, divide = True, dformat = 'mtn', return_nitems = False):
    r'''Computes the norm loss of the input ``item``, a generic tensor :math:`v`,
    which can be given by the difference between ground-truth data and reconstructions
    or predictions, i.e. :math:`\mathrm{item} = (v - v')`. The presence of missing
    data is accounted for in the following way. Assume that the data is :math:`v \in \mathbb{R}^{D^v}`,
    with :math:`D^v = \{D_1^v, \dots, D_d^v\}` a multi-index (batch size, timesteps, number of features).
    Then we mask this data batch with a mask :math:`\Omega^v \in \{0, 1\}^{D^v}`, such that
    
    .. math::
        \Omega_{itj}^v = 
        \begin{cases}
            1 \quad \text{if} \;\; v_{itj} \in \mathbb{R} \\
            0 \quad \text{if} \;\; v_{itj} \; \text{is non-numeric}
        \end{cases}
    
    The mask is used for masking missing values and to infer the number of
    legitimate present data. If the loss has a value :math:`L`, but only one item
    is responsible, i.e. there is only one legitimate value, then the loss
    "pro-capita" is :math:`L` whatsoever. But if we have a loss :math:`L` being
    caused by :math:`m` legitimate items, then the loss pro-capita is :math:`L / m`,
    which is the unbiased batch-wise loss we are interested in.
    
    The actual loss computation is then 
    
    .. math::
        L(v,v') = \sum_{t = 0}^{T} \frac{1}{\sum_{i = 1}^{m}\sum_{j = 0}^{N^v} \Omega_{itj}^v / Nv} 
          \sum_{i = 0}^{m}\sum_{j = 0}^{N^v} \Omega_{itj}^v \, ( v_{itj} - v_{itj}' )^2
    
    The normalization term accounts matches with the number legitimate data (sum of the
    elements of the mask divided by the number of features). It may give an estimate
    of the actual number of data, since the assumption is that data are row-wise 
    present or missing.
    
    In test stage it could be useful to consider the root mean squared error version of
    this loss, and it sufficies to pass the ``rmse = True`` keyword.
    
    **NOTE**: This function has been developed and tested in such a way for 
    comply for the equality between the two::
        >>> NormLoss(v, p = 2, rmse = True, divide = False)    
        >>> torch.linalg.norm(v, ord = 2, dim = -1).sum()
    
    since we are interested in a row-wise norm, to be summed over the batch items
    (indeed the rows of the tensor :math:`v`).
    
    **NOTE**: The equality between ``NormLoss`` and ``torch.nn.functional.mse_loss`` or
    ``sklearn.metrics.mean_squared_error`` should be expected only if the tensor ``item``
    is a 2-dimensional element. It is due to the fact that this method is designed to
    compute the mean squared error of time series, but the temporal dimension is not 
    accounted for in the average. The mean error is cumulated for each time step. It is
    thought that dividing also for the time series length ends up underestimating the 
    performance metric. In the case of wind speed, however, there is no difference.
    
    Parameters
    ----------
    item : ``torch.Tensor``
        The tensor which one wants the norm of.
    mask : ``torch.Tensor``
        The mask :math:`\Omega^v`. The default is None, in this case the mask is 
        assumed to be a tensor of ones, like if there were no missing values, and
        each item contributes to the loss.
    rmse : ``bool``
        Whether to compute the square root of the norm or not. The default is False.
    divide : ``bool``
        Whether to divide the loss for the number of the present items. The default is True.
    dformat : ``str``. Defaults to ``'mtn'``
        The format of the data passed. ``m`` in the batch dimension, ``t`` is the time
        dimension and ``n`` is the features dimension. The preferable choice is 
        ``'mtn'``, meaning that the data passed are in the following format : 
        ``(batch_size, time_series_length, num_features)``. In this way the loss
        reductions are done in such a way to comply with the equation above.
    return_nitems : ``bool``
        Whether to return the number of effective items in the given batch. Default
        is ``False``.
        
    Returns
    -------
    loss : ``torch.Tensor``
        Weighted loss computed, if ``return_nitems`` is ``False``.
    loss, nitems : ``tuple`` of ``(torch.Tensor, int)``
        The weighted loss and the number of effective items in the batch.

    '''
    if item.__class__ is not torch.Tensor:
        item = torch.Tensor(item)
    #end
    
    if item.shape.__len__() == 1:
        item = item.reshape(-1,1)
    #end
    
    if mask is None:
        mask = torch.ones_like(item)
    #end
    
    argument = (item.mul(mask)).pow(2)
    
    # FIRST SUMMATION : ON FEATURES !!!
    if dformat == 'mtn':
        loss = argument.sum(dim = -1)
    elif dformat == 'mnt':
        loss = argument.sum(dim = 1)
    #end
    
    no_items = False
    nitems = 1
    
    if mask.sum() == 0.:
        no_items = True
        nitems = 1.
    else:
        no_items = False
        num_features = mask.shape[-1]
        nitems = mask.sum() / num_features
    #end
    
    loss = loss.sum(dim = 0).sum() # SUMMATION OVER BATCH FIRST AND OVER TIME THEN
    
    if divide:
        loss = loss.div(nitems)
    #end
    
    if rmse:
        loss = loss.sqrt()
    #end
    
    if no_items == True:
        nitems = 0
    #end
    
    if return_nitems:
        return loss, nitems
    else:
        return loss
    #end
#end

class L2NormLoss(nn.Module):
    r'''
    Wraps the ``NormLoss`` function, makes it an instantiable class, like
    the default ``torch.nn`` modules for loss computation, e.g. ``torch.nn.MSELoss`` etc.
    '''
    
    def __init__(self, dformat = 'mnt', rmse = False, divide = True, return_nitems = False):
        r'''
        SEE ABOVE, FEW CHANGES
        '''
        super(L2NormLoss, self).__init__()
        
        self.dformat = dformat
        self.rmse = rmse
        self.divide = divide
        self.return_nitems = return_nitems
    #end
    
    def forward(self, item, mask):
        
        if mask is None: 
            mask = torch.ones_like(item)
        #end
        
        return NormLoss(item, mask, rmse = self.rmse, divide = self.divide,
                        dformat = self.dformat, return_nitems = self.return_nitems)
    #end
#end


def CE_Loss(label, guess, mask = None, dformat = 'mnt', divide = True, return_nitems = False):
    '''
    Cross entropy loss between categorical ground-truth and output integers.
    
    UNDER MAINTAINANCE!!!
    
    Parameters
    ----------
    label : ``torch.Tensor`` 
        Ground-truths, int in [0, C)
    guess : ``torch.Tensor``
        Output raw probabilities (no softmax applied)
    mask : ``torch.Tensor``
        Binary mask to manage missing values. Defaults to ``None``, in this case
        the mask is a full rank tensor of ones.
    dformat : ``str``
        See ``NormLoss`` documentation
    divide : ``bool``
        See ``NormLoss`` documentation
    
    Returns
    -------
    loss : ``torch.Tensor``
        The computed scalar loss. It has the trace of gradients computed
    
    '''
    
    if guess.__class__ is not torch.Tensor:
        guess = torch.Tensor(guess)
    if label.__class__ is not torch.Tensor:
        label = torch.Tensor(label)
    #end
    
    if mask is None:
        mask = torch.ones_like(label)
    #end
    
    # Convert one-hot encoding to categorical labels for targets
    if dformat == 'mtn':
        label = label.transpose(1,2)
        guess = guess.transpose(1,2)
        
        if label.shape[1] > 1:
            label = torch.argmax(label, dim = 1).reshape(label.shape[0], label.shape[2])
        #end
        
    elif dformat == 'mnt':
        if label.shape[1] > 1:
            label = torch.argmax(label, dim = 1).reshape(label.shape[0], label.shape[2])
        #end
    #end
    
    # For 3D data, cross entropy has shape (batch_size, timesteps)
    # No features
    # So the only sum saturates the dimension dim = 1
    argument = nn.functional.cross_entropy(guess, label, reduction = 'none')
    
    no_items = False
    nitems = 1
    
    if mask.sum() == 0.:
        no_items = True
        nitems = 1.
    else:
        no_items = False
        num_features = mask.shape[-1]
        nitems = mask.sum() / num_features
    #end
    
    loss = argument.sum(dim = 0).sum() # SUMMATION OVER BATCH FIRST AND OVER TIME THEN
    
    if divide: # Batch-wise effective (according to mask) average
        loss = loss.div(nitems)
    #end
    
    if no_items == True:
        nitems = 0
    #end
    
    if return_nitems:
        return loss, nitems
    else:
        return loss
    #end
    
    return loss
#end


class CrossEntropyLoss(nn.Module):
    
    def __init__(self, divide = True, dformat = 'mnt', return_nitems = False):
        super(CrossEntropyLoss, self).__init__()
        
        self.divide = divide
        self.return_nitems = return_nitems
        self.dformat = dformat
    #end
    
    def forward(self, guess, target, mask = None):
        
        return CE_Loss(guess, target, mask = mask,
                       divide = self.divide, dformat = self.dformat,
                       return_nitems = False)
    #end
#end


def MulticlassAccuracy(guess, label, dformat = 'mtn'):
    
    if guess.__class__ is not torch.Tensor:
        guess = torch.Tensor(guess)
    if label.__class__ is not torch.Tensor:
        label = torch.Tensor(label)
    #end
    
    batch_accuracy = (guess == label).float().mean(dim = 1).mean(dim = 0)
    return batch_accuracy
    
#end