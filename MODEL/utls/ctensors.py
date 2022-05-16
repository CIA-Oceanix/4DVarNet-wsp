
import torch

class CTensor(torch.Tensor):
    
    def __init__(self, data):
        super(CTensor, self).__init__()
        
        self.data = torch.Tensor(data)
    #end
        
    def remove_nans(self):
        
        self.data[self.data.isnan()] = 0.
    #end
    
    def get_mask(self, remove_nans = True):
        
        mask = torch.zeros_like(self.data)
        mask[self.data.isnan().logical_not()] = 1.
        mask[self.data == 0] = 0
        
        if remove_nans:
            self.remove_nans()
        #end
        
        return mask
    #end
    
    def get_nitem(self):
        
        self_mask = self.get_mask()
        num_features = self_mask.shape[-1]
        return self_mask.sum().div(num_features)
    #end
    
    def set_nparams(self, nparams):
        
        self.nparams = nparams
    #end
    
    def denormalize(self, inplace = False):
        
        denorm_data = (self.data - self.nparams[0]) / (self.nparams[1] - self.nparams[0])
        
        if inplace:
            self.data = denorm_data
        else:
            return denorm_data
        #end
    #end
    
    def forward(self):
        
        return self.data
    #end
#end

