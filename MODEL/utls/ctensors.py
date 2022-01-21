import numpy as np
import torch

class CTensor(torch.Tensor):

    def __init__(self, data):
        super(CTensor, self).__init__()
        
        self.data = torch.Tensor(data)
    #end
    
    def type_convert(self, dtype):
        
        self.data = torch.Tensor(self.data).type(dtype)
    #end
    
    def get_mask(self):
        
        mask = torch.zeros_like(self.data)
        mask[self.data.isnan().logical_not()] = 1.
        mask[self.data == 0] = 0
        self.data[self.data.isnan()] = 0.
        return mask
    #end
    
    def get_nitem(self):
        
        self_mask = self.get_mask()
        num_features = self_mask.shape[-1]
        return self_mask.sum().div(num_features)
    #end
    
    def forward(self):
        return self.data
    #end
#end


x = np.arange(25).reshape(5,5) * 1. + 1
x[[2,3],:] = np.nan
y = CTensor(x)
