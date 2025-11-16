from torch import nn

class TransposeLastTwo(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x): 
        if x.ndim == 2: 
            x.unsqueeze(1)
        return x.transpose(1, 2)
