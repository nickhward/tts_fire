from torch import nn
class WaveformEncoder(nn.Module): 
    def __init__(self, stride_list, kernel_list): 
        super(WaveformEncoder, self).__init__()
        layers = []        
        for stride, kernel in zip(stride_list, kernel_list): 
            layers += [nn.Conv1d(512, 512, kernel_size=kernel, stride=stride)]

        self.convs = nn.Sequential(*layers)


    def forward(self,x):
        return self.convs(x)        

     
