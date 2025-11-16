from torch import nn
from model_utils.transpose import TransposeLastTwo


def get_conv_block(in_channel, out_channel, stride, kernel): 
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size=kernel, stride=stride), 
        TransposeLastTwo(), 
        nn.LayerNorm(out_channel),
        TransposeLastTwo(), 
        nn.GELU()
    )
            

class WaveformEncoder(nn.Module): 
    def __init__(
        self,
        stride_list=[10,3,3,3,3,2,2],
        kernel_list=[5,2,2,2,2,2,2],
        out_channel=512
    ):
        
        super().__init__()
        in_channel = 1 
        conv_blocks = [] 
        for stride, kernel in zip(stride_list, kernel_list): 

            conv_block = get_conv_block(in_channel, out_channel, stride, kernel)
            conv_blocks.append(conv_block)
            in_channel = out_channel 


        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(self, x): 

        for conv_block in self.conv_blocks: 
            x = conv_block(x)

        return x 


        

        

