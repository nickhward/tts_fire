from torch import nn
import torch 
from models.hubert.waveform_encoder import WaveformEncoder

def test_waveform_encoder_output_shape():
    
    model = WaveformEncoder(kernel_list=[10,3,3,3,3,2,2], stride_list=[5,2,2,2,2,2,2])

    model.eval()

    batch_size = 4
    channels = 512
    time = 16000
    
    x = torch.randn(batch_size, channels, time)
    
    with torch.no_grad(): 
        y = model(x)

    print(dir(y))
    print(y.shape)
    print(y.ndim)
    assert y.ndim == 3
