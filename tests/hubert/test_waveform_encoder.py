from torch import nn
import torch 
from models.hubert.waveform_encoder import WaveformEncoder

def test_waveform_encoder_output_shape():
    
    model = WaveformEncoder(kernel_list=[10,3,3,3,3,2,2], stride_list=[5,2,2,2,2,2,2])

    model.eval()

    batch_size = 4
    sample_rate = 16000
    channel = 1

    
    x = torch.randn(batch_size,channel, sample_rate)
    
    with torch.no_grad(): 
        y = model(x)

    assert y.ndim == 3

