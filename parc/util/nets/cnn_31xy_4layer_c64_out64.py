import torch
import torch.nn as nn

import parc.util.nets.cnn_tokenizer as cnn_tokenizer


class CNN_Tokenizer_Messy3Layers(cnn_tokenizer.CNN_Tokenizer):
    def __init__(self, input_dict, activation):
        super().__init__(input_dict, activation)
        return
    
    def _build_net(self):
        layers = [
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0), # 27x27
            self._activation,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0), # 23x23
            self._activation,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0), # 19x19
            self._activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0), # 16x16
            self._activation,
        ]
        self._net = nn.Sequential(*layers)

        # final shape is 16 by 16

        self._unfold = nn.Unfold(kernel_size = 2,
                                 stride = 2,
                                 padding = 0,
                                 dilation = 1) # 8x8 final tokens of shape 64x2x2
        
        self._obs_dim_x = 31
        self._obs_dim_y = 31
        return

def build_net(input_dict, activation):

    cnn = CNN_Tokenizer_Messy3Layers(input_dict, activation)
    return cnn, None