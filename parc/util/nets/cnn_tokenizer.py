import abc

import torch
import torch.nn as nn


class CNN_Tokenizer(nn.Module):

    def __init__(self, input_dict, activation):
        super().__init__()
        self._activation = activation
        self._read_config(input_dict)
        self._build_net()
        self._test_net()
        
        assert(isinstance(self._net, nn.Module))
        assert(isinstance(self._unfold, nn.Module))
        assert(isinstance(self._obs_dim_x, int))
        assert(isinstance(self._obs_dim_y, int))
        assert(isinstance(self._out_token_dim, int))
        assert(isinstance(self._num_out_tokens, int))

        return
    
    def _read_config(self, config):
        return

    @abc.abstractmethod
    def _build_net(self):
        return

    def _test_net(self):
        example_obs = torch.zeros(size=(1, 1, self._obs_dim_x,  self._obs_dim_y))
        example_conv_obs = self._net(example_obs)
        out_shape = example_conv_obs.shape[1:]
        print("**********obs conv net output shape:", out_shape, "************")

        self._cnn_out_shape = out_shape

        example_tokens = self._unfold(example_conv_obs)
        out_shape = example_tokens.shape[1:]
        print("**********obs conv net token shape:", out_shape, "************")

        self._out_token_dim = example_tokens.shape[1]
        self._num_out_tokens = example_tokens.shape[2]
        return

    def forward(self, x: torch.Tensor):

        x = self._net(x)
        x = self._unfold(x)
        x = x.permute(0, 2, 1) # swap token and channel dimensions

        return x
    
    def forward_no_unfold(self, x: torch.Tensor):
        return self._net(x)

def build_net(input_dict, activation):

    cnn = CNN_Tokenizer(input_dict, activation)

    return cnn, None