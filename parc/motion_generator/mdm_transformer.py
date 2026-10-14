from typing import List

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import parc.util.nets.net_builder as net_builder
from parc.motion_generator.diffusion_util import (
    MLP,
    MDMKeyType,
    PositionalEncoding,
    TimestepEmbedder,
)


class MDMTransformer(nn.Module):
    # This model just takes the "encoder" part of the original transformer model.
    # It is not really "encoding" anything, and is just using the
    # transformer encoder architecture as a neural network for its
    # self-attention mechanism with positional awareness

    def __init__(self, out_dim: int, in_dim: int, d_model: int, num_heads: int, d_hid: int,
                 num_layers: int, gen_seq_len: int, diffusion_timesteps: int,
                 dropout: float = 0.5,
                 num_prev_states: int = 2,
                 use_obs = True,
                 use_target = True,
                 target_dim: int = 2,
                 target_mlp_layers = [512, 256],
                 cnn_cfg = dict(),
                 in_mlp_layers: List[int] = [1024],
                 out_mlp_layers: List[int] = [1024]):
        # out_dim is the dimension of the output
        # in_dim is the dimension of the input (for motion frames)
        # d_model is dimension input to the transformer after an MLP acts on the input
        # num_heads is the number of heads in the transformer
        # d_hid is the dimension of the feed forward MLP in the transformer layers
        # nlayers is the number of transformer layers
        # dropout is the dropout rate
        # obs_dim is the dimension of the observations

        super().__init__()
        self._model_type = 'Transformer'
        #self.gen_seq_pos_encoder = PositionalEncoding(d_model, dropout, gen_seq_len)
        
        # Get index information for condition tokens
        self._time_token_idx = 0
        self._num_cond_tokens = 1
        self._use_obs = use_obs
        if use_obs:

            cnn_input_dict = dict()

            self._obs_conv_net, info = net_builder.build_net(cnn_cfg["net_name"], 
                                                             cnn_input_dict,
                                                             torch.nn.LeakyReLU())

            #self._out_cnn_token_dim = info["out_token_dim"]
            self._obs_embed = nn.Linear(self._obs_conv_net._out_token_dim, d_model)

            self._obs_token_idx = self._num_cond_tokens
            self._num_cond_tokens += self._obs_conv_net._num_out_tokens


        self._use_target = use_target
        if use_target:
            mlp_target_dims = [target_dim]
            mlp_target_dims.extend(target_mlp_layers)
            mlp_target_dims.append(d_model)
            self._in_mlp_target = MLP(mlp_target_dims)
            self._target_token_idx = self._num_cond_tokens
            self._num_cond_tokens += 1
            print("target token idx:", self._target_token_idx)


        # prev state noise indicator token
        self._embed_prev_state_noise_indicator = nn.Embedding(2, d_model)
        self._prev_state_noise_indictator_idx = self._num_cond_tokens
        self._num_cond_tokens += 1

        self._num_prev_states = num_prev_states
        self._full_seq_len = self._num_cond_tokens + gen_seq_len
        
        self._in_pos_encoding = PositionalEncoding(d_model, dropout, seq_len=self._full_seq_len) 
        encoder_layers = TransformerEncoderLayer(d_model, num_heads, d_hid, dropout, activation="gelu", batch_first=True)
        
        
        self._transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self._d_model = d_model

        total_out_mlp_layers = [d_model] + out_mlp_layers + [out_dim]
        self._out_mlp = MLP(total_out_mlp_layers)
        self._embed_t = TimestepEmbedder(d_model, d_model, diffusion_timesteps)

        total_in_mlp_layers = [in_dim] + in_mlp_layers + [d_model]
        self._in_mlp_gen_seq = MLP(total_in_mlp_layers)
        return
    
    def create_key_padding_mask(self, batch_size, device):
        return torch.zeros(size=(batch_size, self._full_seq_len), dtype=torch.bool, device=device)

    def embed_conds(self, conds, key_padding_mask):
        tokens = []
        if self._use_obs:
            # expecting (batch_size, 1, dim_x, dim_y) for obs,
            # because we unsqueeze the sequence dim (1).
            # Coincidentally, this is the right shape to pass into the conv net
            obs = conds[MDMKeyType.OBS_KEY]

            obs_tokens = self._obs_conv_net(obs)
            obs_tokens = self._obs_embed(obs_tokens)

            num_obs_tokens = obs_tokens.shape[1]
            obs_mask = conds[MDMKeyType.OBS_FLAG_KEY]
            key_padding_mask[:, self._obs_token_idx:self._obs_token_idx+num_obs_tokens] = ~obs_mask.unsqueeze(-1)
            obs_tokens = obs_tokens * obs_mask.to(dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

            tokens.append(obs_tokens)

        if self._use_target:
            target_token = self._in_mlp_target(conds[MDMKeyType.TARGET_KEY])
            target_mask = conds[MDMKeyType.TARGET_FLAG_KEY]
            key_padding_mask[:, self._target_token_idx] = ~target_mask
            target_token = target_token * target_mask.to(dtype=torch.float32).unsqueeze(dim=-1).unsqueeze(dim=-1)
            tokens.append(target_token)

        
        ind = conds[MDMKeyType.PREV_STATE_NOISE_IND_KEY]
        prev_state_ind_token = self._embed_prev_state_noise_indicator(ind.to(dtype=torch.int64)).unsqueeze(dim=1) # unsqueeze sequence dim
        tokens.append(prev_state_ind_token)

        prev_state_mask = conds[MDMKeyType.PREV_STATE_FLAG_KEY]
        key_padding_mask[:, self._num_cond_tokens:self._num_cond_tokens+self._num_prev_states] = ~prev_state_mask.unsqueeze(-1)

        return tokens

    def forward(self, x, conds, t) -> torch.tensor:
        # x: a sequence of motion frames
        # conds: dictionary of conditions
        # t: the current diffusion timestep
        batch_size = x.shape[0]
        device = x.device
        key_padding_mask = self.create_key_padding_mask(batch_size, device)

        # Run inputs through nns to get tokens, which are input into the transformer
        t_token = self._embed_t(t)
        tokens = [t_token]

        cond_tokens = self.embed_conds(conds, key_padding_mask)
        tokens.extend(cond_tokens)
        
        ind = conds[MDMKeyType.PREV_STATE_NOISE_IND_KEY]
        noised_prev_state = x[:, 0:self._num_prev_states, :]
        clean_prev_state = conds[MDMKeyType.PREV_STATE_KEY][:, 0:self._num_prev_states, :]
        x[:, 0:self._num_prev_states, :] = torch.where(ind.unsqueeze(dim=-1).unsqueeze(dim=-1), 
                                                       clean_prev_state, noised_prev_state)

        gen_motion_tokens = self._in_mlp_gen_seq(x)
        tokens.append(gen_motion_tokens)
        input_tokens = torch.cat(tokens, dim=1)
        input_tokens = self._in_pos_encoding(input_tokens)

        output = self._transformer_encoder.forward(input_tokens, src_key_padding_mask=key_padding_mask)
        output = self._out_mlp(output)

        # our output has an extra tokens corresponding to the condition tokens.
        # We just ignore those tokens.
        return output[:, self._num_cond_tokens:, :]
    
    def fast_forward(self, x, conds, t, cond_tokens, key_padding_mask):
        # Run inputs through nns to get tokens, which are input into the transformer
        t_token = self._embed_t(t)
        tokens = [t_token]

        tokens.extend(cond_tokens)
        
        ind = conds[MDMKeyType.PREV_STATE_NOISE_IND_KEY]
        noised_prev_state = x[:, 0:self._num_prev_states, :]
        clean_prev_state = conds[MDMKeyType.PREV_STATE_KEY][:, 0:self._num_prev_states, :]
        x[:, 0:self._num_prev_states, :] = torch.where(ind.unsqueeze(dim=-1).unsqueeze(dim=-1), 
                                                       clean_prev_state, noised_prev_state)

        gen_motion_tokens = self._in_mlp_gen_seq(x)
        tokens.append(gen_motion_tokens)
        input_tokens = torch.cat(tokens, dim=1)
        input_tokens = self._in_pos_encoding(input_tokens)

        output = self._transformer_encoder.forward(input_tokens, src_key_padding_mask=key_padding_mask)
        output = self._out_mlp(output)

        # our output has an extra tokens corresponding to the condition tokens.
        # We just ignore those tokens.
        return output[:, self._num_cond_tokens:, :]