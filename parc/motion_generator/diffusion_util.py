import enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeZStyle(enum.Enum):
    RELATIVE_TO_ROOT = 0
    RELATIVE_TO_ROOT_FLOOR = 1

class MDMFrameType(enum.Enum):
    ROOT_POS = 0
    ROOT_ROT = 1
    JOINT_POS = 2
    JOINT_VEL = 3
    JOINT_ROT = 4
    CONTACTS = 5
    FLOOR_HEIGHTS = 6

class MDMKeyType(enum.Enum):
    OBS_KEY = 0
    PREV_STATE_KEY = 1
    PREV_STATE_FLAG_KEY = 2
    TARGET_KEY = 3
    TARGET_FLAG_KEY = 4
    FINAL_STATE_KEY = 5
    PREV_STATE_NOISE_IND_KEY = 6
    OBS_FLAG_KEY = 8
    GUIDANCE_PARAMS = 9
    ## for CondMDI
    NOISE_IND_KEY = 10
    CANON_FRAME_IDX = 11
    CLEAN_STATE_KEY = 12

class MDMCustomGuidance:
    obs_cfg_scale = None

class TargetInfo:
    def __init__(self, future_pos: torch.Tensor, future_rot: torch.Tensor):
        self.future_pos = future_pos
        self.future_rot = future_rot
        return

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    betas[0] = beta_1 according to this
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * 3.1415927410125732 * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]) # final shape is [timesteps]
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = 0.0001 * scale
    beta_end = 0.02 * scale
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

class DiffusionRates:
    def __init__(self, timesteps, device):
        # define beta schedule
        self.betas = cosine_beta_schedule(timesteps=timesteps).to(device)

        # define alphas 
        self.alphas = 1. - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#reverse-diffusion-process
        # tilde{mu}(x_t, x_0)
        self.posterior_std = torch.sqrt(self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.posterior_mean_coef1 = self.sqrt_alphas_cumprod_prev * self.betas / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * self.sqrt_alphas / (1.0 - self.alphas_cumprod)

        return
    
class MLP(nn.Module):
    def __init__(self, dims, hidden_act = nn.SiLU(), device="cpu", dropout=0.0):
        super().__init__()

        layers = []

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Dropout(dropout))
            layers.append(hidden_act)
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.model = nn.Sequential(*layers)
        self.to(device)

        return
    def forward(self, x):

        return self.model(x)
    
class MotionDiffMLP(nn.Module):
    def __init__(self, hidden_dims, diffusion_timesteps, code_dim, t_emb_size, x_emb_size, state_dim, seq_len, device):

        super(MotionDiffMLP, self).__init__()

        layers = []
        code_emb_size = t_emb_size
        dims = [t_emb_size + code_emb_size + seq_len*x_emb_size] + hidden_dims# + [state_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(dims[-1], seq_len*state_dim))

        self.embed_t = TimestepEmbedder(t_emb_size, t_emb_size, diffusion_timesteps)
        self.embed_code = nn.Linear(code_dim, t_emb_size)

        self.embed_x = nn.ModuleList([nn.Linear(state_dim, x_emb_size) for i in range(seq_len)])

        self.model = nn.Sequential(*layers)
        self.to(device)
        return

    def forward(self, x, t, a):
        time_emb = self.embed_t(t)
        code_emb = self.embed_code(a)
        x_embs = []
        for i in range(x.shape[1]):
            self.embed_x[i](x[:, i, :])
            x_embs.append(self.embed_x[i](x[:, i, :]))
        x_emb = torch.cat(x_embs, dim=-1)
        x_emb = x_emb.unsqueeze(dim=1)
        model_input = torch.cat((x_emb, time_emb, code_emb), dim=-1)
        output = self.model(model_input)
        output = output.reshape(output.shape[0], -1, x.shape[-1])
        return output

class EmbedAction(nn.Module):
    # maps an action index to a vector
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, idx):
        output = self.action_embedding[idx]
        return output
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 2, dropout: float = 0.1, seq_len: int = 26):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)
    
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, output_dim, total_timesteps):
        super().__init__()
        self.latent_dim = latent_dim

        # we are just using the position buffer, not the forward method of PositionalEncoding
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, 0.0, total_timesteps) # [0, T-1]

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, output_dim),
        )

    def forward(self, timesteps):
        # timesteps is B x 1 x 1
        # sequence_pos_encoder.pe is 1 x T x d_model
        # We just want the timesteps to access the T dim of the pos encoder array
        pos_encoded_time = self.sequence_pos_encoder.pe[0, timesteps.squeeze(-1).squeeze(-1), :]
        # this will be B x d_model
        pos_encoded_time = pos_encoded_time.unsqueeze(dim=1)
        # now it will be B x 1 x d_model, which is the shape we want
        return self.time_embed(pos_encoded_time)