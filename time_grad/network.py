import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from .epsilon_theta import EpsilonTheta
from .gaussian_diffusion import GaussianDiffusion
from .scaler import MeanScaler
from .distribution_output import DiffusionOutput

class TimeGrad(nn.Module):
    def __init__(
        self,
        num_features: int, # including covariates
        num_target_features: int, 
        context_length: int,
        non_covariate_col_idx: List[int],
        hidden_size: int = 40,
        num_layers: int = 2,
        num_cells: int = 40,
        dropout_rate: float = 0.1,
        diff_steps: int = 100,
        beta_end: float = 0.1,
        beta_schedule: str = "linear",
        residual_layers: int = 8,
        residual_channels: int = 8,
        dilation_cycle_length: int = 2,
        residual_hidden: int = 64,
        conditioning_length: int = 100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.num_target_features = num_target_features
        self.context_length = context_length
        self.non_covariate_col_idx = non_covariate_col_idx

        self.rnn = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True
        )

        self.denoise_fn = EpsilonTheta(
            target_dim=num_target_features,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
            residual_hidden=residual_hidden
        )

        self.diffusion = GaussianDiffusion(
            self.denoise_fn,
            input_size=num_target_features,
            diff_steps=diff_steps,
            loss_type='l2',
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

        self.distr_output = DiffusionOutput(
            self.diffusion, input_size=num_target_features, cond_size=conditioning_length
        )

        self.proj_dist_args = self.distr_output.get_args_proj(num_cells)

        self.scaler = MeanScaler(keepdim=True)

    def unroll(
        self, 
        time_feat: torch.Tensor,
        begin_state: Optional[Union[List[torch.Tensor], torch.Tensor]] = None
    ):
        # unroll encoder
        outputs, state = self.rnn(time_feat, begin_state)
        return outputs, state, time_feat

    def unroll_encoder(self, time_feat):
        # sequence = past_target_cdf
        # sequence_length = self.history_length

        # (batch_size, sub_seq_len, target_dim, num_lags)
        # lags = self.get_lagged_subsequences(
        #     sequence=sequence,
        #     sequence_length=sequence_length,
        #     indices=self.lags_seq,
        #     subsequences_length=subsequences_length,
        # )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, target_dim)
        time_feat_non_covariates = time_feat[:, :, self.non_covariate_col_idx]
        _, scale = self.scaler(
            time_feat_non_covariates,
            torch.ones_like(time_feat_non_covariates)
        )

        outputs, states, inputs = self.unroll(
            time_feat=time_feat,
            begin_state=None
        )

        return outputs, states, scale, inputs

    def distr_args(self, rnn_outputs: torch.Tensor):
        (distr_args,) = self.proj_dist_args(rnn_outputs)

        # # compute likelihood of target given the predicted parameters
        # distr = self.distr_output.distribution(distr_args, scale=scale)

        # return distr, distr_args
        return distr_args

    def forward(self, time_feat, y):
        rnn_outputs, _, scale, _ = self.unroll_encoder(time_feat)

        distr_args = self.distr_args(rnn_outputs=rnn_outputs)
        self.diffusion.scale = scale

        # put together target sequence
        # (batch_size, seq_len, target_dim)
        # target = torch.cat((time_feat[:, :, self.non_covariate_col_idx], y), dim=1)

        likelihoods = self.diffusion.log_prob(y, rnn_outputs).unsqueeze(-1)
        loss = likelihoods.mean()
    
        return loss, likelihoods, distr_args


class TimeGradPredictionNetwork(TimeGrad):
    def __init__(self, num_parallel_samples: int, prediction_length: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.prediction_length = prediction_length

    def sampling_decoder(
        self,
        time_feat: torch.Tensor,
        scale: torch.Tensor,
        begin_states: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        # blows-up the dimension of each tensor to
        # batch_size * self.num_sample_paths for increasing parallelism
        repeated_past_target_cdf = repeat(time_feat[:, :, self.non_covariate_col_idx])
        repeated_time_feat = repeat(time_feat)
        repeated_scale = repeat(scale)
        self.diffusion.scale = repeated_scale
        # repeated_target_dimension_indicator = repeat(target_dimension_indicator)

        repeated_states = [repeat(s, dim=1) for s in begin_states]
        # future_samples = []

        # # for each future time-units we draw new samples for this time-unit
        # # and update the state
        # for k in range(self.prediction_length):
        #     rnn_outputs, repeated_states, _ = self.unroll(
        #         begin_state=repeated_states,
        #         time_feat=repeated_time_feat[:, k:k + 1, ...]
        #     )

        #     # distr_args = self.distr_args(rnn_outputs=rnn_outputs)

        #     # (batch_size, 1, target_dim)
        #     new_samples = self.diffusion.sample(cond=rnn_outputs)

        #     # (batch_size, seq_len, target_dim)
        #     future_samples.append(new_samples)
        #     repeated_past_target_cdf = torch.cat(
        #         (repeated_past_target_cdf, new_samples), dim=1
        #     )

        # # (batch_size * num_samples, prediction_length, target_dim)
        # samples = torch.cat(future_samples, dim=1)
        rnn_outputs, repeated_states, _ = self.unroll(
            begin_state=repeated_states,
            time_feat=time_feat
        )
        samples = self.diffusion.sample(cond=rnn_outputs.reshape((rnn_outputs.shape[1], 1, rnn_outputs.shape[2])))

        # (batch_size, num_samples, prediction_length, target_dim)
        return samples.reshape(
            (
                -1,
                self.num_parallel_samples,
                self.prediction_length,
                self.num_target_features,
            )
        )

    def forward(
        self, time_feat
    ) -> torch.Tensor:
        # unroll the decoder in "prediction mode", i.e. with past data only
        _, begin_states, scale, _ = self.unroll_encoder(time_feat)

        return self.sampling_decoder(
            time_feat=time_feat,
            scale=scale,
            begin_states=begin_states
        )
