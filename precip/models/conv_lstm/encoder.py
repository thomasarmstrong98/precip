import logging
from dataclasses import dataclass

import torch
from torch import nn

from precip.models.conv_lstm.model import ConvLSTM


@dataclass
class EncoderConvParams:
    in_channels: int
    out_channels: int
    kernel_size: tuple[int]
    stride: tuple[int]
    padding: int


@dataclass
class EncoderConvLSTMParams:
    input_channel: int
    hidden_channel: int
    kernel_size: tuple[int]
    num_layers: int
    batch_first: bool = False
    bias: bool = True
    return_all_layers: bool = False


class Encoder(nn.Module):
    def __init__(self, model_params: list[tuple[EncoderConvParams, EncoderConvLSTMParams]]):
        super().__init__()

        self.n_blocks = len(model_params)
        self.conv_blocks = nn.ModuleList()
        self.conv_lstm_blocks = nn.ModuleList()

        for (conv_config, conv_lstm_config) in model_params:
            self.conv_blocks.append(
                nn.Sequential(nn.Conv2d(**conv_config.__dict__), nn.LeakyReLU(negative_slope=0.2))
            )

            self.conv_lstm_blocks.append(ConvLSTM(**conv_lstm_config.__dict__))

    def block_forward(self, x, subnet, rnn):
        batch_size, sequence_size, input_channel, height, width = x.size()
        x = torch.reshape(x, (-1, input_channel, height, width))
        x = subnet(x)
        x = torch.reshape(x, (sequence_size, batch_size, x.size(1), x.size(2), x.size(3)))
        outputs_stage, states = rnn(x, None)
        
        # only keep the last state
        states = states[-1]

        return outputs_stage, states

    # input: 5D S*B*I*H*W
    def forward(self, x):
        hidden_states = []
        
        for block in range(self.n_blocks):
            x, internal_states = self.block_forward(
                x, self.conv_blocks[block], self.conv_lstm_blocks[block]
            )
            hidden_states.append(internal_states)
        return tuple(hidden_states)
