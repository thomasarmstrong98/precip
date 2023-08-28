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

        for _block, (conv_config, conv_lstm_config) in enumerate(model_params):
            self.conv_blocks.append(
                nn.Sequential(nn.Conv2d(**conv_config.__dict__), nn.LeakyReLU(negative_slope=0.2))
            )

            self.conv_lstm_blocks.append(ConvLSTM(**conv_lstm_config.__dict__))

    def forward_by_stage(self, x, subnet, rnn):
        batch_size, sequence_size, input_channel, height, width = x.size()
        x = torch.reshape(x, (-1, input_channel, height, width))
        x = subnet(x)
        x = torch.reshape(x, (seq_number, batch_size, x.size(1), x.size(2), x.size(3)))
        # hidden = torch.zeros((batch_size, rnn._cell._hidden_size, x.size(3), x.size(4))).to(cfg.GLOBAL.DEVICE)
        # cell = torch.zeros((batch_size, rnn._cell._hidden_size, x.size(3), x.size(4))).to(cfg.GLOBAL.DEVICE)
        # state = (hidden, cell)
        outputs_stage, state_stage = rnn(x, None)

        return outputs_stage, state_stage

    # input: 5D S*B*I*H*W
    def forward(self, x):
        hidden_states = []
        logging.debug(input.size())
        for block in range(self.n_blocks):
            input, state_stage = self.forward_by_stage(
                input, self.conv_blocks[block], self.conv_lstm_blocks[block]
            )
            hidden_states.append(state_stage)
        return tuple(hidden_states)
