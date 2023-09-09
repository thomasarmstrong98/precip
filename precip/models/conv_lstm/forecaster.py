from torch import nn
import torch

from .model import ConvLSTM

convlstm_forecaster_params = [
    [
        OrderedDict({"deconv1_leaky_1": [192, 192, 4, 2, 1]}),
        OrderedDict({"deconv2_leaky_1": [192, 64, 5, 3, 1]}),
        OrderedDict(
            {
                "deconv3_leaky_1": [64, 8, 7, 5, 1],
                "conv3_leaky_2": [8, 8, 3, 1, 1],
                "conv3_3": [8, 1, 1, 1, 0],
            }
        ),
    ],
    [
        ConvLSTM(
            input_channel=192,
            num_filter=192,
            b_h_w=(batch_size, 16, 16),
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        ConvLSTM(
            input_channel=192,
            num_filter=192,
            b_h_w=(batch_size, 32, 32),
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        ConvLSTM(
            input_channel=64,
            num_filter=64,
            b_h_w=(batch_size, 96, 96),
            kernel_size=3,
            stride=1,
            padding=1,
        ),
    ],
]


class Forecaster(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()

        self.layers = nn.ModuleList([])

        c1 = nn.Sequential(
            nn.ConvTranspose2d(192, 192, 4, 2, 1),
            nn.LeakyReLU(0.05),
        )

        c2 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 3, 1),
            nn.LeakyReLU(0.05),
        )

        c3 = nn.Sequential(
            nn.ConvTranspose2d(64, 8, 7, 5, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(
                8,
                8,
                3,
                1,
                1,
            ),
            nn.LeakyReLU(0.05),
            nn.Conv2d(8, 1, 1, 1, 0),
        )
        
        conv_lstm1 = 

    def forward_by_stage(self, input, state, subnet, rnn):
        input, state_stage = rnn(input, state, seq_len=cfg.HKO.BENCHMARK.OUT_LEN)
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(
            input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3))
        )

        return input

        # input: 5D S*B*I*H*W

    def forward(self, hidden_states):
        input = self.forward_by_stage(
            None, hidden_states[-1], getattr(self, "stage3"), getattr(self, "rnn3")
        )
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(
                input,
                hidden_states[i - 1],
                getattr(self, "stage" + str(i)),
                getattr(self, "rnn" + str(i)),
            )
        return input


def make_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    conv_type: str = "normal",
) -> nn.Module:
    if conv_type == "normal":
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    elif conv_type == "transpose":
        return nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
    else:
        raise NotImplementedError()
