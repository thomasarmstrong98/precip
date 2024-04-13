"""Originally adapted from https://github.com/aserdega/convlstmgru, MIT License Andriy Serdega"""

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class ConvLSTMCell(nn.Module):
    """ConvLSTM Cell"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        bias=True,
        activation=torch.tanh,
        batchnorm=False,
    ):
        """
        ConLSTM Cell

        Args:
            input_dim: Number of input channels
            hidden_dim: Number of hidden channels
            kernel_size: Kernel size
            bias: Whether to add bias
            activation: Activation to use
            batchnorm: Whether to use batch norm
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.bias = bias
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size // 2, self.kernel_size // 2),
            bias=self.bias,
            padding_mode="replicate",  # zero-padding causing issue for chained convs
        )

        self.reset_parameters()

    def forward(
        self, x: torch.Tensor, prev_state: list
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute forward pass

        Args:
            x: Input tensor of [Batch, Channel, Height, Width]
            prev_state: Previous hidden state

        Returns:
            The new hidden state and output
        """
        h_prev, c_prev = prev_state

        combined = torch.cat((x, h_prev), dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)

        g = self.activation(cc_g)
        c_cur = f * c_prev + i * g

        o = torch.sigmoid(cc_o)

        h_cur = o * self.activation(c_cur)

        return h_cur, c_cur

    def init_hidden(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the hidden state
        Args:
            x: Input tensor to initialize for

        Returns:
            Tuple containing the hidden states
        """
        b, t, c, h, w = x.size()  # c = 1 even if using grayscale inputs
        state = (
            torch.zeros(b, self.hidden_dim, h, w),
            torch.zeros(b, self.hidden_dim, h, w),
        )
        state = (state[0].type_as(x), state[1].type_as(x))
        return state

    def reset_parameters(self) -> None:
        """Resets parameters"""
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain("tanh"))
        self.conv.bias.data.zero_()


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        num_layers: int,
        bias=True,
        activation=torch.tanh,
        batchnorm=False,
    ):
        """
        ConvLSTM module

        Args:
            input_dim: Input dimension size
            hidden_dim: Hidden dimension size
            kernel_size: Kernel size
            num_layers: Number of layers
            bias: Whether to add bias
            activation: Activation function
            batchnorm: Whether to use batch norm
        """
        super(ConvLSTM, self).__init__()
        self.output_channels = hidden_dim
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        activation = self._extend_for_multilayer(activation, num_layers)

        if not len(kernel_size) == len(hidden_dim) == len(activation) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = True
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                    activation=activation[i],
                    batchnorm=batchnorm,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self, x: torch.Tensor, hidden_state: Optional[list] = None
    ) -> tuple[Tensor, list[tuple[Any, Any]]]:
        """
        Computes the output of the ConvLSTM

        Args:
            x: Input Tensor of shape [Batch, Time, Channel, Width, Height]
            hidden_state: List of hidden states to use, if none passed, it will be generated

        Returns:
            The layer output and list of last states
        """
        cur_layer_input = torch.unbind(x, dim=int(self.batch_first))

        if not hidden_state:
            hidden_state = self.get_init_states(x)

        seq_len = len(cur_layer_input)

        last_state_list = []

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    x=cur_layer_input[t], prev_state=[h, c]
                )
                output_inner.append(h)

            cur_layer_input = output_inner
            last_state_list.append((h, c))

        layer_output = torch.stack(output_inner, dim=int(self.batch_first))

        return layer_output, last_state_list

    def get_init_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Constructs the initial hidden states

        Args:
            x: Tensor to use for constructing state

        Returns:
            The initial hidden states for all the layers in the network
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(x))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        Extends a parameter for multiple layers

        Args:
            param: Parameter to copy
            num_layers: Number of layers

        Returns:
            The extended parameter
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class DownConvLSTM(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, kernel_size, stride, padding
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, hidden_channels, kernel_size, stride, padding
        )
        self.conv_lstm = ConvLSTM(
            hidden_channels, out_channels, kernel_size, num_layers=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.size()
        x = rearrange(x, "b t c h w -> (b t) c h w ")
        x = rearrange(self.conv(x), "(b t) c h w -> b t c h w", b=b, t=t)
        x, _ = self.conv_lstm(x)
        return x


class UpBiLinearConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        upsample_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()

        self.conv_lstm = ConvLSTM(in_channels, out_channels, kernel_size, 1)
        if upsample_size is not None:
            self.up_scale = nn.UpsamplingBilinear2d(size=upsample_size)
        else:
            self.up_scale = nn.UpsamplingBilinear2d(scale_factor=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.conv_lstm(x)
        b, t, c, h, w = x.size()
        x = rearrange(
            self.up_scale(rearrange(x, "b t c h w -> (b t c) h w")),
            "(b t c) h w -> b t c h w",
            b=b,
            t=t,
            c=c,
        )

        return x


class UNetConvLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.d1 = DownConvLSTM(1, 8, 64, 4, 2, 1)
        self.d2 = DownConvLSTM(64, 128, 128, 3, 2, 1)
        self.d3 = DownConvLSTM(128, 256, 256, 3, 2, 1)

        self.u3 = UpBiLinearConvLSTM(256, 128, 3)
        self.u2 = UpBiLinearConvLSTM(256, 32, 3)
        self.u1 = UpBiLinearConvLSTM(64, 8, 1)
        self.out = nn.Sequential(nn.Conv2d(8, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
