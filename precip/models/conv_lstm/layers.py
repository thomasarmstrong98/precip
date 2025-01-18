from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor, nn


from precip.models.unet.layers import unet_up_collate


class ConvLSTMCell(nn.Module):
    """ConvLSTM Cell"""

    def __init__(
        self,
        input_channel: int,
        hidden_channel: int,
        kernel_size: int,
        bias=True,
        activation=torch.tanh,
        normalize=False,
    ):
        """
        ConLSTM Cell

        Args:
            input_channel: Number of input channels
            hidden_channel: Number of hidden channels
            kernel_size: Kernel size
            bias: Whether to add bias
            activation: Activation to use
            batchnorm: Whether to use batch norm
        """
        super().__init__()

        self.input_channel = input_channel
        self.hidden_channel = hidden_channel

        self.kernel_size = kernel_size
        self.bias = bias
        self.activation = activation
        if normalize:
            self.normalisation = nn.Identity()  # TODO
        else:
            self.normalisation = nn.Identity()

        self.conv = nn.Conv2d(
            in_channels=self.input_channel + self.hidden_channel,
            out_channels=4 * self.hidden_channel,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size // 2, self.kernel_size // 2),
            bias=self.bias,
            padding_mode="replicate",
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

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channel, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)

        g = self.activation(cc_g)
        try:
            c_cur = f * c_prev + i * g
        except:
            print(f.shape)
            print(c_prev.shape)
            print(i.shape)
            print(g.shape)
            raise Exception()

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
        b, c, h, w = x.size()  # c = 1 even if using grayscale inputs
        state = (
            torch.zeros(b, self.hidden_channel, h, w),
            torch.zeros(b, self.hidden_channel, h, w),
        )
        state = (state[0].type_as(x), state[1].type_as(x))
        return state

    def reset_parameters(self) -> None:
        """Resets parameters, optimizing gain for tahn"""
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain("tanh"))
        if self.bias:
            self.conv.bias.data.zero_()


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        num_layers: int,
        bias: bool = False,
        activation=torch.tanh,
        return_state_history: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.num_layers = num_layers
        self.bias = bias
        self.return_state_history = return_state_history

        self.cells = nn.ModuleList([])

        for step in range(self.num_layers):
            self.cells.append(
                ConvLSTMCell(
                    input_channel=self.input_channels
                    if step == 0
                    else self.hidden_channels,
                    hidden_channel=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    bias=self.bias,
                    activation=activation,
                )
            )

    def forward(
        self, x: torch.Tensor, hidden_state: Optional[list] = None
    ) -> tuple[Tensor, Optional[list[tuple[Tensor, Tensor]]]]:
        """
        Computes the output of the ConvLSTM

        Args:
            x: Input Tensor of shape [Batch, Time, Channel, Width, Height]
            hidden_state: List of hidden states to use, if none passed, it will be generated

        Returns:
            The layer outputs and list of last states
        """
        time_sliced_input = torch.unbind(x, dim=1)  # T x (B, C, H, W)

        if hidden_state is None:
            hidden_state = self.cells[0].init_hidden(time_sliced_input[0])
        h, c = hidden_state

        if self.return_state_history:
            final_state_history = list()
        else:
            final_state_history = None

        for layer in range(self.num_layers):
            output_inner = list()

            for t in range(len(time_sliced_input)):
                h, c = self.cells[layer](x=time_sliced_input[t], prev_state=[h, c])
                output_inner.append(h)

            time_sliced_input = output_inner

            if self.return_state_history:
                final_state_history.append((h, c))

        layer_output = torch.stack(output_inner, dim=1)
        return layer_output, final_state_history

    def reset_parameters(self) -> None:
        """
        Reset parameters for each ConvLSTMCell
        """
        for c in self.cells:
            c.reset_parameters()


class DownConvLSTM(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_factor: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(pool_factor),
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
        )
        self.conv_lstm = ConvLSTM(
            hidden_channels,
            out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
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
        num_layers: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        upsample_size: Optional[Tuple[int, int]] = None,
        upsample_scale: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.conv_lstm = ConvLSTM(
            in_channels, out_channels, kernel_size=kernel_size, num_layers=num_layers
        )
        if upsample_size is not None:
            self.up_scale = nn.UpsamplingBilinear2d(size=upsample_size)
        elif upsample_scale is not None:
            self.up_scale = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        else:
            raise RuntimeError("Must specify either upsample_size or upsample_scale")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x1.size()
        x1 = rearrange(
            self.up_scale(rearrange(x1, "b t c h w -> (b t) c h w")),
            "(b t) c h w -> b t c h w",
            b=b,
            t=t,
            c=c,
        )
        x1 = unet_up_collate(x1, x2, dim=2)

        x1, _ = self.conv_lstm(x1)
        return x1
