import torch
import torch.nn.functional as F
from torch import nn

from .layers.evolution import EvolutionNetwork
from .layers.generative import GenerativeDecoder, GenerativeEncoder, NoiseProjector


def make_grid(size: tuple[int, int, int, int], cuda: bool = False):
    b, c, h, w = size
    xx = torch.arange(0, w).view(1, -1).repeat(h, 1)
    yy = torch.arange(0, h).view(-1, 1).repeat(1, w)

    if cuda:
        xx = xx.cuda()
        yy = yy.cuda()

    xx = xx.view(1, 1, h, w).repeat(b, 1, 1, 1)
    yy = yy.view(1, 1, h, w).repeat(b, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    return grid


def warp(
    state: torch.Tensor,
    flow: torch.Tensor,
    grid: torch.Tensor,
    mode="bilinear",
    padding_mode="zeros",
):
    _, _, h, w = state.size()
    vgrid = grid + flow

    vgrid[:, 0, ...] = 2.0 * vgrid[:, 0, ...].clone() / max(w - 1, 1) - 1.0
    vgrid[:, 1, ...] = 2.0 * vgrid[:, 1, ...].clone() / max(h - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(
        state, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True
    )


class NowcastNet(nn.Module):
    def __init__(
        self, input_seq_len: int, output_seq_len: int, use_cuda: bool = True
    ) -> None:
        super().__init__()
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

        self.evolution_network = EvolutionNetwork(
            self.input_seq_len, self.output_seq_len, 4
        )
        self.generative_encoder = GenerativeEncoder(input_seq_len + output_seq_len)
        self.noise_projector = NoiseProjector(32)
        self.generative_decoder = GenerativeDecoder(
            input_channels=(self.generative_encoder.output_channel_size + 64),
            output_channels=self.output_seq_len,
        )
        self.use_cuda = use_cuda
        self.grid = make_grid((1, 1, 256, 256), cuda=use_cuda)

    def forward(self, x: torch.Tensor):
        b, t, h, w = x.size()
        intensity, motion = self.evolution_network(x)
        _motion = motion.reshape(b, self.output_seq_len, 2, h, w)
        _intensity = intensity.reshape(b, self.output_seq_len, 1, h, w)

        latest_frame = x[:, -1:, ...]
        grid = self.grid.repeat(b, 1, 1, 1)

        steps = []
        for i in range(self.output_seq_len):
            latest_frame = warp(
                latest_frame, _motion[:, i], grid, mode="nearest", padding_mode="border"
            )
            latest_frame = latest_frame + _intensity[:, i]
            steps.append(latest_frame)

        evolution_result = torch.cat(steps, dim=1)
        evo_feature = self.generative_encoder(torch.cat([x, evolution_result], dim=1))
        ngf = 32
        noise = torch.randn(b, ngf, 256 // 32, 256 // 32).to(
            "cuda" if self.use_cuda else "cpu"
        )
        noise_f = (
            self.noise_projector(noise)
            .reshape(1, -1, 4, 4, 8, 8)
            .permute(0, 1, 4, 5, 2, 3)
            .reshape(b, -1, 256 // 8, 256 // 8)
        )

        # build input to generative decoder
        feature = torch.cat([evo_feature, noise_f], dim=1)
        forecasts = self.generative_decoder(feature, evolution_result)

        # targets are in [0, 1], cast our forecasts into the same space
        return torch.sigmoid(forecasts)
