import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _groups(channels):
    for g in [8, 6, 4, 3, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class AdaConv(nn.Module):
    """
    Paper-inspired AdaConv:

        AdaConv(X, Y)
        = MLP(Filter_X * X) ⊙ (Filter_Y * Y)

    X: sound-speed latent feature
    Y: wavefield / residual latent feature
    """

    def __init__(self, channels):
        super().__init__()

        self.filter_x = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.filter_y = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
            ),
        )

        self.mix = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

    def forward(self, x, y):
        gate = self.mlp(
            self.filter_x(x)
        )

        field = self.filter_y(y)

        return self.mix(
            gate * field
        )


class RefinementBlock(nn.Module):
    """
    Learnable smoother S_h.
    """

    def __init__(self, channels):
        super().__init__()

        self.ada = AdaConv(
            channels
        )

        self.norm = nn.GroupNorm(
            _groups(channels),
            channels,
        )

        # Small initial correction stabilizes recurrent V-cycles.
        self.logit_scale = nn.Parameter(
            torch.tensor(-2.0)
        )

    def forward(self, x, y):
        corr = self.ada(
            x,
            y,
        )

        corr = F.gelu(
            self.norm(corr)
        )

        scale = torch.sigmoid(
            self.logit_scale
        )

        return scale * corr


class PDEKernelBlock(nn.Module):
    """
    Learnable PDE-informed K_h approximation.
    """

    def __init__(self, channels):
        super().__init__()

        self.ada = AdaConv(
            channels
        )

        self.norm = nn.GroupNorm(
            _groups(channels),
            channels,
        )

    def forward(self, x, y):
        r = self.ada(
            x,
            y,
        )

        return self.norm(r)


class Restriction(nn.Module):
    """
    480 -> 239 -> 119 -> 59 -> 29 -> 14 -> 6

    Conv2d(kernel=3,stride=2,padding=0) gives
    exactly the paper's grid hierarchy.
    """

    def __init__(self, channels):
        super().__init__()

        self.op = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )

    def forward(self, x):
        return self.op(x)


class Prolongation(nn.Module):
    """
    Learned coarse-to-fine interpolation.

    output_padding is chosen to invert the exact
    paper grid hierarchy.
    """

    def __init__(
        self,
        channels,
        output_padding,
    ):
        super().__init__()

        self.op = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=0,
            output_padding=output_padding,
        )

    def forward(self, x):
        return self.op(x)


class MgNOVCycle(nn.Module):
    """
    Seven-level multigrid V-cycle.

    Fine-to-coarse:
      480, 239, 119, 59, 29, 14, 6
    """

    def __init__(
        self,
        channels=12,
    ):
        super().__init__()

        self.channels = channels
        self.num_levels = 7

        # PDE kernels K_h
        self.k_blocks = nn.ModuleList(
            [
                PDEKernelBlock(
                    channels
                )
                for _ in range(
                    self.num_levels
                )
            ]
        )

        # Pre/post smoothers S_h
        self.pre_smooth = nn.ModuleList(
            [
                RefinementBlock(
                    channels
                )
                for _ in range(
                    self.num_levels
                )
            ]
        )

        self.post_smooth = nn.ModuleList(
            [
                RefinementBlock(
                    channels
                )
                for _ in range(
                    self.num_levels - 1
                )
            ]
        )

        # Learned fine -> coarse maps.
        self.restrict_x = nn.ModuleList(
            [
                Restriction(
                    channels
                )
                for _ in range(6)
            ]
        )

        self.restrict_r = nn.ModuleList(
            [
                Restriction(
                    channels
                )
                for _ in range(6)
            ]
        )

        # Edge index:
        #
        # 0: 480 -> 239   reverse needs +1
        # 1: 239 -> 119   reverse +0
        # 2: 119 -> 59    reverse +0
        # 3: 59  -> 29    reverse +0
        # 4: 29  -> 14    reverse +0
        # 5: 14  -> 6     reverse +1
        #
        up_padding = [
            1,
            0,
            0,
            0,
            0,
            1,
        ]

        self.prolong = nn.ModuleList(
            [
                Prolongation(
                    channels,
                    output_padding=op,
                )
                for op in up_padding
            ]
        )

    def forward(
        self,
        x_fine,
        y_fine,
    ):
        # ------------------------------------------------------
        # Downward sweep
        # ------------------------------------------------------

        x_levels = [
            x_fine
        ]

        y_skips = []

        x = x_fine
        y = y_fine

        for level in range(6):

            # Pre-smoothing
            y = (
                y
                + self.pre_smooth[level](
                    x,
                    y,
                )
            )

            y_skips.append(
                y
            )

            # PDE residual-like feature
            r = self.k_blocks[level](
                x,
                y,
            )

            # Restrict residual and medium information.
            y = self.restrict_r[level](
                r
            )

            x = self.restrict_x[level](
                x
            )

            x_levels.append(
                x
            )

        # ------------------------------------------------------
        # Coarsest level: 6 x 6
        # ------------------------------------------------------

        y = (
            y
            + self.pre_smooth[6](
                x,
                y,
            )
        )

        y = (
            y
            + self.k_blocks[6](
                x,
                y,
            )
        )

        # ------------------------------------------------------
        # Upward sweep
        # ------------------------------------------------------

        for level in reversed(
            range(6)
        ):
            y = self.prolong[level](
                y
            )

            target = y_skips[
                level
            ]

            # Safety only. Correct output_padding should
            # already make these identical.
            if y.shape[-2:] != target.shape[-2:]:
                y = F.interpolate(
                    y,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # Coarse correction
            y = target + y

            # Post-smoothing
            y = (
                y
                + self.post_smooth[level](
                    x_levels[level],
                    y,
                )
            )

        return y


class MgNOBackgroundWavefield(nn.Module):
    """
    Diff-ANO-style MgNO input:

        sound-speed X0
        homogeneous background wavefield Y0

    Input:
        [B,3,480,480]

        channel 0:
            normalized speed

        channel 1:
            normalized background wave real

        channel 2:
            normalized background wave imag

    Output:
        [B,2,480,480]
    """

    def __init__(
        self,
        channels=12,
        recurrent_iters=4,
        use_checkpoint=True,
    ):
        super().__init__()

        self.channels = channels
        self.recurrent_iters = recurrent_iters
        self.use_checkpoint = (
            use_checkpoint
        )

        # Paper: physical inputs are lifted to
        # common latent channels.
        self.speed_lift = nn.Sequential(
            nn.Conv2d(
                1,
                channels,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
            ),
        )

        self.wave_lift = nn.Sequential(
            nn.Conv2d(
                2,
                channels,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
            ),
        )

        # IMPORTANT:
        # One V-cycle with shared parameters,
        # recurrently applied l times.
        self.vcycle = MgNOVCycle(
            channels=channels
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                2,
                kernel_size=1,
            ),
        )

        # Start close to the physical homogeneous
        # background solution.
        nn.init.zeros_(
            self.head[-1].weight
        )

        nn.init.zeros_(
            self.head[-1].bias
        )

    def forward(self, inp):
        if inp.ndim != 4:
            raise RuntimeError(
                f"Expected [B,3,H,W], got {inp.shape}"
            )

        if inp.shape[1] != 3:
            raise RuntimeError(
                f"Expected 3 input channels, got {inp.shape}"
            )

        speed = inp[
            :,
            0:1,
        ]

        bg = inp[
            :,
            1:3,
        ]

        x = self.speed_lift(
            speed
        )

        y = self.wave_lift(
            bg
        )

        for _ in range(
            self.recurrent_iters
        ):
            if (
                self.training
                and self.use_checkpoint
            ):
                # Compatible with older PyTorch:
                # re-computation trades compute for memory.
                y = checkpoint(
                    self.vcycle,
                    x,
                    y,
                )
            else:
                y = self.vcycle(
                    x,
                    y,
                )

        delta = self.head(
            y
        )

        # Physics-informed residual parameterization:
        # homogeneous background + learned scattering.
        return bg + delta


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


if __name__ == "__main__":

    model = MgNOBackgroundWavefield(
        channels=12,
        recurrent_iters=4,
        use_checkpoint=False,
    )

    print(
        "parameters =",
        f"{count_parameters(model):,}"
    )

    x = torch.randn(
        1,
        3,
        480,
        480,
    )

    with torch.no_grad():
        y = model(x)

    print(
        "input  =",
        tuple(x.shape)
    )

    print(
        "output =",
        tuple(y.shape)
    )

    assert y.shape == (
        1,
        2,
        480,
        480,
    )

    print(
        "[PASS] MgNO shape sanity check."
    )
