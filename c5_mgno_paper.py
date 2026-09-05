import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# =============================================================================
# Diff-ANO-style AdaConv
# =============================================================================

class AdaConv(nn.Module):
    """
    Diff-ANO Eq. (21)-style adaptive convolution:

        AdaConv(X, Y)
        = MLP(Filter_X * X) ⊙ (Filter_Y * Y)

    The paper states:
      - Filter_X / Filter_Y: 3x3 convolution
      - MLP: two hidden layers
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

        # Two hidden layers:
        #
        # input -> hidden1 -> hidden2 -> output
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
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
            ),
        )

    def forward(self, x, y):
        adaptive_filter = self.mlp(
            self.filter_x(x)
        )

        primary = self.filter_y(y)

        return (
            adaptive_filter
            * primary
        )


class PDEKernel(nn.Module):
    """
    Learnable K_h(X_h, U_h).
    """

    def __init__(self, channels):
        super().__init__()

        self.op = AdaConv(
            channels
        )

    def forward(self, x, u):
        return self.op(
            x,
            u,
        )


class Smoother(nn.Module):
    """
    Learnable S_h(X_h, r_h).

    Multigrid update:
        U <- U + S_h(X, residual)
    """

    def __init__(self, channels):
        super().__init__()

        self.op = AdaConv(
            channels
        )

        # Stabilize recurrent multigrid iterations.
        self.raw_scale = nn.Parameter(
            torch.tensor(-2.0)
        )

    def forward(self, x, residual):
        scale = torch.sigmoid(
            self.raw_scale
        )

        return (
            scale
            * self.op(
                x,
                residual,
            )
        )


# =============================================================================
# Grid transfer
# =============================================================================

class Restrict(nn.Module):
    """
    kernel=3, stride=2:

        480 -> 239
        239 -> 119
        119 -> 59
        59  -> 29
        29  -> 14
        14  -> 6
    """

    def __init__(self, channels):
        super().__init__()

        self.op = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        return self.op(x)


class Prolong(nn.Module):

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
            bias=False,
        )

    def forward(self, x):
        return self.op(x)


# =============================================================================
# One shared seven-level V-cycle
# =============================================================================

class DiffANOVCycle(nn.Module):
    """
    Seven grids:

        480
        239
        119
        59
        29
        14
        6

    State variables:

        X_h : sound-speed latent
        F_h : source / forcing latent
        U_h : current wavefield latent

    Update follows the MgNO / Diff-ANO residual form:

        r_h = F_h - K_h(X_h, U_h)
        U_h = U_h + S_h(X_h, r_h)
    """

    def __init__(
        self,
        channels=12,
    ):
        super().__init__()

        self.channels = channels
        self.num_levels = 7

        # K_h
        self.K = nn.ModuleList(
            [
                PDEKernel(
                    channels
                )
                for _ in range(
                    self.num_levels
                )
            ]
        )

        # pre-smoother
        self.S_pre = nn.ModuleList(
            [
                Smoother(
                    channels
                )
                for _ in range(
                    self.num_levels
                )
            ]
        )

        # post-smoother
        self.S_post = nn.ModuleList(
            [
                Smoother(
                    channels
                )
                for _ in range(
                    self.num_levels - 1
                )
            ]
        )

        # Fine -> coarse operators.
        #
        # We separately restrict:
        #   X
        #   U
        #   residual F-KU
        self.Rx = nn.ModuleList(
            [
                Restrict(
                    channels
                )
                for _ in range(6)
            ]
        )

        self.Ru = nn.ModuleList(
            [
                Restrict(
                    channels
                )
                for _ in range(6)
            ]
        )

        self.Rr = nn.ModuleList(
            [
                Restrict(
                    channels
                )
                for _ in range(6)
            ]
        )

        # Exact reverse sizes:
        #
        # 239 -> 480 : output_padding=1
        # 119 -> 239 : 0
        # 59  -> 119 : 0
        # 29  -> 59  : 0
        # 14  -> 29  : 0
        # 6   -> 14  : 1
        output_padding = [
            1,
            0,
            0,
            0,
            0,
            1,
        ]

        self.P = nn.ModuleList(
            [
                Prolong(
                    channels,
                    op,
                )
                for op in output_padding
            ]
        )

    def residual(
        self,
        level,
        x,
        f,
        u,
    ):
        return (
            f
            - self.K[level](
                x,
                u,
            )
        )

    def forward(
        self,
        x,
        f,
        u,
    ):
        # ------------------------------------------------------------
        # Downward sweep
        # ------------------------------------------------------------

        fine_states = []

        for level in range(6):

            # r = f - K(x,u)
            r = self.residual(
                level,
                x,
                f,
                u,
            )

            # pre-smoothing
            u = (
                u
                + self.S_pre[level](
                    x,
                    r,
                )
            )

            # Recompute residual after smoothing.
            r = self.residual(
                level,
                x,
                f,
                u,
            )

            # Store fine-grid state for upward correction.
            fine_states.append(
                (
                    x,
                    f,
                    u,
                )
            )

            # Classical multigrid-inspired restriction:
            #
            #   u_2h = Pi(u_h)
            #   f_2h = R(f_h - K_h u_h)
            #
            u = self.Ru[level](
                u
            )

            f = self.Rr[level](
                r
            )

            x = self.Rx[level](
                x
            )

        # ------------------------------------------------------------
        # Coarsest level: 6x6
        # ------------------------------------------------------------

        level = 6

        r = self.residual(
            level,
            x,
            f,
            u,
        )

        u = (
            u
            + self.S_pre[level](
                x,
                r,
            )
        )

        # ------------------------------------------------------------
        # Upward sweep
        # ------------------------------------------------------------

        for level in reversed(
            range(6)
        ):
            x_fine, f_fine, u_fine = (
                fine_states[level]
            )

            correction = self.P[level](
                u
            )

            if (
                correction.shape[-2:]
                != u_fine.shape[-2:]
            ):
                raise RuntimeError(
                    "Grid-size mismatch at level "
                    f"{level}: "
                    f"coarse correction="
                    f"{correction.shape}, "
                    f"fine={u_fine.shape}"
                )

            # Add coarse correction.
            u = (
                u_fine
                + correction
            )

            # Return to fine-grid physical states.
            x = x_fine
            f = f_fine

            # post-smoothing
            r = self.residual(
                level,
                x,
                f,
                u,
            )

            u = (
                u
                + self.S_post[level](
                    x,
                    r,
                )
            )

        return u


# =============================================================================
# Full MgNO
# =============================================================================

class MgNOBackgroundWavefield(nn.Module):
    """
    Diff-ANO-inspired physical input:

        X0 : sound speed
        Y0 : homogeneous-background wavefield

    Three latent states are constructed:

        X latent : medium state
        F latent : forcing/source state
        U latent : wavefield solution state

    A single V-cycle G_theta is recurrently reused.

    MgNO-I:
        channels = 12
        recurrent_iters = 4

    MgNO-II:
        channels = 24
        recurrent_iters = 6
    """

    def __init__(
        self,
        channels=12,
        recurrent_iters=4,
        use_checkpoint=True,
    ):
        super().__init__()

        self.channels = channels

        self.recurrent_iters = (
            recurrent_iters
        )

        self.use_checkpoint = (
            use_checkpoint
        )

        # ------------------------------------------------------------
        # L operator:
        # physical input -> three latent states
        # ------------------------------------------------------------

        self.x_lift = nn.Sequential(
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

        # Source identity is encoded by the homogeneous
        # background wavefield.
        self.f_lift = nn.Sequential(
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

        # Initial wavefield latent.
        self.u_lift = nn.Sequential(
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

        # ------------------------------------------------------------
        # Shared G_theta
        # ------------------------------------------------------------

        self.vcycle = DiffANOVCycle(
            channels=channels
        )

        # ------------------------------------------------------------
        # P operator
        # ------------------------------------------------------------

        self.project = nn.Sequential(
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

        # Start from exact homogeneous background.
        nn.init.zeros_(
            self.project[-1].weight
        )

        nn.init.zeros_(
            self.project[-1].bias
        )

    def forward(self, inp):

        if inp.ndim != 4:
            raise RuntimeError(
                f"Expected [B,3,H,W], got "
                f"{inp.shape}"
            )

        if inp.shape[1] != 3:
            raise RuntimeError(
                f"Expected 3 channels, got "
                f"{inp.shape[1]}"
            )

        speed = inp[
            :,
            0:1,
        ]

        background = inp[
            :,
            1:3,
        ]

        # Three latent states.
        x = self.x_lift(
            speed
        )

        f = self.f_lift(
            background
        )

        u = self.u_lift(
            background
        )

        # Shared recurrent V-cycle.
        for _ in range(
            self.recurrent_iters
        ):

            if (
                self.training
                and self.use_checkpoint
            ):
                u = checkpoint(
                    self.vcycle,
                    x,
                    f,
                    u,
                    use_reentrant=False,
                )

            else:
                u = self.vcycle(
                    x,
                    f,
                    u,
                )

        delta = self.project(
            u
        )

        # Learn the heterogeneous scattering correction
        # relative to homogeneous background solution.
        return (
            background
            + delta
        )


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


if __name__ == "__main__":

    torch.manual_seed(
        20260904
    )

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
        "[PASS] residual MgNO V-cycle "
        "shape sanity check."
    )
