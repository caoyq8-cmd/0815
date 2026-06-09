# dds_core.py
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass
class DDSConfig:
    num_steps: int = 20
    cg_iters: int = 5
    dc_lambda: float = 1.0
    prior_lambda: float = 0.1
    clamp_min: float = 1400.0
    clamp_max: float = 1700.0
    use_ddim_like_schedule: bool = True
    verbose: bool = True


def default_time_schedule(num_steps: int, device: torch.device) -> Tensor:
    """
    一个简单的从 1 -> 0 的时间步调度。
    如果你已有现成 diffusion scheduler，可以替换这里。
    """
    t = torch.linspace(1.0, 1e-3, num_steps + 1, device=device)
    return t


def finite_diff_grad(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    eps: float = 1e-3,
) -> Tensor:
    """
    数值差分梯度，仅用于兜底调试，不建议正式跑大实验。
    正式实验请优先使用 autograd 的 measurement loss。
    """
    x = x.detach()
    grad = torch.zeros_like(x)
    flat_x = x.view(-1)
    flat_g = grad.view(-1)

    for i in range(flat_x.numel()):
        old = flat_x[i].item()

        flat_x[i] = old + eps
        fp = f(x).item()

        flat_x[i] = old - eps
        fm = f(x).item()

        flat_x[i] = old
        flat_g[i] = (fp - fm) / (2.0 * eps)

    return grad


def cg_solve(
    A_fn: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Optional[Tensor] = None,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> Tensor:
    """
    共轭梯度法，求解 A x = b
    假设 A 对称正定。这里用于线性化后的 data-consistency 子问题。
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - A_fn(x)
    p = r.clone()
    rs_old = torch.sum(r * r)

    for _ in range(max_iter):
        Ap = A_fn(p)
        denom = torch.sum(p * Ap).clamp_min(1e-12)
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r)
        if torch.sqrt(rs_new) < tol:
            break
        beta = rs_new / rs_old.clamp_min(1e-12)
        p = r + beta * p
        rs_old = rs_new

    return x


def estimate_x0_from_score(
    x_t: Tensor,
    t: Tensor,
    score_model: torch.nn.Module,
    sigma_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    """
    从 score 模型得到 x0 估计。
    这里写成一个通用模板：
      score = ∇_x log p_t(x_t)
      x0_hat ≈ x_t + sigma(t)^2 * score(x_t, t)

    注意：
    1) 这不是唯一形式；
    2) 你需要按照你当前训练的 score/DDPM 模型定义去改。
    """
    score = score_model(x_t, t)
    sigma_t = sigma_fn(t).view(-1, 1, 1, 1)
    x0_hat = x_t + sigma_t ** 2 * score
    return x0_hat


def ddim_like_denoise_step(
    x_t: Tensor,
    x0_hat: Tensor,
    t_cur: Tensor,
    t_next: Tensor,
) -> Tensor:
    """
    一个非常简化的 DDIM-like 更新。
    如果你仓库已有现成 predictor/update，请优先替换为正式版本。
    """
    alpha = t_next / t_cur.clamp_min(1e-6)
    alpha = alpha.view(-1, 1, 1, 1)
    x_next = alpha * x_t + (1.0 - alpha) * x0_hat
    return x_next


def build_linearized_dc_system(
    x_prior: Tensor,
    y: Tensor,
    forward_op: Callable[[Tensor], Tensor],
    adjoint_op: Callable[[Tensor], Tensor],
    dc_lambda: float,
    prior_lambda: float,
) -> Tuple[Callable[[Tensor], Tensor], Tensor]:
    """
    构造线性化 data-consistency 子问题：

      min_x  0.5 * dc_lambda * ||A x - y||^2 + 0.5 * prior_lambda * ||x - x_prior||^2

    对应正规方程：
      (dc_lambda * A^T A + prior_lambda * I) x
      = dc_lambda * A^T y + prior_lambda * x_prior
    """
    rhs = dc_lambda * adjoint_op(y) + prior_lambda * x_prior

    def A_fn(z: Tensor) -> Tensor:
        return dc_lambda * adjoint_op(forward_op(z)) + prior_lambda * z

    return A_fn, rhs


def dds_reconstruct(
    y: Tensor,
    x_init: Tensor,
    score_model: torch.nn.Module,
    sigma_fn: Callable[[Tensor], Tensor],
    forward_op: Callable[[Tensor], Tensor],
    adjoint_op: Callable[[Tensor], Tensor],
    cfg: DDSConfig,
) -> Dict[str, Tensor]:
    """
    DDS 主循环：
    1. 从当前 x_t 得到 x0_hat（扩散先验/去噪估计）
    2. 解 data-consistency 子问题，得到 x_dc
    3. 用 DDIM-like 方式更新到下一步

    返回：
      recon: 最终重建
      history_meas: 每步 measurement loss
      history_prior: 每步 prior loss
    """
    device = x_init.device
    times = default_time_schedule(cfg.num_steps, device=device)

    x_t = x_init.clone()
    history_meas = []
    history_prior = []

    for k in range(cfg.num_steps):
        t_cur = times[k].expand(x_t.shape[0])
        t_next = times[k + 1].expand(x_t.shape[0])

        # 1) 扩散先验给出 x0_hat
        x0_hat = estimate_x0_from_score(
            x_t=x_t,
            t=t_cur,
            score_model=score_model,
            sigma_fn=sigma_fn,
        )

        x0_hat = torch.clamp(x0_hat, cfg.clamp_min, cfg.clamp_max)

        # 2) data consistency 子问题：用 CG 求解
        A_fn, rhs = build_linearized_dc_system(
            x_prior=x0_hat,
            y=y,
            forward_op=forward_op,
            adjoint_op=adjoint_op,
            dc_lambda=cfg.dc_lambda,
            prior_lambda=cfg.prior_lambda,
        )

        x_dc = cg_solve(
            A_fn=A_fn,
            b=rhs,
            x0=x0_hat,
            max_iter=cfg.cg_iters,
            tol=1e-6,
        )

        x_dc = torch.clamp(x_dc, cfg.clamp_min, cfg.clamp_max)

        # 记录损失
        with torch.no_grad():
            meas_loss = F.mse_loss(forward_op(x_dc), y).item()
            prior_loss = F.mse_loss(x_dc, x0_hat).item()
            history_meas.append(meas_loss)
            history_prior.append(prior_loss)

            if cfg.verbose:
                print(
                    f"[DDS] step={k+1:03d}/{cfg.num_steps} "
                    f"t={t_cur[0].item():.4f} "
                    f"meas={meas_loss:.6e} "
                    f"prior={prior_loss:.6e} "
                    f"x_range=({x_dc.min().item():.2f}, {x_dc.max().item():.2f})"
                )

        # 3) 更新到下一步 latent
        if cfg.use_ddim_like_schedule:
            x_t = ddim_like_denoise_step(
                x_t=x_t,
                x0_hat=x_dc,
                t_cur=t_cur,
                t_next=t_next,
            )
        else:
            x_t = x_dc

        x_t = torch.clamp(x_t, cfg.clamp_min, cfg.clamp_max)

    return {
        "recon": x_t,
        "history_meas": torch.tensor(history_meas, device=device),
        "history_prior": torch.tensor(history_prior, device=device),
    }