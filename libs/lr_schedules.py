# -*- coding: utf-8 -*-
"""Custom LR schedules for Ultralytics training (epoch-level LambdaLR)."""
from __future__ import annotations

import math
from typing import Callable, Optional

LogFn = Callable[[str], None]

SCHEDULE_CHOICES = (
    ("linear", "Linear 线性衰减（默认）"),
    ("cosine", "Cosine 余弦退火（Ultralytics cos_lr）"),
    ("onecycle", "OneCycleLR 单周期（先升后降）"),
    ("truncated_cosine", "Truncated Cosine 截断余弦"),
)

SCHEDULE_LABELS = {k: v for k, v in SCHEDULE_CHOICES}


# OneCycleLR default parameters
ONECYCLE_MAX_LR = 3e-3
ONECYCLE_DIV_FACTOR = 20.0          # initial_lr = max_lr / div_factor
ONECYCLE_FINAL_DIV_FACTOR = 10000.0  # final_lr = max_lr / final_div_factor
ONECYCLE_PCT_START = 0.3            # warmup for first 30% of epochs
ONECYCLE_ANNEAL_STRATEGY = "cos"   # cosine annealing


def _onecycle_epoch_lf(lrf: float, epochs: int, pct_start: float = 0.3):
    """Epoch-level 1-cycle envelope: lr0×lrf → max_lr → lr0×lrf (Smith-style).

    Warmup: linear from min_lr (lr0×lrf) to max_lr over pct_start of total epochs.
    Anneal: cosine from max_lr back to min_lr for the remaining (1 - pct_start).
    
    min_lr = lr0 × lrf  (where lrf = 1 / final_div_factor)
    max_lr = lr0 × div_factor
    """
    min_lr_mult = max(1e-7, float(lrf))        # = 1 / final_div_factor
    max_lr_mult = 1.0                          # peak multiplier (lr0)
    warmup_epochs = max(1, int(epochs * max(0.05, min(0.95, pct_start))))

    def lf(epoch: int) -> float:
        if epochs <= 1:
            return min_lr_mult
        if epoch <= 0:
            return min_lr_mult
        t = epoch / (epochs - 1)               # fractional progress [0, 1]
        pct = epoch / epochs                   # fraction of total epochs
        if pct <= pct_start:
            # Linear warmup from min_lr to max_lr
            frac = pct / max(pct_start, 1e-6)
            return min_lr_mult + (max_lr_mult - min_lr_mult) * frac
        else:
            # Cosine annealing from max_lr to min_lr
            frac = (pct - pct_start) / max(1.0 - pct_start, 1e-6)
            return max_lr_mult - (max_lr_mult - min_lr_mult) * (0.5 * (1 + math.cos(math.pi * frac)))
        return min_lr_mult

    return lf


def _truncated_cosine_lf(lrf: float, epochs: int, tmax_frac: float):
    """Cosine decay for the first tmax_frac of epochs, then flat at lr0×lrf."""

    t_max = max(1, int(epochs * max(0.05, min(1.0, tmax_frac))))

    def lf(epoch: int) -> float:
        if epoch >= t_max:
            return lrf
        return lrf + (1.0 - lrf) * 0.5 * (1.0 + math.cos(math.pi * epoch / t_max))

    return lf


def build_lf(
    schedule: str,
    lrf: float,
    epochs: int,
    cos_tmax_frac: float = 0.75,
    onecycle_pct_start: float = ONECYCLE_PCT_START,
):
    """Return lr multiplier lf(epoch) used by Ultralytics (actual lr = initial_lr × lf)."""
    lrf = max(1e-7, float(lrf))
    epochs = max(1, int(epochs))

    if schedule == "linear":
        inner = lambda x: max(1 - x / epochs, 0) * (1.0 - lrf) + lrf
    elif schedule == "cosine":
        from ultralytics.utils.torch_utils import one_cycle

        inner = one_cycle(1, lrf, epochs)
    elif schedule == "onecycle":
        inner = _onecycle_epoch_lf(lrf, epochs, pct_start=onecycle_pct_start)
    elif schedule == "truncated_cosine":
        inner = _truncated_cosine_lf(lrf, epochs, cos_tmax_frac)
    else:
        inner = lambda x: max(1 - x / epochs, 0) * (1.0 - lrf) + lrf

    def lf(epoch) -> float:
        try:
            v = float(inner(epoch))
        except Exception:
            v = lrf
        if not math.isfinite(v):
            v = lrf
        return max(lrf, min(1.0, v))

    return lf


def _apply_lf_to_trainer(trainer, schedule: str, lrf: float, cos_tmax_frac: float, onecycle_pct_start: float = ONECYCLE_PCT_START) -> None:
    import torch.optim as optim

    lf = build_lf(schedule, lrf, trainer.epochs, cos_tmax_frac, onecycle_pct_start)
    trainer.lf = lf
    trainer.scheduler = optim.lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=trainer.lf)
    trainer.scheduler.last_epoch = trainer.start_epoch - 1


def install_lr_schedule_hooks(
    model,
    schedule: str,
    lrf: float,
    cos_tmax_frac: float = 0.75,
    onecycle_pct_start: float = ONECYCLE_PCT_START,
    log: Optional[LogFn] = None,
) -> None:
    """Patch trainer LR setup for onecycle / truncated_cosine (survives pipeline rebuild)."""
    schedule = (schedule or "linear").lower()
    if schedule not in ("onecycle", "truncated_cosine"):
        return

    lrf = float(lrf)
    cos_tmax_frac = float(cos_tmax_frac)
    onecycle_pct_start = float(onecycle_pct_start)

    def _on_pretrain_routine_end(trainer) -> None:
        def _custom_setup_scheduler() -> None:
            _apply_lf_to_trainer(trainer, schedule, lrf, cos_tmax_frac, onecycle_pct_start)

        trainer._setup_scheduler = _custom_setup_scheduler
        _custom_setup_scheduler()
        if log:
            extra = ""
            if schedule == "onecycle":
                extra = f", pct_start={onecycle_pct_start:.0%}, final_lr=max_lr/{ONECYCLE_FINAL_DIV_FACTOR:.0f}"
            elif schedule == "truncated_cosine":
                extra = f", 余弦阶段={cos_tmax_frac:.0%} epochs"
            log(
                f"LR schedule: {SCHEDULE_LABELS.get(schedule, schedule)} "
                f"(lrf={lrf:g}, epochs={trainer.epochs}{extra})"
            )

    model.add_callback("on_pretrain_routine_end", _on_pretrain_routine_end)


def resolve_cos_lr(schedule: str) -> bool:
    """Map GUI schedule name to Ultralytics cos_lr flag."""
    return (schedule or "linear").lower() == "cosine"
