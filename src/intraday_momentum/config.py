"""All tunable parameters in one place, so every experiment is fully described by its config."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters of the intraday momentum strategy. Defaults = the paper's specification."""

    name: str = "paper"
    lookback_days: int = 14            # days used to estimate the average move from the open (sigma)
    vol_multiplier: float = 1.0        # VM: widens (>1) or narrows (<1) the Noise Area
    decision_every_min: int = 30       # trade only at HH:00 and HH:30
    first_decision_min: int = 30       # first decision 30 min after the open, i.e. 10:00

    # "opposite_band": hold until price crosses the opposite band (base model, paper Table 1)
    # "band_vwap":     trailing stop at max(UB, VWAP) for longs / min(LB, VWAP) for shorts (paper Table 2)
    stop: Literal["opposite_band", "band_vwap"] = "opposite_band"

    # "full": 100% of equity; "vol_target": scale exposure to a daily volatility target (paper Table 3)
    sizing: Literal["full", "vol_target"] = "full"
    vol_target_daily: float = 0.02
    vol_lookback_days: int = 14
    max_leverage: float = 4.0


@dataclass(frozen=True)
class CostConfig:
    """Per-share transaction costs, charged on every share bought or sold."""

    commission_per_share: float = 0.0035   # Interactive Brokers entry-level tier (paper)
    slippage_per_share: float = 0.001      # paper's own live estimate; half-spread of SPY is ~0.005
    min_commission_per_order: float = 0.35 # IBKR minimum per order (as in the authors' reference code)

    @property
    def per_share(self) -> float:
        return self.commission_per_share + self.slippage_per_share

    def order_cost(self, shares: int) -> float:
        """Cost of one order of `shares` shares: max(minimum, commission) + slippage."""
        return max(self.min_commission_per_order, self.commission_per_share * shares) + self.slippage_per_share * shares
