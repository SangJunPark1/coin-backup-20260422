from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .config import RiskConfig
from .models import Fill, Side, Signal


@dataclass
class RiskState:
    starting_equity: float
    day_key: str = ""
    entries_today: int = 0
    exits_today: int = 0
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    def __init__(self, config: RiskConfig, starting_equity: float) -> None:
        self.config = config
        self.state = RiskState(starting_equity=starting_equity)

    def ensure_trading_day(self, timestamp: datetime, current_equity: float) -> None:
        day_key = korea_day_key(timestamp)
        if self.state.day_key == "":
            self.state.day_key = day_key
            return
        if self.state.day_key == day_key:
            return

        self.state.day_key = day_key
        self.state.starting_equity = current_equity
        self.state.entries_today = 0
        self.state.exits_today = 0
        self.state.halted = False
        self.state.halt_reason = ""

    def approve(self, signal: Signal, current_equity: float, position_fraction: float) -> tuple[bool, str]:
        self._update_halt_from_equity(current_equity)
        if signal.side == Side.SELL:
            if self.state.halted:
                return True, f"approved risk-reducing exit: {self.state.halt_reason}"
            return True, "approved risk-reducing exit"
        if self.state.halted:
            return False, self.state.halt_reason
        if signal.side == Side.HOLD:
            return False, "hold signal"
        if self.state.entries_today >= self.config.max_entries_per_day:
            return False, "max daily entries reached"
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self.state.halted = True
            self.state.halt_reason = "max consecutive losses reached"
            return False, self.state.halt_reason
        if signal.side == Side.BUY and position_fraction > self.config.max_position_fraction:
            return False, "position fraction exceeds risk limit"
        return True, "approved"

    def record_fill(self, fill: Fill) -> None:
        if fill.side == Side.BUY:
            self.state.entries_today += 1
        elif fill.side == Side.SELL:
            self.state.exits_today += 1
            if fill.realized_pnl < 0:
                self.state.consecutive_losses += 1
            elif fill.realized_pnl > 0:
                self.state.consecutive_losses = 0

    def _update_halt_from_equity(self, current_equity: float) -> None:
        pnl_pct = (current_equity / self.state.starting_equity - 1.0) * 100.0
        if pnl_pct >= self.config.daily_profit_target_pct:
            self.state.halted = True
            self.state.halt_reason = f"daily profit target reached: {pnl_pct:.2f}%"
        elif pnl_pct <= -self.config.daily_loss_limit_pct:
            self.state.halted = True
            self.state.halt_reason = f"daily loss limit reached: {pnl_pct:.2f}%"


def korea_day_key(timestamp: datetime) -> str:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return (timestamp.astimezone(timezone.utc) + timedelta(hours=9)).date().isoformat()
