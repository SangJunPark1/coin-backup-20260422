import unittest
from datetime import datetime, timezone

from coin_mvp.broker import PaperBroker
from coin_mvp.config import RiskConfig, StrategyConfig
from coin_mvp.models import Candle, Position, Side, Signal
from coin_mvp.report import calculate_max_consecutive_losses, calculate_max_drawdown
from coin_mvp.risk import RiskManager
from coin_mvp.strategy import MovingAverageStrategy, btc_regime_allows_entries, calculate_ema, volatility_adjusted_position_fraction


class PaperBrokerTest(unittest.TestCase):
    def test_buy_and_sell_updates_cash_and_position(self):
        broker = PaperBroker("KRW-BTC", starting_cash=1_000_000, fee_rate=0.0005, slippage_bps=0)

        buy = broker.buy(price=50_000_000, cash_to_use=200_000, reason="test buy")
        self.assertIsNotNone(buy)
        self.assertGreater(broker.position.qty, 0)
        self.assertLess(broker.cash, 1_000_000)

        sell = broker.sell_all(price=51_000_000, reason="test sell")
        self.assertIsNotNone(sell)
        self.assertEqual(broker.position.qty, 0)
        self.assertGreater(sell.realized_pnl, 0)


class RiskManagerTest(unittest.TestCase):
    def test_daily_loss_halts_trading(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=1.0,
                daily_loss_limit_pct=1.0,
                max_entries_per_day=3,
                max_position_fraction=0.25,
                max_consecutive_losses=2,
            ),
            starting_equity=1_000_000,
        )

        approved, reason = risk.approve(
            Signal(Side.BUY, "test", price=100.0),
            current_equity=989_000,
            position_fraction=0.2,
        )
        self.assertFalse(approved)
        self.assertIn("daily loss limit", reason)

    def test_sell_is_allowed_after_entry_limit(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=1.0,
                daily_loss_limit_pct=1.0,
                max_entries_per_day=1,
                max_position_fraction=0.25,
                max_consecutive_losses=2,
            ),
            starting_equity=1_000_000,
        )
        risk.state.entries_today = 1

        approved, reason = risk.approve(
            Signal(Side.SELL, "exit", price=100.0),
            current_equity=1_000_000,
            position_fraction=0.2,
        )
        self.assertTrue(approved)
        self.assertIn("risk-reducing exit", reason)

    def test_new_korea_day_resets_daily_target_base(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=3.0,
                daily_loss_limit_pct=5.0,
                max_entries_per_day=12,
                max_position_fraction=0.35,
                max_consecutive_losses=4,
            ),
            starting_equity=1_000_000,
        )
        risk.ensure_trading_day(datetime(2026, 4, 19, 0, 0, tzinfo=timezone.utc), 1_000_000)
        risk.state.entries_today = 3
        risk.state.halted = True
        risk.state.halt_reason = "daily profit target reached: 3.00%"

        risk.ensure_trading_day(datetime(2026, 4, 19, 15, 0, tzinfo=timezone.utc), 1_030_000)

        self.assertEqual(risk.state.starting_equity, 1_030_000)
        self.assertEqual(risk.state.entries_today, 0)
        self.assertFalse(risk.state.halted)


class ReportMetricsTest(unittest.TestCase):
    def test_max_drawdown_uses_cumulative_realized_pnl(self):
        self.assertEqual(calculate_max_drawdown([1000, -300, -500, 200]), -800)

    def test_max_consecutive_losses(self):
        self.assertEqual(calculate_max_consecutive_losses([1000, -1, -2, 3, -4, -5, -6]), 3)


class StrategyFilterTest(unittest.TestCase):
    def test_entry_blocks_overextended_move(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            max_recent_momentum_pct=1.0,
            min_volume_ratio=0.5,
            long_trend_ema_window=0,
        )
        candles = make_candles([100, 101, 102, 103, 104, 112], volume=10.0)

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)

    def test_btc_regime_blocks_weak_trend(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            btc_short_window=3,
            btc_long_window=5,
            min_btc_momentum_pct=-0.5,
        )
        candles = make_candles([100, 99, 98, 97, 96, 94], volume=10.0)

        allowed, reason, _ = btc_regime_allows_entries(candles, config)

        self.assertFalse(allowed)
        self.assertIn("btc regime blocked", reason)

    def test_entry_blocks_high_rsi(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            max_entry_rsi=70.0,
            min_volume_ratio=0.5,
            max_recent_momentum_pct=10.0,
            long_trend_ema_window=0,
        )
        candles = make_candles([100, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113], volume=10.0)

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)

    def test_volatility_reduces_position_fraction(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            target_recent_volatility_pct=1.0,
            min_volatility_position_fraction=0.4,
        )
        candles = make_candles([100, 105, 97, 108, 96, 110, 95, 112, 94, 115, 93, 118, 92, 120, 91, 122, 90, 124, 89, 126, 88], volume=10.0)

        fraction = volatility_adjusted_position_fraction(candles, config)

        self.assertLess(fraction, config.position_fraction)
        self.assertGreaterEqual(fraction, config.position_fraction * config.min_volatility_position_fraction)

    def test_long_ema_blocks_entries_below_trend(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_volume_ratio=0.5,
            long_trend_ema_window=8,
        )
        candles = make_candles([120, 118, 116, 114, 100, 101, 102, 103], volume=10.0)

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)

    def test_calculate_ema_returns_value_when_enough_closes(self):
        self.assertIsNotNone(calculate_ema([1, 2, 3, 4, 5], 5))
        self.assertIsNone(calculate_ema([1, 2, 3], 5))


def make_candles(closes: list[float], volume: float) -> list[Candle]:
    return [
        Candle(
            market="KRW-BTC",
            timestamp=datetime(2026, 4, 20, index, tzinfo=timezone.utc),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=volume,
        )
        for index, close in enumerate(closes)
    ]


if __name__ == "__main__":
    unittest.main()
