import unittest
import pandas as pd # Required for type hints in abstract methods
import numpy as np  # Required for type hints in abstract methods
from src.strategies.base import BaseStrategy, TradingContext, ValidationResult
from typing import Any, Dict, List, Tuple, Optional # For type hints

class MinimalStrategy(BaseStrategy):
    REQUIRED_PARAMS: List[str] = []

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        super().__init__(strategy_name, symbol, params)

    def validate_params(self) -> None:
        pass

    def get_required_indicator_configs(self) -> List[Dict[str, Any]]:
        return []

    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame:
        return data_feed

    def _generate_signals(self,
                          data_with_indicators: pd.DataFrame,
                          current_position_open: bool,
                          current_position_direction: int,
                          current_entry_price: float
                         ) -> Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
        return 0, "MARKET", None, None, None, 1.0 # Ensure position_size_pct is float

    def generate_order_request(self,
                               data: pd.DataFrame,
                               current_position: int,
                               available_capital: float,
                               symbol_info: Dict[str, Any]
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        return None

class TestBaseStrategyContext(unittest.TestCase):

    def test_set_trading_context_on_strategy(self):
        """Test that set_trading_context can be called and sets context."""
        strategy_params = {} 
        strategy_instance = MinimalStrategy(
            strategy_name="TestMinimalStrategy",
            symbol="BTCUSDT",
            params=strategy_params
        )

        dummy_pair_config = {
            'symbol': 'BTCUSDT',
            'baseAsset': 'BTC',
            'quoteAsset': 'USDT',
            'filters': [
                {'filterType': 'PRICE_FILTER', 'tickSize': '0.01'},
                {'filterType': 'LOT_SIZE', 'stepSize': '0.00001'}
            ]
        }

        context = TradingContext(
            pair_config=dummy_pair_config,
            is_futures=False,
            leverage=1,
            initial_equity=10000.0,
            account_type="SPOT"
        )

        validation_result = strategy_instance.set_trading_context(context)

        self.assertIsInstance(validation_result, ValidationResult)
        self.assertTrue(validation_result.is_valid, f"Context validation failed: {validation_result.messages}")
        self.assertIsNotNone(strategy_instance.trading_context)
        self.assertEqual(strategy_instance.trading_context, context)
        self.assertEqual(strategy_instance.price_precision, 2)
        self.assertEqual(strategy_instance.quantity_precision, 5)

if __name__ == '__main__':
    unittest.main()
