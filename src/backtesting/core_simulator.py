# src/backtesting/core_simulator.py
"""
Ce module définit BacktestRunner, le moteur principal de simulation de backtest.
Il prend en entrée des données OHLCV avec tous les indicateurs nécessaires déjà
calculés et une instance de stratégie configurée, puis simule l'exécution des trades.
"""
import logging
import uuid # Pour les identifiants de cycle de trade uniques pour le log OOS
import time # Pour time.perf_counter()
import weakref # Potentiellement pour event_dispatcher
import concurrent.futures # Pourrait être utilisé pour des tâches parallèles non séquentielles

from datetime import timezone # Pour s'assurer que les timestamps sont UTC
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING, Union

import pandas as pd
import numpy as np
import numba # Pour @jit

if TYPE_CHECKING:
    from src.strategies.base import IStrategy # Utiliser l'interface
    from src.core.interfaces import IDataValidator, ICacheManager, IEventDispatcher # Nouvelles interfaces
    from decimal import Decimal # Pour les types dans les placeholders

# Utilisation des simulateurs de slippage et de frais des modules existants.
try:
    from src.utils.simulation_elements.slippage_simulator import SlippageSimulator
    from src.utils.simulation_elements.fees_simulator import FeeSimulator
    from src.utils.exchange_utils import get_precision_from_filter, get_filter_value, adjust_precision, adjust_quantity_to_step_size
    # Importer les nouvelles interfaces si elles sont déjà créées
    from src.core.interfaces import IDataValidator, ICacheManager, IEventDispatcher
except ImportError as e:
    logging.basicConfig(level=logging.ERROR) # Fallback logging
    logging.getLogger(__name__).critical(
        f"BacktestRunner: Erreur d'importation critique: {e}. "
        "Assurez-vous que les modules/interfaces existent et sont corrects."
    )
    # Définition de placeholders pour permettre au reste du module de se charger
    class SlippageSimulator: # type: ignore
        def __init__(self, **kwargs): pass
        def simulate_slippage(self, price: float, direction: int, **kwargs) -> float: return price
    class FeeSimulator: # type: ignore
        def __init__(self, **kwargs): pass
        def calculate_fee(self, trade_value_quote: float) -> float: return 0.0
    
    def get_precision_from_filter(symbol_info: Dict[str, Any], filter_type: str, key: str) -> Optional[int]: return 8
    def get_filter_value(symbol_info: Dict[str, Any], filter_type: str, key: str) -> Optional[float]: return 0.0
    def adjust_precision(value: Union[float, str, 'Decimal'], precision: Optional[int], tick_size: Optional[Union[float, str, 'Decimal']] = None, rounding_mode_str: str = "ROUND_HALF_UP") -> Optional[float]:
        if value is None: return None
        try: return round(float(str(value)), precision or 8)
        except: return None
    def adjust_quantity_to_step_size(quantity: Union[float, str, 'Decimal'], symbol_info: Dict[str, Any], qty_precision: Optional[int] = None, rounding_mode_str: str = "ROUND_FLOOR") -> Optional[float]:
        try: return round(float(str(quantity)), qty_precision or 8)
        except: return None
    
    # Placeholders pour les interfaces si non importables
    class IDataValidator: pass # type: ignore
    class ICacheManager: pass # type: ignore
    class IEventDispatcher: pass # type: ignore

    raise

logger = logging.getLogger(__name__)
DEFAULT_BATCH_SIZE = 100 # Taille de batch pour le traitement des signaux

class BacktestRunner:
    """
    Simule des stratégies de trading sur des données historiques.
    Gère l'exécution des ordres, la gestion de position, le calcul du PnL,
    et le suivi des métriques de performance.
    Refactorisé pour utiliser des composants injectés et des optimisations.
    """
    def __init__(self,
                 df_ohlcv_with_indicators: pd.DataFrame,
                 strategy_instance: 'IStrategy', # Utiliser l'interface IStrategy
                 initial_equity: float,
                 leverage: int,
                 symbol: str,
                 pair_config: Dict[str, Any],
                 trading_fee_bps: float,
                 slippage_config_dict: Dict[str, Any],
                 is_futures: bool,
                 run_id: str,
                 # Nouveaux paramètres injectés
                 data_validator: 'IDataValidator',
                 cache_manager: 'ICacheManager',
                 event_dispatcher: 'IEventDispatcher',
                 is_oos_simulation: bool = False,
                 verbosity: int = 1):

        self.log_prefix = f"[BacktestRunnerV2][{symbol.upper()}][Run:{run_id}]"
        self._log(f"Initialisation V2. Equity: {initial_equity:.2f}, Levier: {leverage}x, Verbosity: {verbosity}", level=1)

        self.strategy = strategy_instance
        self.initial_equity = float(initial_equity)
        self.current_equity = float(initial_equity)
        self.leverage = int(leverage)
        self.symbol = symbol.upper()
        self.pair_config = pair_config
        self.is_futures = is_futures
        self.run_id = run_id
        self.is_oos_simulation = is_oos_simulation
        self.verbosity = verbosity

        # Injection des dépendances
        self.data_validator = data_validator
        self.cache_manager = cache_manager
        self.event_dispatcher = event_dispatcher # Conserver une référence forte

        # Valider les données en entrée en utilisant le DataValidator injecté
        # Le DataFrame passé ici doit déjà contenir tous les indicateurs nécessaires
        # calculés par la stratégie ou un IndicatorCalculator en amont.
        # La validation ici se concentre sur la structure OHLCV de base et l'index.
        self._log(f"Validation des données OHLCV initiales via DataValidator. Shape: {df_ohlcv_with_indicators.shape}", level=2)
        df_validated_internal, validation_report = self.data_validator.validate_ohlcv_data(
            df_ohlcv_with_indicators.copy(), # Valider sur une copie
            required_columns=['open', 'high', 'low', 'close', 'volume'], # Colonnes de base
            symbol=self.symbol
        )
        if not validation_report.is_valid:
            errors_str = "; ".join(validation_report.errors)
            msg = f"Validation initiale des données OHLCV échouée: {errors_str}"
            logger.critical(f"{self.log_prefix} {msg}")
            raise ValueError(msg)
        if validation_report.warnings:
             self._log(f"Avertissements de validation des données initiales: {'; '.join(validation_report.warnings)}", level=1, is_error=True)

        self.df_ohlcv = df_validated_internal # Utiliser le DataFrame validé (et potentiellement nettoyé par le validateur)
        self._log(f"Données OHLCV validées. Shape final: {self.df_ohlcv.shape}", level=2)


        self.slippage_simulator = SlippageSimulator(**slippage_config_dict)
        self.fee_simulator = FeeSimulator(fee_bps=trading_fee_bps)

        self.base_asset: str = self.pair_config.get('baseAsset', '')
        self.quote_asset: str = self.pair_config.get('quoteAsset', '')
        self.price_precision: Optional[int] = get_precision_from_filter(self.pair_config, 'PRICE_FILTER', 'tickSize')
        self.quantity_precision: Optional[int] = get_precision_from_filter(self.pair_config, 'LOT_SIZE', 'stepSize')
        self.min_notional_filter: float = get_filter_value(self.pair_config, 'MIN_NOTIONAL', 'minNotional') or \
                                         get_filter_value(self.pair_config, 'NOTIONAL', 'minNotional') or 0.0
        self.min_quantity_filter: float = get_filter_value(self.pair_config, 'LOT_SIZE', 'minQty') or 0.0
        self.price_tick_size: Optional[float] = get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize')
        self.quantity_step_size: Optional[float] = get_filter_value(self.pair_config, 'LOT_SIZE', 'stepSize')

        self._reset_position_state()
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        if not self.df_ohlcv.empty:
             self.equity_curve.append({'timestamp': self.df_ohlcv.index.min(), 'equity': self.initial_equity, 'type': 'initial'})
        else:
             self.equity_curve.append({'timestamp': pd.Timestamp.now(tz='UTC'), 'equity': self.initial_equity, 'type': 'initial'})

        self.daily_equity: Dict[Any, float] = {}
        self.oos_detailed_trades_log: List[Dict[str, Any]] = []
        
        self._log(f"BacktestRunner V2 initialisé pour la stratégie '{self.strategy.strategy_name}'.", level=1)

    def _log(self, message: str, level: int = 1, is_error: bool = False) -> None:
        """Gère le logging basé sur le niveau de verbosité."""
        if self.verbosity >= level:
            if is_error:
                logger.error(f"{self.log_prefix} {message}")
            else:
                if level <= 1: logger.info(f"{self.log_prefix} {message}")
                else: logger.debug(f"{self.log_prefix} {message}")

    def _reset_position_state(self) -> None:
        """Réinitialise l'état de la position de trading."""
        self.position_open: bool = False
        self.position_size_base: float = 0.0
        self.position_size_quote: float = 0.0
        self.entry_price: float = 0.0
        self.entry_time: Optional[pd.Timestamp] = None
        self.current_sl_price: Optional[float] = None
        self.current_tp_price: Optional[float] = None
        self.trade_direction: int = 0
        self.current_trade_entry_fee: float = 0.0
        self.current_trade_cycle_id: Optional[str] = None
        self.current_entry_order_params_theoretical: Optional[Dict[str, Any]] = None
        self.current_oco_params_theoretical: Optional[Dict[str, Any]] = None
        self._log("État de la position réinitialisé.", level=2)

    def _calculate_max_position_size_base(self, entry_price_estimate: float, capital_for_trade: float) -> float:
        """Calcule la taille maximale de position en actif de base."""
        if entry_price_estimate <= 1e-9:
            self._log(f"Prix d'entrée estimé ({entry_price_estimate}) trop bas. Taille max pos = 0.", level=2, is_error=True)
            return 0.0
        max_notional_value_quote = capital_for_trade * self.leverage
        max_size_base_raw = max_notional_value_quote / entry_price_estimate
        self._log(f"CalcMaxPosSize: CapTrade={capital_for_trade:.2f}, Lev={self.leverage}, PxEst={entry_price_estimate:.{self.price_precision or 8}f}, MaxNotionnel={max_notional_value_quote:.2f}, TailleMaxBaseBrute={max_size_base_raw:.8f}", level=2)
        return max_size_base_raw

    def _apply_exchange_filters(self, quantity_base: float, price: float) -> float:
        """Applique les filtres de l'exchange (minQty, minNotional, stepSize) à une quantité."""
        # Utilise adjust_quantity_to_step_size pour arrondir au stepSize
        qty_adjusted_to_step_val = adjust_quantity_to_step_size(
            quantity_base,
            self.pair_config,
            qty_precision=self.quantity_precision, # La précision est pour l'affichage, stepSize pour l'ajustement
            rounding_mode_str="ROUND_FLOOR" # Typiquement, on arrondit vers le bas pour les quantités
        )
        if qty_adjusted_to_step_val is None:
            self._log(f"Erreur lors de l'ajustement de la quantité {quantity_base:.8f} au stepSize. Considérée comme nulle.", level=2, is_error=True)
            return 0.0
        
        qty_adjusted_to_step = qty_adjusted_to_step_val

        if qty_adjusted_to_step <= 1e-9:
             self._log(f"Quantité {quantity_base:.8f} -> {qty_adjusted_to_step:.{self.quantity_precision or 8}f} (après stepSize). Nulle.", level=2)
             return 0.0

        if self.min_quantity_filter > 0 and qty_adjusted_to_step < self.min_quantity_filter:
            self._log(f"Filtre Échec: Qty adj {qty_adjusted_to_step:.{self.quantity_precision or 8}f} < minQty {self.min_quantity_filter:.{self.quantity_precision or 8}f}.", level=2)
            return 0.0

        notional_value = qty_adjusted_to_step * price
        if self.min_notional_filter > 0 and notional_value < self.min_notional_filter:
            self._log(f"Filtre Échec: Notionnel {notional_value:.2f} < minNotional {self.min_notional_filter:.2f}.", level=2)
            return 0.0
        
        self._log(f"Filtres OK: Qty brute: {quantity_base:.8f} -> Qty finale: {qty_adjusted_to_step:.{self.quantity_precision or 8}f}. Notionnel: {notional_value:.2f}", level=2)
        return qty_adjusted_to_step

    # @numba.jit(nopython=True) # Difficile à appliquer directement à cause des appels de classe et Pandas
    # Une fonction interne purement numérique pourrait être Numba-fiée si elle est identifiée comme un goulot.
    def _simulate_order_execution(self,
                                  timestamp: pd.Timestamp,
                                  current_bar_ohlc: pd.Series, # Vue de la ligne du DataFrame principal
                                  order_type_sim: str,
                                  signal_direction: int,
                                  limit_price_target: Optional[float] = None,
                                  stop_price_trigger: Optional[float] = None,
                                  target_quantity_base_for_entry: Optional[float] = None
                                 ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Simule l'exécution d'un ordre pour la barre actuelle."""
        exec_log_prefix = f"{self.log_prefix}[ExecSim][{timestamp.strftime('%Y-%m-%d %H:%M')}]"
        
        # Utiliser .loc pour accéder aux valeurs de la Series pour éviter SettingWithCopyWarning
        # si current_bar_ohlc est une vue, bien que nous ne modifions pas ici.
        open_p = current_bar_ohlc.loc['open']
        high_p = current_bar_ohlc.loc['high']
        low_p = current_bar_ohlc.loc['low']
        # close_p = current_bar_ohlc.loc['close'] # Non utilisé directement pour l'exécution de marché/limite ici

        base_execution_price: Optional[float] = None
        is_entry_order = "ENTRY" in order_type_sim.upper()

        if order_type_sim in ('MARKET_ENTRY', 'MARKET_EXIT'):
            base_execution_price = open_p # Exécution au prix d'ouverture de la barre suivante
        elif order_type_sim == 'LIMIT_ENTRY' and limit_price_target is not None:
            if signal_direction == 1:  # Achat LIMIT
                if low_p <= limit_price_target: # Si le plus bas de la barre atteint le prix limite
                    base_execution_price = min(open_p, limit_price_target) # Exécuté au meilleur de open ou limite
            elif signal_direction == -1: # Vente LIMIT (short)
                if high_p >= limit_price_target:
                    base_execution_price = max(open_p, limit_price_target)
        elif order_type_sim == 'TP_EXIT' and limit_price_target is not None: # Take Profit (ordre LIMIT)
            if self.trade_direction == 1 and high_p >= limit_price_target: # TP sur un LONG
                base_execution_price = max(open_p, limit_price_target) # Exécuté au TP ou mieux (open)
            elif self.trade_direction == -1 and low_p <= limit_price_target: # TP sur un SHORT
                base_execution_price = min(open_p, limit_price_target)
        elif order_type_sim == 'SL_EXIT' and stop_price_trigger is not None: # Stop Loss (ordre STOP MARKET)
            if self.trade_direction == 1 and low_p <= stop_price_trigger: # SL sur un LONG
                base_execution_price = min(open_p, stop_price_trigger) # Exécuté au SL ou pire (open)
            elif self.trade_direction == -1 and high_p >= stop_price_trigger: # SL sur un SHORT
                base_execution_price = max(open_p, stop_price_trigger)
        
        if base_execution_price is None:
            self._log(f"{exec_log_prefix}[{order_type_sim}] Non exécutable. LmtCible:{limit_price_target}, StpTrig:{stop_price_trigger}", level=3)
            return None, None, None

        # Déterminer la direction de l'impact du slippage
        slippage_impact_direction = 0 # 1 pour achat (prix augmente), -1 pour vente (prix baisse)
        if is_entry_order:
            slippage_impact_direction = 1 if signal_direction == 1 else -1
        else: # Ordre de sortie
            slippage_impact_direction = -1 if self.trade_direction == 1 else 1 # Inverse de la direction de la position

        slipped_price = self.slippage_simulator.simulate_slippage(
            price=base_execution_price,
            direction=slippage_impact_direction,
            # Fournir des données de contexte optionnelles si disponibles et utiles
            order_book_depth_at_price=current_bar_ohlc.get('volume', 0), # Volume comme proxy de profondeur
            market_volatility_pct=(high_p - low_p) / low_p if low_p > 1e-9 else 0.0
        )
        
        # Ajuster le prix slippé à la précision de l'exchange
        final_executed_price = adjust_precision(
            value=slipped_price,
            precision=self.price_precision,
            tick_size=self.price_tick_size, # Utiliser tick_size pour l'ajustement
            rounding_mode_str="ROUND_HALF_UP" # Ou un autre mode si pertinent
        )

        if final_executed_price is None:
            logger.error(f"{exec_log_prefix}[{order_type_sim}] Erreur d'ajustement de précision pour prix slippé {slipped_price}. Prix base: {base_execution_price}")
            return None, None, None
        
        executed_quantity_base: float
        if is_entry_order:
            if target_quantity_base_for_entry is None or target_quantity_base_for_entry <= 1e-9:
                self._log(f"{exec_log_prefix}[{order_type_sim}] Quantité cible d'entrée invalide ({target_quantity_base_for_entry}). Pas d'exécution.", level=1, is_error=True)
                return None, None, None
            executed_quantity_base = self._apply_exchange_filters(target_quantity_base_for_entry, final_executed_price)
            if executed_quantity_base <= 1e-9:
                self._log(f"{exec_log_prefix}[{order_type_sim}] Quantité d'entrée {target_quantity_base_for_entry:.8f} nulle/invalide après filtres. Pas d'exécution.", level=1)
                return None, None, None
        else: # Ordre de sortie, utiliser la taille de la position actuelle
            if not self.position_open or self.position_size_base <= 1e-9:
                self._log(f"{exec_log_prefix}[{order_type_sim}] Tentative de sortie mais pas de position ouverte valide (Taille: {self.position_size_base}).", level=2)
                return None, None, None
            executed_quantity_base = self.position_size_base # Sortir de toute la position

        trade_value_quote = executed_quantity_base * final_executed_price
        fee_paid_quote = self.fee_simulator.calculate_fee(trade_value_quote)

        self._log(f"{exec_log_prefix}[{order_type_sim}] Exé: PxBase={base_execution_price:.{self.price_precision or 8}f}, "
                  f"PxSlip={slipped_price:.{self.price_precision or 8}f}, PxFinal={final_executed_price:.{self.price_precision or 8}f}, "
                  f"QtyBase={executed_quantity_base:.{self.quantity_precision or 8}f}, ValQt={trade_value_quote:.2f}, FraisQt={fee_paid_quote:.4f}", level=2)

        trade_details_for_event: Optional[Dict[str, Any]] = None

        if is_entry_order:
            self.position_open = True
            self.trade_direction = signal_direction
            self.entry_price = final_executed_price
            self.entry_time = timestamp
            self.position_size_base = executed_quantity_base
            self.position_size_quote = trade_value_quote
            self.current_trade_entry_fee = fee_paid_quote
            
            self.current_equity -= fee_paid_quote # Déduire les frais d'entrée
            self.equity_curve.append({'timestamp': timestamp, 'equity': self.current_equity, 'type': 'entry_fee'})
            self._log(f"ENTRÉE {'LONG' if self.trade_direction == 1 else 'SHORT'} @ {self.entry_price:.{self.price_precision or 8}f}, "
                      f"Qty: {self.position_size_base:.{self.quantity_precision or 8}f}. Frais: {fee_paid_quote:.4f}. Equity après frais: {self.current_equity:.2f}", level=1)
            
            if self.is_oos_simulation: # Pour le log OOS détaillé
                self.current_trade_cycle_id = str(uuid.uuid4()) # Nouveau cycle de trade
            
            trade_details_for_event = {
                "trade_cycle_id": self.current_trade_cycle_id, "timestamp": timestamp.isoformat(), "type": "ENTRY",
                "direction": "LONG" if self.trade_direction == 1 else "SHORT", "price": final_executed_price,
                "quantity_base": executed_quantity_base, "fee_quote": fee_paid_quote,
                "equity_after_trade": self.current_equity,
                "entry_order_params_theoretical": self.current_entry_order_params_theoretical, # Ajouté pour OOS log
                "oco_params_theoretical": self.current_oco_params_theoretical # Ajouté pour OOS log
            }

        else: # Ordre de sortie
            exit_price = final_executed_price
            pnl_gross_quote = (exit_price - self.entry_price) * self.position_size_base \
                if self.trade_direction == 1 else (self.entry_price - exit_price) * self.position_size_base
            
            pnl_net_quote = pnl_gross_quote - self.current_trade_entry_fee - fee_paid_quote
            
            self.current_equity += pnl_net_quote # PnL net est déjà après tous les frais
            self.equity_curve.append({'timestamp': timestamp, 'equity': self.current_equity, 'type': 'trade_pnl'})
            
            trade_log_entry = {
                'entry_time': self.entry_time, 'exit_time': timestamp,
                'entry_price': self.entry_price, 'exit_price': exit_price,
                'size_base': self.position_size_base, 'size_quote_at_entry': self.position_size_quote,
                'pnl_gross_quote': pnl_gross_quote, 'pnl_net_quote': pnl_net_quote,
                'entry_fee_quote': self.current_trade_entry_fee, 'exit_fee_quote': fee_paid_quote,
                'total_fees_quote': self.current_trade_entry_fee + fee_paid_quote,
                'direction': 'long' if self.trade_direction == 1 else 'short',
                'leverage': self.leverage, 'exit_reason': order_type_sim
            }
            self.trades.append(trade_log_entry)
            self._log(f"SORTIE {'LONG' if self.trade_direction == 1 else 'SHORT'} @ {exit_price:.{self.price_precision or 8}f} (Raison: {order_type_sim}). "
                      f"PnL Net: {pnl_net_quote:.2f}. Frais Sortie: {fee_paid_quote:.4f}. Equity après trade: {self.current_equity:.2f}", level=1)

            trade_details_for_event = {
                "trade_cycle_id": self.current_trade_cycle_id, "timestamp": timestamp.isoformat(), "type": "EXIT",
                "direction": "LONG" if self.trade_direction == 1 else "SHORT", "price": final_executed_price,
                "quantity_base": self.position_size_base, "fee_quote": fee_paid_quote,
                "pnl_net_quote": pnl_net_quote, "equity_after_trade": self.current_equity,
                "exit_reason": order_type_sim
            }
            if self.is_oos_simulation:
                self._record_oos_detailed_trade(trade_log_entry) # Enregistrer le trade complété pour OOS
            
            self._reset_position_state() # Réinitialiser après une sortie

        if trade_details_for_event:
            self.emit_event('trade_executed', trade_details_for_event)

        return final_executed_price, executed_quantity_base, fee_paid_quote

    def _record_oos_detailed_trade(self, trade_info_completed: Dict[str, Any]) -> None:
        """Enregistre un trade complété dans le log OOS détaillé."""
        if not self.current_trade_cycle_id:
            self.current_trade_cycle_id = str(uuid.uuid4()) # Fallback si ID de cycle manquant
            self._log(f"ID de cycle OOS manquant pour enregistrement détaillé, nouveau généré : {self.current_trade_cycle_id}", level=2)

        entry_params_theo = self.current_entry_order_params_theoretical or {}
        oco_params_theo = self.current_oco_params_theoretical or {}
        
        intent_type = "UNKNOWN_INTENT"
        if entry_params_theo.get("type"):
            intent_type = f"{str(entry_params_theo['type']).upper()}_ENTRY_{trade_info_completed['direction'].upper()}"
            if oco_params_theo:
                intent_type = f"OTOCO_VIA_{intent_type}"

        log_entry_oos = {
            "trade_cycle_id": self.current_trade_cycle_id, "run_id": self.run_id,
            "strategy_name": self.strategy.strategy_name, "pair_symbol": self.symbol,
            "is_oos_simulation": True,
            "entry_timestamp_simulated_utc": trade_info_completed['entry_time'].isoformat() if pd.notna(trade_info_completed.get('entry_time')) else None,
            "order_intent_type_simulated": intent_type,
            "entry_order_theoretical_params": entry_params_theo,
            "oco_order_theoretical_params": oco_params_theo,
            "entry_execution_price_slipped": trade_info_completed['entry_price'],
            "entry_executed_quantity_base": trade_info_completed['size_base'],
            "entry_fee_quote": trade_info_completed['entry_fee_quote'],
            "exit_timestamp_simulated_utc": trade_info_completed['exit_time'].isoformat() if pd.notna(trade_info_completed.get('exit_time')) else None,
            "exit_reason_simulated": trade_info_completed['exit_reason'],
            "exit_execution_price_slipped": trade_info_completed['exit_price'],
            "exit_fee_quote": trade_info_completed['exit_fee_quote'],
            "pnl_gross_quote_simulated": trade_info_completed['pnl_gross_quote'],
            "pnl_net_quote_simulated": trade_info_completed['pnl_net_quote'],
            "total_fees_quote_simulated": trade_info_completed['total_fees_quote']
        }
        self.oos_detailed_trades_log.append(log_entry_oos)
        self._log(f"Log OOS détaillé enregistré pour cycle ID {self.current_trade_cycle_id}", level=2)

    def emit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Émet un événement via le dispatcher d'événements injecté."""
        if hasattr(self, 'event_dispatcher') and self.event_dispatcher:
            try:
                # Utiliser weakref si le dispatcher est conçu pour cela, sinon appel direct
                if isinstance(self.event_dispatcher, weakref.ReferenceType):
                    dispatcher_instance = self.event_dispatcher()
                    if dispatcher_instance:
                        dispatcher_instance.dispatch(event_type, event_data) # type: ignore
                else: # Appel direct
                    self.event_dispatcher.dispatch(event_type, event_data) # type: ignore
                self._log(f"Event '{event_type}' émis. Data keys: {list(event_data.keys())}", level=3)
            except Exception as e:
                self._log(f"Erreur lors de l'émission de l'événement '{event_type}': {e}", level=1, is_error=True)
        else:
            self._log(f"Event dispatcher non disponible. Événement '{event_type}' non émis.", level=2)


    def _process_signals_batch(self,
                               df_batch_slice: pd.DataFrame, # Le slice de données pour ce batch
                               full_history_df_up_to_batch_start: pd.DataFrame # L'historique complet jusqu'au début de ce batch
                              ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Traite un batch de barres de données, génère des signaux et simule des ordres.
        Cette méthode est appelée par run_simulation pour chaque batch.
        Elle maintient et met à jour l'état de la position et de l'équité.

        Args:
            df_batch_slice (pd.DataFrame): Le slice de DataFrame (OHLCV + indicateurs) pour le batch actuel.
                                           L'index est un DatetimeIndex.
            full_history_df_up_to_batch_start (pd.DataFrame): DataFrame contenant tout l'historique
                                                              jusqu'au début de `df_batch_slice`.
                                                              Utilisé pour fournir le contexte
                                                              nécessaire à strategy.get_signal().

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
                - trades_in_batch: Liste des trades exécutés dans ce batch.
                - equity_points_in_batch: Points de la courbe d'équité générés dans ce batch.
                - oos_log_entries_in_batch: Entrées de log OOS détaillées générées.
        """
        batch_log_prefix = f"{self.log_prefix}[BatchProc][{df_batch_slice.index.min().strftime('%Y%m%d-%H%M') if not df_batch_slice.empty else 'empty_batch'}]"
        self._log(f"{batch_log_prefix} Démarrage traitement du batch de {len(df_batch_slice)} barres.", level=2)
        
        batch_trades: List[Dict[str, Any]] = []
        batch_equity_points: List[Dict[str, Any]] = []
        batch_oos_log_entries: List[Dict[str, Any]] = []

        # Concaténer l'historique précédent avec le batch actuel pour la stratégie
        # On ne fait la copie qu'une fois par batch pour l'historique.
        # df_feed_for_strategy sera étendu itérativement à l'intérieur de la boucle du batch.
        # Initialiser avec l'historique avant le début de ce batch.
        df_feed_for_strategy_cumulative = full_history_df_up_to_batch_start.copy()


        for i in range(len(df_batch_slice)):
            current_timestamp_in_batch: pd.Timestamp = df_batch_slice.index[i] # type: ignore
            current_row_data_in_batch: pd.Series = df_batch_slice.iloc[i]

            # Ajouter la barre actuelle du batch au DataFrame cumulatif pour la stratégie
            # Utiliser .loc pour ajouter la ligne afin d'éviter des problèmes d'alignement d'index si non trié (bien que ça devrait l'être)
            # S'assurer que current_row_data_in_batch est une ligne avec le bon index (current_timestamp_in_batch)
            # df_feed_for_strategy_cumulative.loc[current_timestamp_in_batch] = current_row_data_in_batch # Peut être lent
            # Alternative plus sûre si l'index est garanti unique et trié :
            current_row_as_df = pd.DataFrame([current_row_data_in_batch], index=[current_timestamp_in_batch])
            df_feed_for_strategy_cumulative = pd.concat([df_feed_for_strategy_cumulative, current_row_as_df])
            # S'assurer de l'unicité de l'index après concat, au cas où
            if df_feed_for_strategy_cumulative.index.has_duplicates:
                 df_feed_for_strategy_cumulative = df_feed_for_strategy_cumulative[~df_feed_for_strategy_cumulative.index.duplicated(keep='last')]


            # Enregistrer l'équité journalière
            current_date_normalized = current_timestamp_in_batch.normalize()
            if not self.daily_equity or current_date_normalized not in self.daily_equity or \
               (i == len(df_batch_slice) - 1 and self.daily_equity.get(current_date_normalized) != self.current_equity) : # Toujours enregistrer la dernière du batch
                self.daily_equity[current_date_normalized] = self.current_equity
                # Pas besoin d'ajouter à batch_equity_points ici, c'est pour les changements dus aux trades/frais

            exit_executed_this_bar = False
            if self.position_open:
                # Vérifier SL
                if self.current_sl_price is not None:
                    exec_p_sl, _, fee_sl = self._simulate_order_execution(
                        current_timestamp_in_batch, current_row_data_in_batch, 'SL_EXIT',
                        self.trade_direction, stop_price_trigger=self.current_sl_price
                    )
                    if exec_p_sl is not None:
                        exit_executed_this_bar = True
                        self._log(f"SL ATTEINT & EXÉCUTÉ @ {exec_p_sl:.{self.price_precision or 8}f}", level=1)
                        # Les trades et equity_points sont déjà ajoutés par _simulate_order_execution
                        # On a besoin de les collecter pour ce batch
                        if self.trades and self.trades[-1]['exit_time'] == current_timestamp_in_batch: batch_trades.append(self.trades[-1])
                        if self.equity_curve and self.equity_curve[-1]['timestamp'] == current_timestamp_in_batch: batch_equity_points.append(self.equity_curve[-1])
                        if self.is_oos_simulation and self.oos_detailed_trades_log and self.oos_detailed_trades_log[-1]['exit_timestamp_simulated_utc'] == current_timestamp_in_batch.isoformat():
                            batch_oos_log_entries.append(self.oos_detailed_trades_log[-1])


                # Vérifier TP (seulement si SL n'a pas été exécuté)
                if not exit_executed_this_bar and self.current_tp_price is not None:
                    exec_p_tp, _, fee_tp = self._simulate_order_execution(
                        current_timestamp_in_batch, current_row_data_in_batch, 'TP_EXIT',
                        self.trade_direction, limit_price_target=self.current_tp_price
                    )
                    if exec_p_tp is not None:
                        exit_executed_this_bar = True
                        self._log(f"TP ATTEINT & EXÉCUTÉ @ {exec_p_tp:.{self.price_precision or 8}f}", level=1)
                        if self.trades and self.trades[-1]['exit_time'] == current_timestamp_in_batch: batch_trades.append(self.trades[-1])
                        if self.equity_curve and self.equity_curve[-1]['timestamp'] == current_timestamp_in_batch: batch_equity_points.append(self.equity_curve[-1])
                        if self.is_oos_simulation and self.oos_detailed_trades_log and self.oos_detailed_trades_log[-1]['exit_timestamp_simulated_utc'] == current_timestamp_in_batch.isoformat():
                            batch_oos_log_entries.append(self.oos_detailed_trades_log[-1])
                
                if exit_executed_this_bar:
                    continue # Passer à la barre suivante du batch

            # Obtenir le signal de la stratégie
            # Utiliser une vue pour éviter des copies inutiles si df_feed_for_strategy_cumulative devient grand.
            # Cependant, la stratégie pourrait le modifier si elle ne prend pas une copie.
            # Pour la sécurité, la stratégie devrait copier si elle modifie.
            # Ici, on passe la référence.
            signal_decision = self.strategy.get_signal(
                df_feed_for_strategy_cumulative, # Passe l'historique complet jusqu'à la barre actuelle
                self.position_open,
                self.trade_direction,
                self.entry_price,
                self.current_equity
            )

            sig_type = signal_decision.get("signal", 0)
            sig_order_type = signal_decision.get("order_type", "MARKET")
            sig_limit_px = signal_decision.get("limit_price")
            sig_sl_px = signal_decision.get("sl_price")
            sig_tp_px = signal_decision.get("tp_price")
            sig_pos_size_pct = signal_decision.get("position_size_pct", 1.0)

            if self.is_oos_simulation and sig_type in [1, -1] and not self.position_open:
                self.current_entry_order_params_theoretical = signal_decision.get("entry_order_params_theoretical_for_oos_log")
                self.current_oco_params_theoretical = signal_decision.get("oco_params_theoretical_for_oos_log")


            if self.position_open: # Si une position est toujours ouverte (pas de SL/TP touché)
                # Vérifier les signaux de sortie de la stratégie
                if sig_type == 2 or \
                   (self.trade_direction == 1 and sig_type == -1) or \
                   (self.trade_direction == -1 and sig_type == 1):
                    
                    self._log(f"SIGNAL DE SORTIE STRATÉGIE reçu (Type: {sig_type}). Tentative de sortie au marché.", level=1)
                    _, _, _ = self._simulate_order_execution( # Les résultats sont déjà ajoutés aux listes globales par la méthode
                        current_timestamp_in_batch, current_row_data_in_batch,
                        'MARKET_EXIT', self.trade_direction
                    )
                    if self.trades and self.trades[-1]['exit_time'] == current_timestamp_in_batch: batch_trades.append(self.trades[-1])
                    if self.equity_curve and self.equity_curve[-1]['timestamp'] == current_timestamp_in_batch: batch_equity_points.append(self.equity_curve[-1])
                    if self.is_oos_simulation and self.oos_detailed_trades_log and self.oos_detailed_trades_log[-1]['exit_timestamp_simulated_utc'] == current_timestamp_in_batch.isoformat():
                        batch_oos_log_entries.append(self.oos_detailed_trades_log[-1])

            elif sig_type in [1, -1]: # Pas de position ouverte, et signal d'entrée
                self._log(f"SIGNAL D'ENTRÉE {'LONG' if sig_type == 1 else 'SHORT'} reçu de la stratégie.", level=1)
                
                entry_px_est_for_qty_calc = sig_limit_px if sig_order_type == 'LIMIT' and sig_limit_px is not None else float(current_row_data_in_batch['open'])
                if pd.isna(entry_px_est_for_qty_calc) or entry_px_est_for_qty_calc <= 1e-9:
                    self._log(f"Prix d'estimation pour l'entrée ({entry_px_est_for_qty_calc}) invalide. Pas d'entrée.", level=1, is_error=True)
                    continue
                
                capital_to_risk_for_trade = self.current_equity * (sig_pos_size_pct if sig_pos_size_pct is not None else 1.0)
                target_qty_base_for_entry_raw = self._calculate_max_position_size_base(entry_px_est_for_qty_calc, capital_to_risk_for_trade)
                
                if target_qty_base_for_entry_raw <= 1e-9:
                    self._log(f"Quantité max base calculée ({target_qty_base_for_entry_raw:.8f}) trop petite. Pas d'entrée.", level=1)
                    continue
                
                simulated_entry_order_type = 'LIMIT_ENTRY' if sig_order_type == 'LIMIT' else 'MARKET_ENTRY'
                exec_px_entry, exec_qty_entry, fee_entry = self._simulate_order_execution(
                    current_timestamp_in_batch, current_row_data_in_batch,
                    simulated_entry_order_type, sig_type,
                    limit_price_target=sig_limit_px if sig_order_type == 'LIMIT' else None,
                    target_quantity_base_for_entry=target_qty_base_for_entry_raw
                )
                
                if exec_px_entry is not None and exec_qty_entry is not None and exec_qty_entry > 1e-9:
                    self.current_sl_price = sig_sl_px
                    self.current_tp_price = sig_tp_px
                    self._log(f"SL/TP mis à jour après entrée: SL={self.current_sl_price}, TP={self.current_tp_price}", level=2)
                    # Les trades et equity_points sont déjà ajoutés par _simulate_order_execution
                    if self.equity_curve and self.equity_curve[-1]['timestamp'] == current_timestamp_in_batch: batch_equity_points.append(self.equity_curve[-1])
                    # Note: un trade d'ENTRÉE n'est pas ajouté à `batch_trades` car `batch_trades` est pour les trades *complétés*.
                    # L'événement 'trade_executed' pour l'entrée est émis dans _simulate_order_execution.
                    # Le log OOS détaillé pour l'entrée est fait dans _record_oos_detailed_trade, appelé aussi depuis _simulate_order_execution.
                    # Si on veut loguer l'ouverture de position dans oos_detailed_trades_log, il faudrait une logique ici ou dans _simulate_order_execution.
                    # La version actuelle de _record_oos_detailed_trade est appelée à la *clôture* du trade.
                    # Pour le log OOS, l'entrée est loguée quand le trade est *fermé*.
                else:
                    self._log(f"Ordre d'entrée non exécuté ou quantité nulle. Pas de nouvelle position.", level=1)
                    if self.is_oos_simulation and self.current_trade_cycle_id: # Si un cycle avait été initié
                        self._reset_position_state() # Réinitialiser si l'entrée a échoué


        self._log(f"{batch_log_prefix} Traitement du batch terminé. {len(batch_trades)} trades complétés dans ce batch.", level=2)
        return batch_trades, batch_equity_points, batch_oos_log_entries


    def run_simulation(self) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[Any, float], List[Dict[str, Any]]]:
        """Exécute la simulation de backtest complète, traitant les données par batches."""
        overall_start_time = time.perf_counter()
        self._log(f"Démarrage de la simulation V2 pour {len(self.df_ohlcv)} barres.", level=1)

        if self.df_ohlcv.empty:
            logger.warning(f"{self.log_prefix} Données OHLCV vides. Simulation V2 sautée.")
            return self.trades, pd.DataFrame(self.equity_curve), self.daily_equity, self.oos_detailed_trades_log

        batch_size = DEFAULT_BATCH_SIZE
        num_bars = len(self.df_ohlcv)
        
        # Initialiser l'historique cumulatif avant la première barre du premier batch
        # C'est un DataFrame vide car il n'y a pas d'historique avant le tout début.
        cumulative_history_for_strategy = pd.DataFrame(columns=self.df_ohlcv.columns, index=pd.DatetimeIndex([], tz='UTC'))


        for i in range(0, num_bars, batch_size):
            batch_start_time = time.perf_counter()
            start_index = i
            end_index = min(i + batch_size, num_bars)
            
            # Le slice pour le batch actuel
            current_batch_data_slice = self.df_ohlcv.iloc[start_index:end_index]
            if current_batch_data_slice.empty:
                continue

            batch_log_id = f"Batch_{start_index//batch_size +1}"
            self._log(f"Traitement du {batch_log_id} (barres {start_index}-{end_index-1})", level=2)

            # _process_signals_batch met à jour self.trades, self.equity_curve, self.oos_detailed_trades_log,
            # self.daily_equity, self.current_equity, et l'état de la position interne.
            # Il a besoin de l'historique cumulatif *avant* le début de ce batch pour la stratégie.
            _, _, _ = self._process_signals_batch(
                df_batch_slice=current_batch_data_slice,
                full_history_df_up_to_batch_start=cumulative_history_for_strategy # Passe l'historique jusqu'au début de ce batch
            )
            
            # Mettre à jour l'historique cumulatif pour le prochain batch
            # en ajoutant les données du batch qui vient d'être traité.
            cumulative_history_for_strategy = pd.concat([cumulative_history_for_strategy, current_batch_data_slice])
            if cumulative_history_for_strategy.index.has_duplicates:
                 cumulative_history_for_strategy = cumulative_history_for_strategy[~cumulative_history_for_strategy.index.duplicated(keep='last')]


            batch_end_time = time.perf_counter()
            self._log(f"{batch_log_id} traité en {batch_end_time - batch_start_time:.3f}s.", level=2)

            # Caching intermédiaire de l'equity_curve (exemple simple)
            # La clé de cache pourrait inclure le run_id et l'index du batch
            # cache_key_equity = f"equity_curve_run_{self.run_id}_batch_{start_index//batch_size +1}"
            # self.cache_manager.get_or_compute(cache_key_equity, lambda: list(self.equity_curve), ttl=3600) # Cache pour 1h
            # Note: get_or_compute n'est pas idéal pour juste "mettre" dans le cache.
            # Un `cache_manager.set(key, value, ttl)` serait plus approprié.
            # Pour l'instant, on ne met pas en cache de manière intermédiaire ici.

        # S'assurer que l'équité finale est enregistrée pour le dernier jour
        if not self.df_ohlcv.empty:
            final_date_normalized = self.df_ohlcv.index[-1].normalize() # type: ignore
            if final_date_normalized not in self.daily_equity or self.daily_equity[final_date_normalized] != self.current_equity:
                self.daily_equity[final_date_normalized] = self.current_equity
            
            # S'assurer que le dernier point de la courbe d'équité correspond à l'équité finale
            if self.equity_curve and (self.equity_curve[-1]['timestamp'] != self.df_ohlcv.index[-1] or self.equity_curve[-1]['equity'] != self.current_equity):
                 self.equity_curve.append({'timestamp': self.df_ohlcv.index[-1], 'equity': self.current_equity, 'type': 'final_sim_end'})


        overall_end_time = time.perf_counter()
        total_duration = overall_end_time - overall_start_time
        self._log(f"Simulation V2 terminée. Trades: {len(self.trades)}. Equity finale: {self.current_equity:.2f}. "
                  f"Temps total: {total_duration:.3f}s.", level=1)

        equity_curve_df_final = pd.DataFrame(self.equity_curve)
        if not equity_curve_df_final.empty and 'timestamp' in equity_curve_df_final.columns:
            equity_curve_df_final['timestamp'] = pd.to_datetime(equity_curve_df_final['timestamp'], errors='coerce', utc=True)
            equity_curve_df_final.dropna(subset=['timestamp'], inplace=True)
            # S'assurer que l'index est unique pour la conversion en Series pour les métriques
            equity_curve_df_final = equity_curve_df_final.drop_duplicates(subset=['timestamp'], keep='last').set_index('timestamp').sort_index()


        # Cacher le résultat final de l'equity_curve (la série, pas la liste de dicts)
        # La clé doit être unique pour ce run spécifique.
        final_equity_curve_cache_key = f"final_equity_series_run_{self.run_id}_{self.strategy.strategy_name}_{self.symbol}"
        
        # Pour get_or_compute, la fonction lambda ne serait appelée que si la clé n'existe pas.
        # Ici, on veut "mettre" le résultat. Si le CacheManager a une méthode `set`, ce serait mieux.
        # Sinon, on peut utiliser get_or_compute avec une lambda qui retourne simplement la valeur si on veut la stocker.
        # Pour cet exemple, nous n'allons pas utiliser get_or_compute pour stocker le résultat final ici,
        # car le but est de retourner la courbe, pas de la recalculer si elle manque.
        # Le cache serait plus utile pour des calculs intermédiaires *répétitifs*.
        # Si on voulait cacher le résultat *de cette exécution*:
        # self.cache_manager.set(final_equity_curve_cache_key, equity_curve_df_final['equity'], ttl=...)

        return self.trades, equity_curve_df_final, self.daily_equity, self.oos_detailed_trades_log

