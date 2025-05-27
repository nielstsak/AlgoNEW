# src/strategies/base.py
"""
Ce module définit la classe de base abstraite pour toutes les stratégies de trading.
Elle fournit une interface standardisée, facilite la gestion des paramètres,
et intègre les exigences de la nouvelle architecture de backtesting et de trading live.
"""
from abc import ABC, abstractmethod
import logging
import uuid
from typing import Dict, Any, Tuple, Optional, List, Union

import pandas as pd # Assurer l'import pour Pylance
import numpy as np


# Assurez-vous que ces imports sont corrects par rapport à votre structure de projet.
try:
    from src.utils.exchange_utils import (
        adjust_precision,
        adjust_quantity_to_step_size,
        get_precision_from_filter,
        get_filter_value
    )
except ImportError:
    logging.getLogger(__name__).critical(
        "BaseStrategy: Échec de l'importation des utilitaires d'exchange. "
        "Les fonctionnalités de précision et de filtrage pourraient ne pas fonctionner."
    )
    # Définir des placeholders si les imports échouent
    def get_precision_from_filter(pair_config: Dict, filter_type: str, key: str) -> Optional[int]: return 8
    def adjust_precision(value: Union[float,str,'Decimal'], precision: Optional[int], tick_size: Optional[Union[float,str,'Decimal']] = None, rounding_mode_str: str = "ROUND_HALF_UP") -> Optional[float]: 
        if value is None: return None
        try: return round(float(str(value)), precision or 8)
        except: return None
    def adjust_quantity_to_step_size(quantity: Union[float,str,'Decimal'], symbol_info: Dict, qty_precision: Optional[int] = None, rounding_mode_str: str = "ROUND_FLOOR") -> Optional[float]: 
        try: return round(float(str(quantity)), qty_precision or 8)
        except: return None
    def get_filter_value(symbol_info: Dict, filter_type: str, key: str) -> Optional[float]: return None


logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Classe de base abstraite pour toutes les stratégies de trading.
    """
    REQUIRED_PARAMS: List[str] = []

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        """
        Initialise la stratégie de base.

        Args:
            strategy_name (str): Le nom clé de la stratégie (ex: "ma_crossover_strategy").
            symbol (str): Le symbole de la paire de trading (ex: BTCUSDT).
            params (Dict[str, Any]): Les paramètres spécifiques à cette instance de stratégie.
        """
        self.strategy_name: str = strategy_name
        self.symbol: str = symbol.upper()
        self.params: Dict[str, Any] = params
        self.log_prefix: str = f"[{self.strategy_name}][{self.symbol}]"

        # Attributs peuplés par set_backtest_context
        self.pair_config: Optional[Dict[str, Any]] = None
        self.is_futures: bool = False
        self.leverage: int = 1
        self.initial_equity: float = 0.0
        self.price_precision: Optional[int] = None
        self.quantity_precision: Optional[int] = None
        
        # Attributs pour les assets
        self.base_asset: str = ""
        self.quote_asset: str = ""
        
        # Le type de compte
        self.account_type: Optional[str] = None

        # Valider les paramètres spécifiques à la stratégie
        try:
            self._validate_params()
            logger.info(f"{self.log_prefix} Stratégie initialisée avec les paramètres : {self.params}")
        except ValueError as e_val:
            logger.error(f"{self.log_prefix} Erreur de validation des paramètres lors de l'initialisation : {e_val}")
            raise

    def get_param(self, param_name: str, default: Optional[Any] = None) -> Any:
        """
        Récupère une valeur de paramètre pour la stratégie.
        """
        return self.params.get(param_name, default)

    def set_backtest_context(self,
                             pair_config: Dict[str, Any],
                             is_futures: bool,
                             leverage: int,
                             initial_equity: float,
                             account_type: Optional[str] = "SPOT"
                            ):
        """
        Configure le contexte pour un backtest.
        """
        self.pair_config = pair_config
        self.is_futures = is_futures
        self.leverage = leverage
        self.initial_equity = initial_equity
        self.account_type = account_type

        if self.pair_config:
            self.base_asset = self.pair_config.get('baseAsset', '')
            self.quote_asset = self.pair_config.get('quoteAsset', '')
            self.price_precision = get_precision_from_filter(self.pair_config, 'PRICE_FILTER', 'tickSize')
            self.quantity_precision = get_precision_from_filter(self.pair_config, 'LOT_SIZE', 'stepSize')
            
            if self.price_precision is None:
                 logger.warning(f"{self.log_prefix} Price precision (tickSize) non trouvée dans pair_config pour PRICE_FILTER. Utilisation d'une valeur par défaut.")
            if self.quantity_precision is None:
                 logger.warning(f"{self.log_prefix} Quantity precision (stepSize) non trouvée dans pair_config pour LOT_SIZE. Utilisation d'une valeur par défaut.")
            logger.debug(f"{self.log_prefix} Contexte de backtest défini. Price Precision: {self.price_precision}, Quantity Precision: {self.quantity_precision}, Leverage: {self.leverage}")
        else:
            logger.error(f"{self.log_prefix} pair_config non fourni à set_backtest_context. Les précisions ne peuvent pas être déterminées.")

    @abstractmethod
    def _validate_params(self) -> None:
        """
        Valide les paramètres spécifiques à la stratégie (self.params).
        """
        pass

    @abstractmethod
    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame:
        """
        Assure que le data_feed contient les colonnes d'indicateurs finales.
        """
        pass

    @abstractmethod
    def _generate_signals(self,
                          data_with_indicators: pd.DataFrame,
                          current_position_open: bool,
                          current_position_direction: int, 
                          current_entry_price: float
                         ) -> Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Génère les signaux de trading.
        """
        pass

    def get_signal(self,
                   data_feed: pd.DataFrame, 
                   current_position_open: bool,
                   current_position_direction: int,
                   current_entry_price: float,
                   current_equity: float 
                  ) -> Dict[str, Any]:
        """
        Méthode principale appelée par le simulateur de backtest pour obtenir les décisions de trading.
        """
        if data_feed.empty:
            logger.warning(f"{self.log_prefix} data_feed vide reçu dans get_signal. Retour de HOLD.")
            return self._get_default_signal_response("Data feed vide.")

        try:
            data_ready_for_signals = self._calculate_indicators(data_feed.copy())
        except Exception as e_calc:
            logger.error(f"{self.log_prefix} Erreur dans _calculate_indicators : {e_calc}", exc_info=True)
            return self._get_default_signal_response(f"Erreur de préparation des indicateurs : {e_calc}")

        if data_ready_for_signals.empty:
            logger.warning(f"{self.log_prefix} DataFrame vide après _calculate_indicators. Retour de HOLD.")
            return self._get_default_signal_response("Préparation des indicateurs a résulté en données vides.")

        signal_type, order_type_pref, limit_sugg, sl_sugg, tp_sugg, pos_size_pct_sugg = \
            self._generate_signals(data_ready_for_signals,
                                   current_position_open,
                                   current_position_direction,
                                   current_entry_price)

        # Sourcery: Replace if-expression with `or`
        final_order_type = order_type_pref or self.get_param("order_type_preference", "MARKET")
        final_pos_size_pct = pos_size_pct_sugg if pos_size_pct_sugg is not None else self.get_param('capital_allocation_pct', 1.0)

        entry_order_params_log: Optional[Dict[str, Any]] = None
        oco_params_log: Optional[Dict[str, Any]] = None

        if signal_type in [1, -1] and not current_position_open:
            if data_ready_for_signals.empty:
                 logger.error(f"{self.log_prefix} data_ready_for_signals est vide avant le calcul de la quantité.")
                 return self._get_default_signal_response("Données vides pour calcul de quantité.")

            current_bar_open = data_ready_for_signals['open'].iloc[-1]
            entry_price_for_qty_calc = limit_sugg if final_order_type == 'LIMIT' and limit_sugg is not None else current_bar_open

            if pd.isna(entry_price_for_qty_calc):
                logger.error(f"{self.log_prefix} Prix d'entrée pour calcul de quantité est NaN. Open: {current_bar_open}, LimitSugg: {limit_sugg}")
                return self._get_default_signal_response("Prix d'entrée pour quantité NaN.")

            theoretical_qty = self._calculate_quantity(
                entry_price=entry_price_for_qty_calc,
                available_capital=current_equity,
                qty_precision=self.quantity_precision, 
                symbol_info=self.pair_config or {}, 
                symbol=self.symbol,
                position_size_pct=final_pos_size_pct
            )

            if theoretical_qty is not None and theoretical_qty > 0:
                qty_str = f"{theoretical_qty:.{self.quantity_precision or 8}f}"
                price_str = f"{limit_sugg:.{self.price_precision or 8}f}" if limit_sugg is not None else None
                
                entry_order_params_log = self._build_entry_params_formatted(
                    side="BUY" if signal_type == 1 else "SELL",
                    quantity_str=qty_str,
                    order_type=str(final_order_type), 
                    entry_price_str=price_str
                )
                if sl_sugg is not None and tp_sugg is not None:
                    sl_str = f"{sl_sugg:.{self.price_precision or 8}f}"
                    tp_str = f"{tp_sugg:.{self.price_precision or 8}f}"
                    oco_params_log = self._build_oco_params_formatted(
                        entry_side="BUY" if signal_type == 1 else "SELL",
                        quantity_str=qty_str,
                        sl_price_str=sl_str,
                        tp_price_str=tp_str,
                        stop_limit_price_str=sl_str 
                    )
            else:
                logger.warning(f"{self.log_prefix} Quantité théorique calculée est nulle ou None ({theoretical_qty}). Pas d'ordre d'entrée.")
                signal_type = 0

        final_sl = None
        if sl_sugg is not None and self.pair_config and self.price_precision is not None:
            tick_size_sl = get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize')
            final_sl = adjust_precision(sl_sugg, self.price_precision, tick_size=tick_size_sl)
        elif sl_sugg is not None:
            final_sl = round(sl_sugg, self.price_precision or 8)

        final_tp = None
        if tp_sugg is not None and self.pair_config and self.price_precision is not None:
            tick_size_tp = get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize')
            final_tp = adjust_precision(tp_sugg, self.price_precision, tick_size=tick_size_tp)
        elif tp_sugg is not None:
            final_tp = round(tp_sugg, self.price_precision or 8)

        final_limit = None
        if limit_sugg is not None and self.pair_config and self.price_precision is not None:
            tick_size_limit = get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize')
            final_limit = adjust_precision(limit_sugg, self.price_precision, tick_size=tick_size_limit)
        elif limit_sugg is not None:
            final_limit = round(limit_sugg, self.price_precision or 8)

        return {
            "signal": signal_type,
            "order_type": str(final_order_type), 
            "limit_price": final_limit,
            "sl_price": final_sl,
            "tp_price": final_tp,
            "position_size_pct": final_pos_size_pct,
            "entry_order_params_theoretical_for_oos_log": entry_order_params_log,
            "oco_params_theoretical_for_oos_log": oco_params_log
        }

    def _get_default_signal_response(self, reason: str) -> Dict[str, Any]:
        """Retourne une réponse de signal HOLD par défaut."""
        logger.debug(f"{self.log_prefix} Retour de HOLD : {reason}")
        return {
            "signal": 0, "order_type": "MARKET", "limit_price": None,
            "sl_price": None, "tp_price": None, "position_size_pct": 1.0,
            "entry_order_params_theoretical_for_oos_log": None,
            "oco_params_theoretical_for_oos_log": None
        }

    def _calculate_quantity(self,
                            entry_price: float,
                            available_capital: float,
                            qty_precision: Optional[int], 
                            symbol_info: Dict[str, Any], 
                            symbol: str, 
                            position_size_pct: Optional[float] = None
                           ) -> Optional[float]:
        """
        Calcule la quantité d'actifs de base à trader.
        """
        calc_log_prefix = f"{self.log_prefix}[CalcQty]"
        if entry_price <= 1e-9: 
            logger.error(f"{calc_log_prefix} Prix d'entrée ({entry_price}) invalide pour le calcul de la quantité.")
            return None
        if available_capital <= 1e-9:
            logger.warning(f"{calc_log_prefix} Capital disponible ({available_capital}) nul ou négatif. Aucune quantité ne peut être calculée.")
            return 0.0

        capital_alloc_pct_strat = self.get_param('capital_allocation_pct', 1.0) 
        actual_pos_size_pct = position_size_pct if position_size_pct is not None else capital_alloc_pct_strat
        
        if not (isinstance(actual_pos_size_pct, (float, int)) and 0 < actual_pos_size_pct <= 1.0):
            logger.warning(f"{calc_log_prefix} position_size_pct ({actual_pos_size_pct}) invalide. Utilisation de 100% (1.0).")
            actual_pos_size_pct = 1.0
        
        capital_for_this_trade = available_capital * actual_pos_size_pct
        
        total_position_value_quote = capital_for_this_trade * self.leverage
        
        quantity_base_raw = total_position_value_quote / entry_price
        logger.debug(f"{calc_log_prefix} Capital: {available_capital:.2f}, Alloc%: {actual_pos_size_pct:.2%}, "
                     f"Levier: {self.leverage}x, Capital pour trade: {capital_for_this_trade:.2f}, "
                     f"Valeur totale pos (quote): {total_position_value_quote:.2f}, "
                     f"Prix entrée: {entry_price:.{self.price_precision or 8}f}, Qty brute (base): {quantity_base_raw:.8f}")

        if qty_precision is None:
            logger.warning(f"{calc_log_prefix} qty_precision non disponible pour {symbol}. "
                           "La quantité ne sera pas ajustée finement à la précision de l'exchange.")
        
        adjusted_quantity_base = adjust_quantity_to_step_size(
            quantity_base_raw, 
            symbol_info, 
            qty_precision=qty_precision 
        )
        
        if adjusted_quantity_base is None or adjusted_quantity_base <= 1e-9: 
            logger.warning(f"{calc_log_prefix} Quantité ajustée ({adjusted_quantity_base}) est nulle ou négative après application de step_size/précision. "
                           f"Qty brute: {quantity_base_raw:.8f}")
            return 0.0

        min_qty_filter = get_filter_value(symbol_info, 'LOT_SIZE', 'minQty')
        if min_qty_filter is not None and adjusted_quantity_base < min_qty_filter:
            logger.warning(f"{calc_log_prefix} Quantité ajustée {adjusted_quantity_base:.{qty_precision or 8}f} < minQty requis {min_qty_filter:.{qty_precision or 8}f}. Retour de 0.")
            return 0.0
        
        min_notional_filter = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional')
        if min_notional_filter is not None and (adjusted_quantity_base * entry_price) < min_notional_filter:
            logger.warning(f"{calc_log_prefix} Valeur notionnelle de l'ordre ({adjusted_quantity_base * entry_price:.2f}) "
                           f"< MIN_NOTIONAL requis ({min_notional_filter:.2f}). Retour de 0.")
            return 0.0
            
        logger.info(f"{calc_log_prefix} Quantité finale calculée pour {symbol}: {adjusted_quantity_base:.{qty_precision or 8}f} {self.base_asset or ''}")
        return adjusted_quantity_base

    def _build_entry_params_formatted(self,
                                      side: str, 
                                      quantity_str: str,
                                      order_type: str, 
                                      entry_price_str: Optional[str] = None, 
                                      time_in_force: Optional[str] = None, 
                                      new_client_order_id: Optional[str] = None
                                     ) -> Dict[str, Any]:
        """
        Construit le dictionnaire de paramètres pour un ordre d'entrée.
        """
        if new_client_order_id is None:
            strat_short = self.strategy_name[:min(len(self.strategy_name), 5)].lower()
            sym_short = self.symbol[:min(len(self.symbol),3)].lower()
            ts_short = str(int(pd.Timestamp.now(tz='UTC').timestamp() * 1000))[-6:]
            uuid_short = str(uuid.uuid4().hex)[:4]
            new_client_order_id = f"sim_{strat_short}_{sym_short}_{ts_short}_{uuid_short}"
            new_client_order_id = new_client_order_id[:36]

        params: Dict[str, Any] = {
            "symbol": self.symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity_str,
            "newClientOrderId": new_client_order_id
        }
        if order_type.upper() == "LIMIT":
            if entry_price_str is None:
                msg = f"Le prix (entry_price_str) doit être spécifié pour un ordre LIMIT. Reçu: {entry_price_str}"
                logger.error(f"{self.log_prefix}[BuildEntryParams] {msg}")
                raise ValueError(msg)
            params["price"] = entry_price_str
        
        if time_in_force:
            params["timeInForce"] = time_in_force

        logger.debug(f"{self.log_prefix}[BuildEntryParams] Paramètres d'ordre d'entrée formatés : {params}")
        return params

    def _build_oco_params_formatted(self,
                                    entry_side: str, 
                                    quantity_str: str, 
                                    sl_price_str: str, 
                                    tp_price_str: str, 
                                    stop_limit_price_str: Optional[str] = None, 
                                    stop_limit_time_in_force: Optional[str] = "GTC",
                                    list_client_order_id: Optional[str] = None
                                   ) -> Dict[str, Any]:
        """
        Construit le dictionnaire de paramètres pour un ordre OCO.
        """
        exit_side = "SELL" if entry_side.upper() == "BUY" else "BUY"
        
        # Sourcery: Replace if-expression with `or`
        list_client_order_id = list_client_order_id or f"oco_{self.strategy_name[:min(len(self.strategy_name), 4)].lower()}_{self.symbol[:min(len(self.symbol),3)].lower()}_{str(int(pd.Timestamp.now(tz='UTC').timestamp() * 1000))[-5:]}_{str(uuid.uuid4().hex)[:3]}"
        list_client_order_id = list_client_order_id[:32]


        oco_params: Dict[str, Any] = {
            "symbol": self.symbol,
            "side": exit_side, 
            "quantity": quantity_str, 
            "price": tp_price_str,  
            "stopPrice": sl_price_str, 
            "stopLimitPrice": stop_limit_price_str if stop_limit_price_str else sl_price_str, 
            "listClientOrderId": list_client_order_id
        }
        
        if oco_params.get("stopLimitPrice") and stop_limit_time_in_force:
            oco_params["stopLimitTimeInForce"] = stop_limit_time_in_force
        elif oco_params.get("stopLimitPrice") and not stop_limit_time_in_force:
            oco_params["stopLimitTimeInForce"] = "GTC"

        logger.debug(f"{self.log_prefix}[BuildOCOParams] Paramètres d'ordre OCO formatés : {oco_params}")
        return oco_params

    @abstractmethod
    def get_required_indicator_configs(self) -> List[Dict[str, Any]]:
        """
        Déclare les indicateurs requis par la stratégie et leurs configurations.
        """
        return []

    @abstractmethod
    def generate_order_request(self,
                               data: pd.DataFrame, 
                               current_position: int, 
                               available_capital: float, 
                               symbol_info: Dict[str, Any] 
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Génère une requête d'ordre d'ENTRÉE pour le trading en direct ou une simulation détaillée.
        """
        pass
