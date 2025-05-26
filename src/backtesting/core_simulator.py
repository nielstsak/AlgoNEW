# src/backtesting/core_simulator.py
"""
Ce module définit BacktestRunner, le moteur principal de simulation de backtest.
Il prend en entrée des données OHLCV avec tous les indicateurs nécessaires déjà
calculés et une instance de stratégie configurée, puis simule l'exécution des trades.
"""
import pandas as pd
import numpy as np
import uuid # Pour les identifiants de cycle de trade uniques pour le log OOS
from datetime import datetime, timezone
import logging
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.strategies.base import BaseStrategy

# Utilisation des simulateurs de slippage et de frais des modules existants.
# Si ceux-ci évoluent vers src.utils.simulation_elements, les imports devront être mis à jour.
try:
    from src.utils.slippage import SlippageSimulator
    from src.utils.fees import FeeSimulator
    from src.utils.exchange_utils import get_precision_from_filter, get_filter_value, adjust_precision
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).critical(
        f"BacktestRunner: Erreur d'importation critique pour Slippage/FeeSimulator ou exchange_utils: {e}."
    )
    # Définir des placeholders pour permettre le chargement du module en cas d'erreur d'import
    class SlippageSimulator:
        def __init__(self, **kwargs): pass
        def simulate_slippage(self, price: float, **kwargs) -> float: return price
    class FeeSimulator:
        def __init__(self, **kwargs): pass
        def calculate_fee(self, trade_value_quote: float) -> float: return 0.0
    def get_precision_from_filter(*args) -> Optional[int]: return 8
    def get_filter_value(*args) -> Optional[float]: return 0.0
    def adjust_precision(value: float, precision: Optional[int], tick_size: Optional[float] = None) -> float: return round(value, precision or 8)

    raise # Renvoyer l'erreur pour indiquer un problème critique

logger = logging.getLogger(__name__)

class BacktestRunner:
    """
    Simule des stratégies de trading sur des données historiques.
    Gère l'exécution des ordres, la gestion de position, le calcul du PnL,
    et le suivi des métriques de performance.
    """
    def __init__(self,
                 df_ohlcv_with_indicators: pd.DataFrame,
                 strategy_instance: 'BaseStrategy',
                 initial_equity: float,
                 leverage: int,
                 symbol: str,
                 pair_config: Dict[str, Any], # Infos de l'exchange (filtres, précisions)
                 trading_fee_bps: float,
                 slippage_config: Dict[str, Any], # Dictionnaire de configuration du slippage
                 is_futures: bool,
                 run_id: str, # ID du run WFO global
                 is_oos_simulation: bool = False,
                 verbosity: int = 1):
        """
        Initialise le BacktestRunner.

        Args:
            df_ohlcv_with_indicators (pd.DataFrame): DataFrame avec OHLCV de base (1-min)
                et toutes les colonnes d'indicateurs (_strat) nécessaires.
                L'index doit être un DatetimeIndex.
            strategy_instance (BaseStrategy): Instance de la stratégie configurée.
            initial_equity (float): Capital initial.
            leverage (int): Levier.
            symbol (str): Symbole de la paire (ex: BTCUSDT).
            pair_config (Dict[str, Any]): Informations de l'exchange pour la paire.
            trading_fee_bps (float): Frais de transaction en points de base.
            slippage_config (Dict[str, Any]): Configuration pour le simulateur de slippage.
            is_futures (bool): True si simulation de futures.
            run_id (str): ID du run WFO global.
            is_oos_simulation (bool): True si simulation OOS (logging détaillé).
            verbosity (int): Niveau de verbosité des logs (0, 1, 2).
        """
        self.df_ohlcv = df_ohlcv_with_indicators.copy() # Renommé pour clarté interne
        self.strategy = strategy_instance
        self.initial_equity = float(initial_equity)
        self.current_equity = float(initial_equity)
        self.leverage = int(leverage)
        self.symbol = symbol.upper()
        self.pair_config = pair_config
        self.is_futures = is_futures # Actuellement peu utilisé, mais conservé
        self.run_id = run_id
        self.is_oos_simulation = is_oos_simulation
        self.verbosity = verbosity

        self.log_prefix = f"[BacktestRunner][{self.symbol}][Run:{self.run_id}]"
        self._log(f"Initialisation du BacktestRunner. Equity: {self.initial_equity}, Levier: {self.leverage}x, "
                  f"Futures: {self.is_futures}, OOS: {self.is_oos_simulation}, Verbosity: {self.verbosity}", level=1)
        self._log(f"Données OHLCV reçues shape: {self.df_ohlcv.shape}", level=2)


        # Initialisation des simulateurs de frais et de slippage
        self.slippage_simulator = SlippageSimulator(**slippage_config)
        self.fee_simulator = FeeSimulator(fee_bps=trading_fee_bps)
        self._log(f"Slippage config: {slippage_config}, Fee BPS: {trading_fee_bps}", level=2)

        # Extraction des informations de la paire
        self.base_asset: str = self.pair_config.get('baseAsset', '')
        self.quote_asset: str = self.pair_config.get('quoteAsset', '')
        self.price_precision: Optional[int] = get_precision_from_filter(self.pair_config, 'PRICE_FILTER', 'tickSize')
        self.quantity_precision: Optional[int] = get_precision_from_filter(self.pair_config, 'LOT_SIZE', 'stepSize')
        self.min_notional: float = get_filter_value(self.pair_config, 'MIN_NOTIONAL', 'minNotional') or 0.0
        self.min_quantity: float = get_filter_value(self.pair_config, 'LOT_SIZE', 'minQty') or 0.0

        if self.price_precision is None: logger.warning(f"{self.log_prefix} Précision de prix non trouvée, défaut à 8.")
        if self.quantity_precision is None: logger.warning(f"{self.log_prefix} Précision de quantité non trouvée, défaut à 8.")
        self._log(f"Pair info: Base='{self.base_asset}', Quote='{self.quote_asset}', "
                  f"PricePrec={self.price_precision}, QtyPrec={self.quantity_precision}, "
                  f"MinNotional={self.min_notional}, MinQty={self.min_quantity}", level=2)

        # État de la simulation
        self.position_open: bool = False
        self.position_size_base: float = 0.0       # Taille en actif de base
        self.position_size_quote: float = 0.0      # Valeur en actif de cotation à l'entrée
        self.entry_price: float = 0.0              # Prix d'entrée réel (avec slippage)
        self.entry_time: Optional[pd.Timestamp] = None
        self.current_sl_price: Optional[float] = None
        self.current_tp_price: Optional[float] = None
        self.trade_direction: int = 0              # 1 pour long, -1 pour short, 0 si pas de position
        self.current_trade_entry_fee: float = 0.0  # Frais pour le trade d'entrée actuel

        # Stockage des résultats
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = [{'timestamp': self.df_ohlcv.index.min() if not self.df_ohlcv.empty else pd.Timestamp.now(tz='UTC'), 'equity': self.initial_equity, 'type': 'initial'}]
        self.daily_equity: Dict[Any, float] = {} # Clé: date, Valeur: équité à la fin de cette date

        # Pour le logging OOS détaillé
        self.oos_detailed_trades_log: List[Dict[str, Any]] = []
        self.current_trade_cycle_id: Optional[str] = None # ID unique pour un cycle entrée->sortie
        self.current_entry_order_params_theoretical: Optional[Dict[str, Any]] = None
        self.current_oco_params_theoretical: Optional[Dict[str, Any]] = None
        
        # L'instance de stratégie est déjà configurée, y compris set_backtest_context
        self._log(f"BacktestRunner initialisé pour la stratégie '{self.strategy.strategy_name}'.", level=1)

    def _log(self, message: str, level: int = 1):
        """Logue un message si le niveau de verbosité est suffisant."""
        if self.verbosity >= level:
            logger.info(f"{self.log_prefix} {message}")

    def _calculate_max_position_size_base(self, entry_price_estimate: float) -> float:
        """
        Calcule la taille maximale de position en actif de base, en tenant compte
        de l'équité, du levier, et de la précision de quantité.
        """
        if entry_price_estimate <= 1e-9: # Éviter division par zéro
            self._log(f"Prix d'entrée estimé ({entry_price_estimate}) trop bas pour calculer la taille de position.", level=2)
            return 0.0
        
        # Pour le trading sur marge, la "available_capital" est l'équité.
        # La valeur notionnelle maximale de la position est equity * leverage.
        max_notional_value_quote = self.current_equity * self.leverage
        max_size_base_raw = max_notional_value_quote / entry_price_estimate
        
        # Appliquer la précision de quantité (arrondi vers le bas)
        # La fonction adjust_quantity_to_step_size gère cela correctement.
        # Ici, on s'assure juste que qty_precision est un int pour le formatage.
        qty_prec = self.quantity_precision if self.quantity_precision is not None else 8 # Fallback
        
        # Note: adjust_quantity_to_step_size (de BaseStrategy) fait déjà l'arrondi basé sur stepSize.
        # Ici, on pourrait faire un pré-arrondi si stepSize n'est pas disponible, mais c'est moins précis.
        # Le mieux est de s'appuyer sur _apply_exchange_filters qui utilise adjust_quantity_to_step_size.
        # Pour cette fonction _calculate_max_position_size_base, on retourne une taille "brute" mais arrondie.
        
        factor = 10**qty_prec
        max_size_base_rounded = np.floor(max_size_base_raw * factor) / factor
        
        self._log(f"Calcul max_pos_size: Equity={self.current_equity:.2f}, Lev={self.leverage}, "
                  f"PrixEst={entry_price_estimate:.{self.price_precision or 8}f}, MaxNotionalVal={max_notional_value_quote:.2f}, "
                  f"MaxSizeBaseRaw={max_size_base_raw:.8f}, MaxSizeBaseRounded={max_size_base_rounded:.{qty_prec}f}", level=2)
        return max_size_base_rounded


    def _apply_exchange_filters(self, quantity_base: float, price: float) -> float:
        """
        Applique les filtres minQty et minNotional de l'exchange à une quantité.
        Retourne 0.0 si les filtres ne sont pas respectés.
        Utilise adjust_quantity_to_step_size pour la précision finale.
        """
        # D'abord, ajuster au stepSize (ce qui inclut l'arrondi à la quantity_precision)
        # La méthode adjust_quantity_to_step_size devrait être dans exchange_utils
        # et prendre symbol_info (pair_config ici) pour trouver le stepSize.
        
        # Si self.quantity_precision est None, adjust_quantity_to_step_size devrait avoir un fallback.
        qty_adjusted_to_step = adjust_quantity_to_step_size(quantity_base, self.pair_config, self.quantity_precision)

        if qty_adjusted_to_step <= 1e-9: # Si après ajustement au step, la quantité est quasi nulle
             self._log(f"Quantité {quantity_base:.8f} est devenue {qty_adjusted_to_step:.{self.quantity_precision or 8}f} "
                       f"après ajustement au stepSize. Considérée comme 0.", level=2)
             return 0.0

        # Vérifier minQty
        if self.min_quantity > 0 and qty_adjusted_to_step < self.min_quantity:
            self._log(f"Filtre Échec: Quantité ajustée {qty_adjusted_to_step:.{self.quantity_precision or 8}f} < minQty requis {self.min_quantity:.{self.quantity_precision or 8}f}.", level=2)
            return 0.0

        # Vérifier minNotional
        notional_value = qty_adjusted_to_step * price
        if self.min_notional > 0 and notional_value < self.min_notional:
            self._log(f"Filtre Échec: Valeur notionnelle {notional_value:.2f} (Qty: {qty_adjusted_to_step:.{self.quantity_precision or 8}f} @ Prix: {price:.{self.price_precision or 8}f}) "
                      f"< minNotional requis {self.min_notional:.2f}.", level=2)
            return 0.0
        
        self._log(f"Filtres appliqués: Qty brute: {quantity_base:.8f} -> Qty ajustée: {qty_adjusted_to_step:.{self.quantity_precision or 8}f}. Notionnel: {notional_value:.2f}", level=2)
        return qty_adjusted_to_step

    def _simulate_order_execution(self,
                                  timestamp: pd.Timestamp,
                                  current_bar_ohlc: pd.Series, # Contient open, high, low, close de la barre actuelle
                                  order_type: str, # "MARKET", "LIMIT_ENTRY", "MARKET_EXIT", "SL", "TP"
                                  trade_direction_signal: int, # 1 pour achat/long, -1 pour vente/short (pour l'entrée)
                                                               # ou direction de la position existante pour la sortie
                                  limit_price: Optional[float] = None,
                                  stop_price: Optional[float] = None, # Pour les ordres SL
                                  target_quantity_base_on_entry: Optional[float] = None, # Pour les entrées
                                  is_entry_order: bool = False
                                 ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Simule l'exécution d'un ordre pour la barre actuelle.

        Args:
            timestamp: Timestamp de la barre actuelle.
            current_bar_ohlc: Séries Pandas contenant open, high, low, close de la barre.
            order_type: Type d'ordre à simuler.
            trade_direction_signal: Direction du trade (1 pour long, -1 pour short).
                                   Pour les sorties, c'est la direction de la position à clôturer.
            limit_price: Prix limite pour les ordres LIMIT ou TP.
            stop_price: Prix de déclenchement pour les ordres SL.
            target_quantity_base_on_entry: Quantité cible pour un ordre d'entrée.
            is_entry_order: True si c'est un ordre d'entrée, False si c'est une sortie.

        Returns:
            Tuple[Optional[float], Optional[float], Optional[float]]:
            (prix_execution_final, quantite_executee_base, frais_payes_quote)
            Retourne (None, None, None) si l'ordre n'est pas exécuté.
        """
        exec_log_prefix = f"{self.log_prefix}[ExecSim][{timestamp}][{order_type}]"
        
        open_p = current_bar_ohlc['open']
        high_p = current_bar_ohlc['high']
        low_p = current_bar_ohlc['low']
        # close_p = current_bar_ohlc['close'] # Non utilisé directement pour l'exécution ici

        base_execution_price: Optional[float] = None

        # Déterminer le prix de base de l'exécution (avant slippage)
        if order_type == 'MARKET' or order_type == 'MARKET_EXIT':
            base_execution_price = open_p # Exécution au prix d'ouverture de la barre suivante
        elif order_type == 'LIMIT_ENTRY' and limit_price is not None:
            if trade_direction_signal == 1: # Achat LIMIT
                if low_p <= limit_price: # Le prix a touché ou dépassé la limite à la baisse
                    base_execution_price = min(open_p, limit_price) # Exécuté au meilleur des deux
            else: # Vente LIMIT (-1)
                if high_p >= limit_price: # Le prix a touché ou dépassé la limite à la hausse
                    base_execution_price = max(open_p, limit_price)
        elif order_type == 'SL' and stop_price is not None:
            # Pour un SL sur une position LONG (trade_direction_signal == 1), on vend si le prix baisse.
            # Pour un SL sur une position SHORT (trade_direction_signal == -1), on achète si le prix monte.
            if trade_direction_signal == 1: # SL sur un LONG (ordre de VENTE)
                if low_p <= stop_price:
                    base_execution_price = min(open_p, stop_price) # Vendu au stop_price ou à l'open si pire
            else: # SL sur un SHORT (ordre d'ACHAT) (-1)
                if high_p >= stop_price:
                    base_execution_price = max(open_p, stop_price) # Acheté au stop_price ou à l'open si pire
        elif order_type == 'TP' and limit_price is not None:
            # Pour un TP sur une position LONG (trade_direction_signal == 1), on vend si le prix monte.
            # Pour un TP sur une position SHORT (trade_direction_signal == -1), on achète si le prix baisse.
            if trade_direction_signal == 1: # TP sur un LONG (ordre de VENTE)
                if high_p >= limit_price:
                    base_execution_price = max(open_p, limit_price) # Vendu au limit_price ou à l'open si mieux
            else: # TP sur un SHORT (ordre d'ACHAT) (-1)
                if low_p <= limit_price:
                    base_execution_price = min(open_p, limit_price) # Acheté au limit_price ou à l'open si mieux
        
        if base_execution_price is None:
            self._log(f"{exec_log_prefix} Ordre non exécutable dans cette barre (conditions non remplies). "
                      f"L:{limit_price}, S:{stop_price}, Dir:{trade_direction_signal}, OHLC:[{open_p},{high_p},{low_p}]", level=2)
            return None, None, None

        # Appliquer le slippage
        # La direction du slippage est toujours défavorable.
        # Pour une entrée LONG ou une sortie de SHORT (ACHAT), le prix slippé est > prix de base.
        # Pour une entrée SHORT ou une sortie de LONG (VENTE), le prix slippé est < prix de base.
        slippage_direction_factor = 0
        if is_entry_order:
            slippage_direction_factor = trade_direction_signal # 1 pour achat, -1 pour vente
        else: # C'est un ordre de sortie
            slippage_direction_factor = -trade_direction_signal # Inverse de la position

        slipped_price = self.slippage_simulator.simulate_slippage(
            price=base_execution_price,
            direction=slippage_direction_factor,
            # Les volumes/volatilité de la barre actuelle peuvent être utilisés pour des modèles de slippage plus complexes
            volume_at_price=current_bar_ohlc.get('volume', 0),
            volatility=(high_p - low_p) / low_p if low_p > 1e-9 else 0.0
        )
        
        # Arrondir le prix exécuté à la précision de l'exchange
        final_executed_price = adjust_precision(slipped_price, self.price_precision, get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize'))
        if final_executed_price is None: # Si adjust_precision retourne None (erreur)
            logger.error(f"{exec_log_prefix} Erreur d'ajustement de précision pour le prix slippé {slipped_price}.")
            return None, None, None


        # Déterminer la quantité exécutée
        executed_quantity_base: float
        if is_entry_order:
            if target_quantity_base_on_entry is None or target_quantity_base_on_entry <= 1e-9:
                self._log(f"{exec_log_prefix} Quantité cible pour l'entrée non valide ({target_quantity_base_on_entry}). Pas d'exécution.", level=1)
                return None, None, None
            executed_quantity_base = self._apply_exchange_filters(target_quantity_base_on_entry, final_executed_price)
            if executed_quantity_base <= 1e-9:
                self._log(f"{exec_log_prefix} Quantité d'entrée {target_quantity_base_on_entry:.8f} est devenue nulle après application des filtres. Pas d'exécution.", level=1)
                return None, None, None
        else: # Ordre de sortie, la quantité est la taille de la position actuelle
            if not self.position_open or self.position_size_base <= 1e-9:
                self._log(f"{exec_log_prefix} Tentative de sortie mais pas de position ouverte ou taille de position nulle. Pas d'exécution.", level=2)
                return None, None, None
            executed_quantity_base = self.position_size_base


        # Calculer les frais
        trade_value_quote = executed_quantity_base * final_executed_price
        fee_paid_quote = self.fee_simulator.calculate_fee(trade_value_quote)

        self._log(f"{exec_log_prefix} Exécution simulée : PrixBase={base_execution_price:.{self.price_precision or 8}f}, "
                  f"PrixSlippé={slipped_price:.{self.price_precision or 8}f}, PrixFinal={final_executed_price:.{self.price_precision or 8}f}, "
                  f"QtyBase={executed_quantity_base:.{self.quantity_precision or 8}f}, FraisQuote={fee_paid_quote:.4f}", level=2)

        # Mettre à jour l'état de la simulation
        if is_entry_order:
            self.position_open = True
            self.trade_direction = trade_direction_signal
            self.entry_price = final_executed_price
            self.entry_time = timestamp
            self.position_size_base = executed_quantity_base
            self.position_size_quote = trade_value_quote # Valeur notionnelle à l'entrée
            self.current_trade_entry_fee = fee_paid_quote

            # Logique de marge (simplifiée) : l'équité est affectée par les frais.
            # Le "prêt" est implicite dans le calcul du PnL plus tard.
            self.current_equity -= fee_paid_quote
            self.equity_curve.append({'timestamp': timestamp, 'equity': self.current_equity, 'type': 'entry_fee'})
            
            self._log(f"ENTRÉE {'LONG' if self.trade_direction == 1 else 'SHORT'} @ {self.entry_price:.{self.price_precision or 8}f}, "
                      f"Qty: {self.position_size_base:.{self.quantity_precision or 8}f}. Frais: {fee_paid_quote:.4f}. Equity: {self.current_equity:.2f}", level=1)

            if self.is_oos_simulation:
                self.current_trade_cycle_id = str(uuid.uuid4()) # Nouveau cycle de trade pour OOS log

        else: # C'est une sortie de position
            self.exit_price = final_executed_price
            self.exit_time = timestamp

            # Calcul du PnL
            pnl_gross_quote: float
            if self.trade_direction == 1: # Sortie d'un Long
                pnl_gross_quote = (self.exit_price - self.entry_price) * self.position_size_base
            else: # Sortie d'un Short (self.trade_direction == -1)
                pnl_gross_quote = (self.entry_price - self.exit_price) * self.position_size_base
            
            pnl_net_quote = pnl_gross_quote - self.current_trade_entry_fee - fee_paid_quote # Soustraire les frais d'entrée et de sortie
            self.current_equity += pnl_net_quote # Le PnL net affecte directement l'équité
            
            self.equity_curve.append({'timestamp': timestamp, 'equity': self.current_equity, 'type': 'trade_pnl'})
            
            trade_details = {
                'entry_time': self.entry_time, 'exit_time': self.exit_time,
                'entry_price': self.entry_price, 'exit_price': self.exit_price,
                'size_base': self.position_size_base, 'size_quote_at_entry': self.position_size_quote,
                'pnl_gross_quote': pnl_gross_quote,
                'pnl_net_quote': pnl_net_quote,
                'entry_fee_quote': self.current_trade_entry_fee,
                'exit_fee_quote': fee_paid_quote,
                'total_fees_quote': self.current_trade_entry_fee + fee_paid_quote,
                'direction': 'long' if self.trade_direction == 1 else 'short',
                'leverage': self.leverage,
                'exit_reason': order_type # SL, TP, ou MARKET_EXIT (signal strat)
            }
            self.trades.append(trade_details)
            
            self._log(f"SORTIE {'LONG' if self.trade_direction == 1 else 'SHORT'} @ {self.exit_price:.{self.price_precision or 8}f} (Raison: {order_type}). "
                      f"PnL Net: {pnl_net_quote:.2f}. Frais Sortie: {fee_paid_quote:.4f}. Equity: {self.current_equity:.2f}", level=1)

            if self.is_oos_simulation and self.current_trade_cycle_id:
                self._record_oos_detailed_trade(trade_details)

            self._reset_position_state()

        return final_executed_price, executed_quantity_base, fee_paid_quote

    def _record_oos_detailed_trade(self, trade_info: Dict[str, Any]):
        """Enregistre un log détaillé du cycle de trade pour l'analyse OOS."""
        if not self.current_trade_cycle_id:
            self.current_trade_cycle_id = str(uuid.uuid4()) # Assurer un ID si manquant
            self._log(f"ID de cycle de trade OOS manquant, génération d'un nouveau : {self.current_trade_cycle_id}", level=2)

        # Les paramètres théoriques auraient dû être stockés sur self par run_simulation
        entry_params_theo = self.current_entry_order_params_theoretical or {}
        oco_params_theo = self.current_oco_params_theoretical or {}
        
        is_short_trade = trade_info['direction'] == 'short'
        
        # Construire le type d'intention d'ordre
        intent_type_str = "UNKNOWN_INTENT"
        if entry_params_theo.get("type"):
            intent_type_str = f"{entry_params_theo['type'].upper()}_ENTRY_{'SHORT' if is_short_trade else 'LONG'}"
            if oco_params_theo: # Si OCO était prévu
                 intent_type_str = f"OTOCO_VIA_{intent_type_str}"


        log_entry = {
            "trade_cycle_id": self.current_trade_cycle_id,
            "run_id": self.run_id, # ID du run WFO global
            "strategy_name": self.strategy.strategy_name,
            "pair_symbol": self.symbol,
            "is_oos_simulation": True,
            
            "entry_timestamp_simulated_utc": trade_info['entry_time'].isoformat() if trade_info.get('entry_time') else None,
            "order_intent_type_simulated": intent_type_str,
            
            # Détails de l'ordre d'entrée théorique (tel que la stratégie l'aurait voulu)
            "entry_order_theoretical_params": entry_params_theo,
            # Détails de l'ordre OCO théorique
            "oco_order_theoretical_params": oco_params_theo,

            # Détails de l'exécution réelle (simulée) de l'entrée
            "entry_execution_price_slipped": trade_info['entry_price'],
            "entry_executed_quantity_base": trade_info['size_base'],
            "entry_fee_quote": trade_info['entry_fee_quote'],

            # Détails de la sortie
            "exit_timestamp_simulated_utc": trade_info['exit_time'].isoformat() if trade_info.get('exit_time') else None,
            "exit_reason_simulated": trade_info['exit_reason'],
            "exit_execution_price_slipped": trade_info['exit_price'],
            "exit_fee_quote": trade_info['exit_fee_quote'],
            
            # PnL
            "pnl_gross_quote_simulated": trade_info['pnl_gross_quote'],
            "pnl_net_quote_simulated": trade_info['pnl_net_quote'],
            "total_fees_quote_simulated": trade_info['total_fees_quote']
        }
        self.oos_detailed_trades_log.append(log_entry)
        self._log(f"Log OOS détaillé enregistré pour le cycle ID {self.current_trade_cycle_id}", level=2)

    def _reset_position_state(self):
        """Réinitialise l'état de la position après une clôture."""
        self.position_open = False
        self.position_size_base = 0.0
        self.position_size_quote = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        # self.exit_price et self.exit_time sont déjà mis à jour avant d'appeler ceci
        self.trade_direction = 0
        self.current_sl_price = None
        self.current_tp_price = None
        self.current_trade_entry_fee = 0.0
        
        # Réinitialiser les logs théoriques pour OOS pour le prochain cycle
        self.current_trade_cycle_id = None
        self.current_entry_order_params_theoretical = None
        self.current_oco_params_theoretical = None
        self._log("État de la position réinitialisé.", level=2)

    def run_simulation(self) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[Any, float], List[Dict[str, Any]]]:
        """
        Exécute la simulation de backtest barre par barre.

        Returns:
            Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[Any, float], List[Dict[str, Any]]]:
            - Liste des trades.
            - DataFrame de la courbe d'équité.
            - Dictionnaire de l'équité journalière.
            - Liste des logs de trades OOS détaillés (si applicable).
        """
        self._log(f"Démarrage de la simulation pour {len(self.df_ohlcv)} barres.", level=1)
        if self.df_ohlcv.empty:
            logger.warning(f"{self.log_prefix} Données OHLCV vides. Simulation sautée.")
            return self.trades, pd.DataFrame(self.equity_curve), self.daily_equity, self.oos_detailed_trades_log

        # Assurer que l'index est un DatetimeIndex
        if not isinstance(self.df_ohlcv.index, pd.DatetimeIndex):
            logger.error(f"{self.log_prefix} L'index de df_ohlcv n'est pas un DatetimeIndex. Simulation interrompue.")
            return self.trades, pd.DataFrame(self.equity_curve), self.daily_equity, self.oos_detailed_trades_log
        
        last_recorded_daily_equity_date: Optional[pd.Timestamp] = None

        for i in range(len(self.df_ohlcv)):
            current_timestamp: pd.Timestamp = self.df_ohlcv.index[i] # type: ignore
            current_row: pd.Series = self.df_ohlcv.iloc[i]

            # Enregistrer l'équité journalière
            current_date = current_timestamp.normalize() # Date à minuit
            if last_recorded_daily_equity_date is None or current_date > last_recorded_daily_equity_date:
                self.daily_equity[current_date] = self.current_equity
                last_recorded_daily_equity_date = current_date
            elif i == len(self.df_ohlcv) - 1: # Toujours enregistrer pour la dernière barre
                 self.daily_equity[current_date] = self.current_equity


            # 1. Gérer les sorties (SL/TP) si une position est ouverte
            exit_simulated_this_bar = False
            if self.position_open:
                # Vérifier TP
                if self.current_tp_price is not None:
                    # _simulate_order_execution gère la logique de si le prix est atteint
                    exec_price_tp, _, _ = self._simulate_order_execution(
                        timestamp=current_timestamp, current_bar_ohlc=current_row,
                        order_type='TP', trade_direction_signal=self.trade_direction,
                        limit_price=self.current_tp_price, is_entry_order=False
                    )
                    if exec_price_tp is not None:
                        self._log(f"TP ATTEINT @ {exec_price_tp:.{self.price_precision or 8}f}", level=1)
                        exit_simulated_this_bar = True
                
                # Vérifier SL (seulement si TP n'a pas été atteint sur la même barre)
                if not exit_simulated_this_bar and self.current_sl_price is not None:
                    exec_price_sl, _, _ = self._simulate_order_execution(
                        timestamp=current_timestamp, current_bar_ohlc=current_row,
                        order_type='SL', trade_direction_signal=self.trade_direction,
                        stop_price=self.current_sl_price, is_entry_order=False
                    )
                    if exec_price_sl is not None:
                        self._log(f"SL ATTEINT @ {exec_price_sl:.{self.price_precision or 8}f}", level=1)
                        exit_simulated_this_bar = True
                
                if exit_simulated_this_bar:
                    continue # Passer à la barre suivante si un SL/TP a clôturé la position

            # 2. Obtenir le signal de la stratégie
            # Le slice de données doit inclure la barre actuelle pour que la stratégie la voie.
            df_slice_for_signal = self.df_ohlcv.iloc[:i+1]
            
            signal_dict = self.strategy.get_signal(
                data_feed=df_slice_for_signal,
                current_position_open=self.position_open,
                current_position_direction=self.trade_direction,
                current_entry_price=self.entry_price,
                current_equity=self.current_equity
            )

            signal_type = signal_dict.get("signal", 0)
            order_type_strat = signal_dict.get("order_type", "MARKET")
            limit_price_strat = signal_dict.get("limit_price")
            sl_price_strat = signal_dict.get("sl_price")
            tp_price_strat = signal_dict.get("tp_price")
            pos_size_pct_strat = signal_dict.get("position_size_pct", 1.0)
            
            # Stocker les paramètres théoriques pour le log OOS si c'est une simulation OOS
            # et si un signal d'entrée est généré.
            if self.is_oos_simulation and signal_type in [1, -1] and not self.position_open:
                self.current_entry_order_params_theoretical = signal_dict.get("entry_order_params_theoretical_for_oos_log")
                self.current_oco_params_theoretical = signal_dict.get("oco_params_theoretical_for_oos_log")


            # 3. Traiter les signaux
            if self.position_open:
                # Signal de sortie explicite (2) ou signal de renversement
                if signal_type == 2 or \
                   (self.trade_direction == 1 and signal_type == -1) or \
                   (self.trade_direction == -1 and signal_type == 1):
                    self._log(f"SIGNAL DE SORTIE STRATÉGIE reçu pour position {'LONG' if self.trade_direction==1 else 'SHORT'}.", level=1)
                    self._simulate_order_execution(
                        timestamp=current_timestamp, current_bar_ohlc=current_row,
                        order_type='MARKET_EXIT', trade_direction_signal=self.trade_direction,
                        is_entry_order=False
                    )
                    # _reset_position_state est appelé dans _simulate_order_execution pour les sorties
            
            elif not self.position_open and signal_type in [1, -1]: # Signal d'entrée
                entry_direction = signal_type
                self._log(f"SIGNAL D'ENTRÉE {'LONG' if entry_direction == 1 else 'SHORT'} reçu.", level=1)

                # Estimer le prix d'entrée pour le calcul de la taille
                entry_price_estimate_for_sizing: float
                if order_type_strat == 'LIMIT' and limit_price_strat is not None:
                    entry_price_estimate_for_sizing = limit_price_strat
                else: # Pour MARKET, utiliser l'open de la barre actuelle comme estimation
                    entry_price_estimate_for_sizing = current_row['open']
                
                if pd.isna(entry_price_estimate_for_sizing):
                    self._log(f"Prix d'estimation pour la taille d'entrée est NaN. Open: {current_row['open']}, Limit: {limit_price_strat}. Pas d'entrée.", level=1)
                    continue

                max_possible_qty_base = self._calculate_max_position_size_base(entry_price_estimate_for_sizing)
                
                if max_possible_qty_base <= 1e-9: # 1e-9 pour gérer les erreurs de flottants
                    self._log(f"Quantité maximale possible de base est {max_possible_qty_base:.8f}. Pas assez de capital/levier pour entrer.", level=1)
                    continue

                target_qty_base = max_possible_qty_base * (pos_size_pct_strat if pos_size_pct_strat is not None else 1.0)
                
                if target_qty_base <= 1e-9:
                    self._log(f"Quantité cible de base ({target_qty_base:.8f}) trop petite. Pas d'entrée.", level=1)
                    continue

                # Simuler l'exécution de l'ordre d'entrée
                sim_entry_order_type = 'LIMIT_ENTRY' if order_type_strat == 'LIMIT' else 'MARKET'
                exec_price, exec_qty, _ = self._simulate_order_execution(
                    timestamp=current_timestamp, current_bar_ohlc=current_row,
                    order_type=sim_entry_order_type, trade_direction_signal=entry_direction,
                    limit_price=limit_price_strat if order_type_strat == 'LIMIT' else None,
                    target_quantity_base_on_entry=target_qty_base,
                    is_entry_order=True
                )

                if exec_price is not None and exec_qty is not None and exec_qty > 1e-9:
                    # L'état de la position (self.position_open, self.entry_price, etc.)
                    # est mis à jour dans _simulate_order_execution.
                    # Mettre à jour les SL/TP pour la nouvelle position.
                    self.current_sl_price = sl_price_strat
                    self.current_tp_price = tp_price_strat
                    self._log(f"SL/TP mis à jour après entrée: SL={self.current_sl_price}, TP={self.current_tp_price}", level=2)
                else:
                    # L'entrée a échoué (non exécutée ou quantité nulle après filtres)
                    self._log(f"L'ordre d'entrée n'a pas été exécuté ou quantité nulle. Pas de position prise.", level=1)
                    # S'assurer de réinitialiser les logs OOS si l'entrée a échoué
                    if self.is_oos_simulation:
                        self.current_trade_cycle_id = None
                        self.current_entry_order_params_theoretical = None
                        self.current_oco_params_theoretical = None


        # S'assurer que l'équité de la dernière date est enregistrée si elle ne l'a pas été
        if not self.df_ohlcv.empty:
            final_date = self.df_ohlcv.index[-1].normalize() # type: ignore
            if final_date not in self.daily_equity:
                self.daily_equity[final_date] = self.current_equity
            # Enregistrer l'équité finale dans la courbe
            self.equity_curve.append({'timestamp': self.df_ohlcv.index[-1], 'equity': self.current_equity, 'type': 'final'})

        self._log(f"Simulation terminée. Nombre total de trades: {len(self.trades)}. Équité finale: {self.current_equity:.2f}", level=1)
        
        # Convertir la liste equity_curve en DataFrame pour le retour
        equity_curve_df_final = pd.DataFrame(self.equity_curve)
        if not equity_curve_df_final.empty and 'timestamp' in equity_curve_df_final.columns:
            equity_curve_df_final['timestamp'] = pd.to_datetime(equity_curve_df_final['timestamp'], errors='coerce', utc=True)
            # Pas besoin de set_index ici, car ObjectiveFunctionEvaluator le fera si nécessaire.

        return self.trades, equity_curve_df_final, self.daily_equity, self.oos_detailed_trades_log

