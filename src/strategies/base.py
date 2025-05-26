# src/strategies/base.py
"""
Ce module définit la classe de base abstraite pour toutes les stratégies de trading.
Elle fournit une interface standardisée, facilite la gestion des paramètres,
et intègre les exigences de la nouvelle architecture de backtesting et de trading live.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
import uuid
from typing import Dict, Any, Tuple, Optional, List, Union # Union ajouté pour la compatibilité

# Assurez-vous que ces imports sont corrects par rapport à votre structure de projet.
# Il est supposé que ces utilitaires existent et sont accessibles.
try:
    from src.utils.exchange_utils import (
        adjust_precision,
        adjust_quantity_to_step_size, # Sera utilisé dans _calculate_quantity
        get_precision_from_filter,
        get_filter_value
    )
except ImportError:
    logging.getLogger(__name__).critical(
        "BaseStrategy: Échec de l'importation des utilitaires d'exchange. "
        "Les fonctionnalités de précision et de filtrage pourraient ne pas fonctionner."
    )
    # Définir des placeholders si les imports échouent pour permettre au reste de se charger
    def get_precision_from_filter(pair_config: Dict, filter_type: str, key: str) -> Optional[int]: return 8
    def adjust_precision(value: float, precision: Optional[int], tick_size: Optional[float] = None) -> float: return round(value, precision or 8)
    def adjust_quantity_to_step_size(quantity: float, symbol_info: Dict, qty_precision: Optional[int]) -> float: return round(quantity, qty_precision or 8)
    def get_filter_value(symbol_info: Dict, filter_type: str, key: str) -> Optional[float]: return None


logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Classe de base abstraite pour toutes les stratégies de trading.
    """
    REQUIRED_PARAMS: List[str] = [] # Les classes filles peuvent surcharger cette liste

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
        
        # Le type de compte peut être utile pour certaines logiques de stratégie (ex: MARGIN vs SPOT)
        # Il sera défini dans set_backtest_context si fourni par le simulateur,
        # ou pourrait être un paramètre de la stratégie elle-même.
        self.account_type: Optional[str] = None


        # Valider les paramètres spécifiques à la stratégie (doit être implémenté par la classe fille)
        try:
            self._validate_params()
            logger.info(f"{self.log_prefix} Stratégie initialisée avec les paramètres : {self.params}")
        except ValueError as e_val:
            logger.error(f"{self.log_prefix} Erreur de validation des paramètres lors de l'initialisation : {e_val}")
            raise # Renvoyer l'erreur pour empêcher l'utilisation d'une stratégie mal configurée

    def get_param(self, param_name: str, default: Optional[Any] = None) -> Any:
        """
        Récupère une valeur de paramètre pour la stratégie.

        Args:
            param_name (str): Le nom du paramètre à récupérer.
            default (Optional[Any]): La valeur par défaut à retourner si le paramètre n'est pas trouvé.

        Returns:
            Any: La valeur du paramètre ou la valeur par défaut.
        """
        return self.params.get(param_name, default)

    def set_backtest_context(self,
                             pair_config: Dict[str, Any],
                             is_futures: bool,
                             leverage: int,
                             initial_equity: float,
                             account_type: Optional[str] = "SPOT" # Par défaut SPOT si non spécifié
                            ):
        """
        Configure le contexte pour un backtest.

        Args:
            pair_config (Dict[str, Any]): Configuration de la paire de l'exchange (filtres, précisions).
            is_futures (bool): True si c'est un backtest de futures.
            leverage (int): Levier utilisé pour le backtest.
            initial_equity (float): Capital initial pour le backtest.
            account_type (Optional[str]): Type de compte (ex: "SPOT", "MARGIN", "ISOLATED_MARGIN").
        """
        self.pair_config = pair_config
        self.is_futures = is_futures
        self.leverage = leverage
        self.initial_equity = initial_equity
        self.account_type = account_type

        if self.pair_config:
            self.price_precision = get_precision_from_filter(self.pair_config, 'PRICE_FILTER', 'tickSize')
            self.quantity_precision = get_precision_from_filter(self.pair_config, 'LOT_SIZE', 'stepSize')
            
            if self.price_precision is None:
                 logger.warning(f"{self.log_prefix} Price precision (tickSize) non trouvée dans pair_config pour PRICE_FILTER. Utilisation d'une valeur par défaut ou la stratégie pourrait échouer.")
                 # Une valeur par défaut pourrait être assignée ici, ex: self.price_precision = 8
            if self.quantity_precision is None:
                 logger.warning(f"{self.log_prefix} Quantity precision (stepSize) non trouvée dans pair_config pour LOT_SIZE. Utilisation d'une valeur par défaut ou la stratégie pourrait échouer.")
                 # self.quantity_precision = 8
            logger.debug(f"{self.log_prefix} Contexte de backtest défini. Price Precision: {self.price_precision}, Quantity Precision: {self.quantity_precision}, Leverage: {self.leverage}")
        else:
            logger.error(f"{self.log_prefix} pair_config non fourni à set_backtest_context. Les précisions ne peuvent pas être déterminées.")
            # Les précisions resteront None, ce qui pourrait causer des problèmes plus tard.

    @abstractmethod
    def _validate_params(self) -> None:
        """
        Valide les paramètres spécifiques à la stratégie (self.params).
        Doit être implémentée par les classes filles.
        Doit lever une ValueError si un paramètre est invalide.
        """
        pass

    @abstractmethod
    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame:
        """
        Assure que le data_feed (provenant de IndicatorCalculator) contient les colonnes
        d'indicateurs finales (suffixées par `_strat`) attendues par la logique de signaux.
        Ne calcule plus les indicateurs elle-même mais vérifie leur présence et peut
        effectuer des ajustements finaux (ex: ffill).

        Args:
            data_feed (pd.DataFrame): DataFrame contenant les données OHLCV de base et
                                      les indicateurs calculés par IndicatorCalculator.

        Returns:
            pd.DataFrame: Le DataFrame prêt pour la génération de signaux.
        """
        # Les classes filles doivent vérifier la présence des colonnes _strat dont elles ont besoin
        # et potentiellement appliquer un ffill.
        # Exemple:
        # required_strat_cols = ['EMA_FAST_strat', 'RSI_strat']
        # missing_cols = [col for col in required_strat_cols if col not in data_feed.columns]
        # if missing_cols:
        #     logger.error(f"{self.log_prefix} Colonnes indicateur _strat manquantes: {missing_cols}")
        #     # Gérer l'erreur, peut-être retourner un DataFrame vide ou lever une exception
        # data_feed[required_strat_cols] = data_feed[required_strat_cols].ffill()
        pass

    @abstractmethod
    def _generate_signals(self,
                          data_with_indicators: pd.DataFrame,
                          current_position_open: bool,
                          current_position_direction: int, # 0: no pos, 1: long, -1: short
                          current_entry_price: float
                         ) -> Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Génère les signaux de trading basés sur les indicateurs et l'état actuel du marché/position.

        Args:
            data_with_indicators (pd.DataFrame): DataFrame avec les indicateurs prêts.
            current_position_open (bool): True si une position est actuellement ouverte.
            current_position_direction (int): Direction de la position actuelle (1 pour long, -1 pour short, 0 si pas de position).
            current_entry_price (float): Prix d'entrée de la position actuelle.

        Returns:
            Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
            - signal_type (int): 0 (HOLD), 1 (BUY/LONG), -1 (SELL/SHORT), 2 (EXIT_CURRENT_POSITION).
            - order_type_preference (Optional[str]): "MARKET" ou "LIMIT".
            - limit_price_suggestion (Optional[float]): Prix limite si "LIMIT".
            - sl_price_suggestion (Optional[float]): Suggestion de prix Stop-Loss brut.
            - tp_price_suggestion (Optional[float]): Suggestion de prix Take-Profit brut.
            - position_size_pct_suggestion (Optional[float]): Pourcentage du capital (0.0 à 1.0).
                                                              None pour utiliser le défaut de la stratégie.
        """
        pass

    def get_signal(self,
                   data_feed: pd.DataFrame, # Provient de IndicatorCalculator
                   current_position_open: bool,
                   current_position_direction: int,
                   current_entry_price: float,
                   current_equity: float # Équité actuelle pour le calcul de la taille de position
                  ) -> Dict[str, Any]:
        """
        Méthode principale appelée par le simulateur de backtest pour obtenir les décisions de trading.

        Args:
            data_feed (pd.DataFrame): Données de marché avec indicateurs pré-calculés.
            current_position_open (bool): Si une position est ouverte.
            current_position_direction (int): Direction de la position (1 long, -1 short, 0 none).
            current_entry_price (float): Prix d'entrée de la position ouverte.
            current_equity (float): Équité actuelle du portefeuille.

        Returns:
            Dict[str, Any]: Un dictionnaire contenant la décision de trading et les paramètres d'ordre.
        """
        if data_feed.empty:
            logger.warning(f"{self.log_prefix} data_feed vide reçu dans get_signal. Retour de HOLD.")
            return self._get_default_signal_response("Data feed vide.")

        try:
            # _calculate_indicators vérifie/prépare les colonnes _strat
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

        # Déterminer le type d'ordre final et le pourcentage de taille de position
        final_order_type = order_type_pref if order_type_pref else self.get_param("order_type_preference", "MARKET")
        final_pos_size_pct = pos_size_pct_sugg if pos_size_pct_sugg is not None else self.get_param('capital_allocation_pct', 1.0)

        # Préparer les logs théoriques d'ordre (surtout pour OOS)
        entry_order_params_log: Optional[Dict[str, Any]] = None
        oco_params_log: Optional[Dict[str, Any]] = None

        if signal_type in [1, -1] and not current_position_open:
            # Estimer le prix d'entrée pour le calcul de la quantité
            # Utiliser le 'open' de la barre actuelle pour MARKET, ou le limit_sugg pour LIMIT
            # Note: data_ready_for_signals est indexé par timestamp. iloc[-1] est la barre actuelle.
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
                qty_precision=self.quantity_precision, # Doit être défini par set_backtest_context
                symbol_info=self.pair_config or {}, # pair_config est symbol_info
                symbol=self.symbol,
                position_size_pct=final_pos_size_pct
            )

            if theoretical_qty is not None and theoretical_qty > 0:
                qty_str = f"{theoretical_qty:.{self.quantity_precision or 8}f}"
                price_str = f"{limit_sugg:.{self.price_precision or 8}f}" if limit_sugg is not None else None
                
                entry_order_params_log = self._build_entry_params_formatted(
                    side="BUY" if signal_type == 1 else "SELL",
                    quantity_str=qty_str,
                    order_type=final_order_type, # type: ignore
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
                        # stop_limit_price_str peut être égal à sl_price_str pour un OCO STOP_MARKET
                        stop_limit_price_str=sl_str # Supposition pour STOP_MARKET
                    )
            else:
                logger.warning(f"{self.log_prefix} Quantité théorique calculée est nulle ou None ({theoretical_qty}). Pas d'ordre d'entrée.")
                # Si la quantité est nulle, on ne devrait pas entrer. Changer le signal en HOLD.
                signal_type = 0


        # Ajuster les prix SL/TP/Limit avec la précision de l'exchange
        final_sl = None
        if sl_sugg is not None and self.pair_config and self.price_precision is not None:
            tick_size_sl = get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize')
            final_sl = adjust_precision(sl_sugg, self.price_precision, tick_size=tick_size_sl)
        elif sl_sugg is not None:
            final_sl = round(sl_sugg, self.price_precision or 8) # Fallback si pas de tick_size

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
            "order_type": final_order_type,
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
                            qty_precision: Optional[int], # Précision de la quantité (nombre de décimales)
                            symbol_info: Dict[str, Any], # C'est le pair_config
                            symbol: str, # Juste pour le logging
                            position_size_pct: Optional[float] = None
                           ) -> Optional[float]:
        """
        Calcule la quantité d'actifs de base à trader.
        Gère le trading sur marge et les filtres de l'exchange.
        """
        calc_log_prefix = f"{self.log_prefix}[CalcQty]"
        if entry_price <= 1e-9: # Éviter division par zéro ou prix invalide
            logger.error(f"{calc_log_prefix} Prix d'entrée ({entry_price}) invalide pour le calcul de la quantité.")
            return None
        if available_capital <= 1e-9:
            logger.warning(f"{calc_log_prefix} Capital disponible ({available_capital}) nul ou négatif. Aucune quantité ne peut être calculée.")
            return 0.0

        # Déterminer le pourcentage du capital à utiliser
        capital_alloc_pct_strat = self.get_param('capital_allocation_pct', 1.0) # Depuis les params de la strat
        actual_pos_size_pct = position_size_pct if position_size_pct is not None else capital_alloc_pct_strat
        
        if not (isinstance(actual_pos_size_pct, (float, int)) and 0 < actual_pos_size_pct <= 1.0):
            logger.warning(f"{calc_log_prefix} position_size_pct ({actual_pos_size_pct}) invalide. Utilisation de 100% (1.0).")
            actual_pos_size_pct = 1.0
        
        capital_for_this_trade = available_capital * actual_pos_size_pct
        
        # Valeur totale de la position en actif de cotation, en tenant compte du levier
        # Pour le trading sur marge, le capital_for_this_trade est la marge que l'on est prêt à engager.
        # La valeur totale de la position sera capital_for_this_trade * levier.
        total_position_value_quote = capital_for_this_trade * self.leverage
        
        # Quantité brute en actif de base
        quantity_base_raw = total_position_value_quote / entry_price
        logger.debug(f"{calc_log_prefix} Capital: {available_capital:.2f}, Alloc%: {actual_pos_size_pct:.2%}, "
                     f"Levier: {self.leverage}x, Capital pour trade: {capital_for_this_trade:.2f}, "
                     f"Valeur totale pos (quote): {total_position_value_quote:.2f}, "
                     f"Prix entrée: {entry_price:.{self.price_precision or 8}f}, Qty brute (base): {quantity_base_raw:.8f}")

        if qty_precision is None:
            logger.warning(f"{calc_log_prefix} qty_precision non disponible pour {symbol}. "
                           "La quantité ne sera pas ajustée finement à la précision de l'exchange, utilisant round(8).")
            # Un round simple peut ne pas respecter le stepSize.
            # adjust_quantity_to_step_size est plus robuste.
            # Si qty_precision est None, adjust_quantity_to_step_size pourrait avoir un fallback.
        
        # Ajuster la quantité au stepSize et à la précision
        # adjust_quantity_to_step_size gère l'arrondi vers le bas au multiple de step_size.
        adjusted_quantity_base = adjust_quantity_to_step_size(quantity_base_raw, symbol_info, qty_precision)
        
        if adjusted_quantity_base is None or adjusted_quantity_base <= 1e-9: # 1e-9 pour éviter les floats très petits
            logger.warning(f"{calc_log_prefix} Quantité ajustée ({adjusted_quantity_base}) est nulle ou négative après application de step_size/précision. "
                           f"Qty brute: {quantity_base_raw:.8f}")
            return 0.0

        # Vérifier les filtres minQty et minNotional
        min_qty_filter = get_filter_value(symbol_info, 'LOT_SIZE', 'minQty')
        if min_qty_filter is not None and adjusted_quantity_base < min_qty_filter:
            logger.warning(f"{calc_log_prefix} Quantité ajustée {adjusted_quantity_base:.{qty_precision or 8}f} < minQty requis {min_qty_filter:.{qty_precision or 8}f}. Retour de 0.")
            return 0.0
        
        min_notional_filter = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional')
        # Pour les ordres MARKET, le minNotional peut s'appliquer au prix du marché, pas au prix d'entrée estimé.
        # Ici, on utilise entry_price qui est notre meilleure estimation.
        if min_notional_filter is not None and (adjusted_quantity_base * entry_price) < min_notional_filter:
            logger.warning(f"{calc_log_prefix} Valeur notionnelle de l'ordre ({adjusted_quantity_base * entry_price:.2f}) "
                           f"< MIN_NOTIONAL requis ({min_notional_filter:.2f}). Retour de 0.")
            return 0.0
            
        logger.info(f"{calc_log_prefix} Quantité finale calculée pour {symbol}: {adjusted_quantity_base:.{qty_precision or 8}f} {self.base_asset or ''}")
        return adjusted_quantity_base

    def _build_entry_params_formatted(self,
                                      side: str, # "BUY" ou "SELL"
                                      quantity_str: str,
                                      order_type: str, # "MARKET" ou "LIMIT"
                                      entry_price_str: Optional[str] = None, # Requis pour LIMIT
                                      time_in_force: Optional[str] = None, # Ex: "GTC", "IOC", "FOK"
                                      new_client_order_id: Optional[str] = None
                                     ) -> Dict[str, Any]:
        """
        Construit le dictionnaire de paramètres pour un ordre d'entrée,
        en formatant les valeurs et en générant un ID client unique.
        """
        if new_client_order_id is None:
            # Générer un ID client unique et court pour le backtesting/logging
            # Pour le live, l'OrderExecutionClient pourrait avoir sa propre logique de génération d'ID.
            # Format: sim_stratNameShort_symbolShort_timestampShort_uuidShort
            strat_short = self.strategy_name[:min(len(self.strategy_name), 5)].lower()
            sym_short = self.symbol[:min(len(self.symbol),3)].lower()
            ts_short = str(int(pd.Timestamp.now(tz='UTC').timestamp() * 1000))[-6:]
            uuid_short = str(uuid.uuid4().hex)[:4]
            new_client_order_id = f"sim_{strat_short}_{sym_short}_{ts_short}_{uuid_short}"
            new_client_order_id = new_client_order_id[:36] # Limite de Binance pour certains champs

        params: Dict[str, Any] = {
            "symbol": self.symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity_str, # Doit être une chaîne formatée à la bonne précision
            "newClientOrderId": new_client_order_id
        }
        if order_type.upper() == "LIMIT":
            if entry_price_str is None:
                msg = f"Le prix (entry_price_str) doit être spécifié pour un ordre LIMIT. Reçu: {entry_price_str}"
                logger.error(f"{self.log_prefix}[BuildEntryParams] {msg}")
                raise ValueError(msg)
            params["price"] = entry_price_str # Doit être une chaîne formatée
        
        if time_in_force:
            params["timeInForce"] = time_in_force
        
        # Pour le trading sur marge, des paramètres supplémentaires peuvent être nécessaires
        # (ex: sideEffectType pour MARGIN_BUY, AUTO_REPAY, etc.)
        # Cela dépendra de la logique de l'OrderExecutionClient.
        # Pour le backtest, ces détails sont souvent abstraits.
        if self.account_type == "MARGIN" or self.account_type == "ISOLATED_MARGIN":
            # Par défaut, pour un ordre d'achat sur marge, on emprunte. Pour une vente, on emprunte l'actif de base.
            # Le simulateur gère cela implicitement. Pour le live, ce serait explicite.
            # params["sideEffectType"] = "MARGIN_BUY" # ou "AUTO_REPAY", "AUTO_BORROW_REPAY"
            pass


        logger.debug(f"{self.log_prefix}[BuildEntryParams] Paramètres d'ordre d'entrée formatés : {params}")
        return params

    def _build_oco_params_formatted(self,
                                    entry_side: str, # "BUY" ou "SELL" (le side de l'ordre d'entrée)
                                    quantity_str: str, # Quantité pour les ordres SL/TP
                                    sl_price_str: str, # Prix de déclenchement du Stop-Loss
                                    tp_price_str: str, # Prix limite du Take-Profit
                                    stop_limit_price_str: Optional[str] = None, # Prix limite pour l'ordre STOP_LOSS_LIMIT
                                    stop_limit_time_in_force: Optional[str] = "GTC",
                                    list_client_order_id: Optional[str] = None
                                   ) -> Dict[str, Any]:
        """
        Construit le dictionnaire de paramètres pour un ordre OCO (One-Cancels-the-Other),
        utilisé pour placer un Stop-Loss et un Take-Profit simultanément.
        """
        # Le côté des ordres SL/TP est l'inverse du côté de l'ordre d'entrée
        exit_side = "SELL" if entry_side.upper() == "BUY" else "BUY"
        
        if list_client_order_id is None:
            strat_short = self.strategy_name[:min(len(self.strategy_name), 4)].lower()
            sym_short = self.symbol[:min(len(self.symbol),3)].lower()
            ts_short = str(int(pd.Timestamp.now(tz='UTC').timestamp() * 1000))[-5:]
            uuid_short = str(uuid.uuid4().hex)[:3]
            list_client_order_id = f"oco_{strat_short}_{sym_short}_{ts_short}_{uuid_short}"
            list_client_order_id = list_client_order_id[:32] # Limite Binance pour listClientOrderId

        oco_params: Dict[str, Any] = {
            "symbol": self.symbol,
            "side": exit_side, # Côté pour les ordres de sortie (SL et TP)
            "quantity": quantity_str, # Quantité à vendre/acheter pour sortir
            
            # Paramètres pour l'ordre Take-Profit (qui est un LIMIT ou LIMIT_MAKER)
            "price": tp_price_str,  # Prix limite pour le Take-Profit
            
            # Paramètres pour l'ordre Stop-Loss
            "stopPrice": sl_price_str, # Prix de déclenchement pour le Stop-Loss
            
            # Optionnel: si l'ordre Stop-Loss est un STOP_LOSS_LIMIT (plutôt que STOP_MARKET)
            # `stopLimitPrice` est le prix auquel l'ordre LIMIT est placé une fois `stopPrice` atteint.
            # Si `stopLimitPrice` n'est pas fourni, l'ordre stop est un STOP_MARKET.
            # Pour un STOP_MARKET simulé, on peut omettre stopLimitPrice ou le mettre égal à stopPrice.
            # Pour un vrai STOP_LOSS_LIMIT, il faut une stratégie pour définir ce prix (ex: stopPrice - un petit delta).
            "stopLimitPrice": stop_limit_price_str if stop_limit_price_str else sl_price_str, # Pour simuler STOP_MARKET, on peut le mettre égal au stopPrice
                                                                                             # ou le laisser None si l'API le gère.
                                                                                             # Binance API: stopLimitPrice est requis si stopLimitTimeInForce est envoyé.
            
            "listClientOrderId": list_client_order_id
        }
        
        # stopLimitTimeInForce est requis si stopLimitPrice est envoyé pour Binance
        if oco_params.get("stopLimitPrice") and stop_limit_time_in_force:
            oco_params["stopLimitTimeInForce"] = stop_limit_time_in_force
        elif oco_params.get("stopLimitPrice") and not stop_limit_time_in_force:
            # Si stopLimitPrice est là mais pas de timeInForce, Binance met GTC par défaut pour STOP_LOSS_LIMIT
            # Mais il est bon d'être explicite si on a un stopLimitPrice.
            oco_params["stopLimitTimeInForce"] = "GTC"


        logger.debug(f"{self.log_prefix}[BuildOCOParams] Paramètres d'ordre OCO formatés : {oco_params}")
        return oco_params

    @abstractmethod
    def get_required_indicator_configs(self) -> List[Dict[str, Any]]:
        """
        Déclare les indicateurs requis par la stratégie et leurs configurations.

        Cette méthode doit être implémentée par les classes filles.
        Elle retourne une liste de dictionnaires, chaque dictionnaire décrivant
        un indicateur nécessaire. Exemple de format pour un dictionnaire :
        {
            'indicator_name': 'EMA', # Nom de l'indicateur (ex: 'EMA', 'RSI', 'BBANDS')
            'params': {'length': self.params['ema_period']}, # Paramètres de l'indicateur
            'source_column': 'close', # Colonne source (ex: 'open', 'high', 'low', 'close', 'volume')
            # Fréquence des klines source pour cet indicateur, tirée des hyperparamètres
            'source_kline_frequency_param': self.params.get('indicateur_frequence_ema_rapide'),
            # Suffixe pour la colonne de sortie (ex: 'EMA_FAST_strat')
            'output_column_suffix': 'FAST_strat'
        }
        Ou pour des indicateurs plus complexes retournant un DataFrame (comme BBANDS):
        {
            'indicator_name': 'BBANDS',
            'params': {'length': self.params['bb_period'], 'std': self.params['bb_std']},
            'source_column': 'close',
            'source_kline_frequency_param': self.params.get('indicateur_frequence_bbands'),
            'output_column_map': { # Map des colonnes du DataFrame de l'indicateur aux suffixes _strat
                'BBL_length_std': 'LOWER_strat', # ex: BBL_20_2 -> BB_LOWER_strat
                'BBM_length_std': 'MIDDLE_strat',
                'BBU_length_std': 'UPPER_strat',
                'BBB_length_std': 'BANDWIDTH_strat',
                # 'BBP_length_std': 'PERCENT_strat' # Non utilisé, exemple
            }
        }

        Returns:
            List[Dict[str, Any]]: Une liste de configurations d'indicateurs.
                                  Retourne une liste vide par défaut.
        """
        return []


    @abstractmethod
    def generate_order_request(self,
                               data: pd.DataFrame, # Données avec indicateurs _strat déjà calculés
                               current_position: int, # 0: pas de pos, 1: long, -1: short
                               available_capital: float, # En actif de cotation (ex: USDC)
                               symbol_info: Dict[str, Any] # Infos de l'exchange pour la paire
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Génère une requête d'ordre d'ENTRÉE pour le trading en direct ou une simulation détaillée.
        Ne doit générer un ordre que si current_position est 0 (pas de position ouverte).

        Args:
            data (pd.DataFrame): DataFrame avec les indicateurs _strat prêts.
            current_position (int): État actuel de la position.
            available_capital (float): Capital disponible.
            symbol_info (Dict[str, Any]): Informations de l'exchange pour la paire.

        Returns:
            Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
                - Un tuple contenant :
                    - entry_order_params (Dict[str, Any]): Paramètres formatés pour l'ordre d'entrée.
                    - sl_tp_raw_prices_dict (Dict[str, float]): Dictionnaire avec {'sl_price': float, 'tp_price': float} bruts.
                - Ou None si aucun ordre d'entrée n'est généré.
        """
        pass

