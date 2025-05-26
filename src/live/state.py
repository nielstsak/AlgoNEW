# src/live/state.py
"""
Ce module définit la classe LiveTradingState, responsable de la gestion
(chargement, sauvegarde, mise à jour, transition) de l'état d'une instance
de trading en direct pour une paire de trading spécifique.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, Union, cast
from datetime import datetime, timezone

# Utilisation de file_utils pour la sauvegarde JSON si disponible, sinon fallback.
try:
    from src.utils.file_utils import ensure_dir_exists, save_json, load_json
except ImportError:
    logger_bootstrap = logging.getLogger(__name__ + "_bootstrap_state")
    logger_bootstrap.warning("src.utils.file_utils non trouvé. Utilisation de fallbacks pour ensure_dir_exists, save_json, load_json.")
    def ensure_dir_exists(path: Path) -> bool: # type: ignore
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
    def save_json(filepath: Path, data: Any, indent: int = 4, default_serializer=str) -> bool: # type: ignore
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, default=default_serializer)
            return True
        except Exception:
            return False
    def load_json(filepath: Path) -> Optional[Any]: # type: ignore
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

logger = logging.getLogger(__name__)

# --- Constantes de Statut de Trading ---
STATUT_1_NO_TRADE_NO_OCO = "STATUT_1_NO_TRADE_NO_OCO"
"""Aucun trade en cours, aucun ordre OCO actif ou en attente."""

STATUT_2_ENTRY_FILLED_OCO_PENDING = "STATUT_2_ENTRY_FILLED_OCO_PENDING"
"""Ordre d'entrée exécuté, en attente de placement de l'ordre OCO (Stop-Loss/Take-Profit)."""

STATUT_3_OCO_ACTIVE = "STATUT_3_OCO_ACTIVE"
"""Ordre OCO (Stop-Loss/Take-Profit) actif sur le marché pour la position ouverte."""

# STATUT_4_ERROR_REQUIRES_MANUAL_INTERVENTION = "STATUT_4_ERROR_REQUIRES_MANUAL_INTERVENTION"
# """Un statut d'erreur indiquant qu'une intervention manuelle pourrait être nécessaire."""

# --- Constantes d'Actifs (exemple) ---
# Utilisé pour identifier l'actif de commission ou de cotation principal.
# Devrait être aligné avec la configuration de l'application ou de la paire.
DEFAULT_QUOTE_ASSET_FOR_COMMISSION = "USDC"


class LiveTradingState:
    """
    Gère l'état d'une instance de trading en direct pour une paire spécifique.
    Inclut le chargement, la sauvegarde, la mise à jour et les transitions d'état.
    """
    def __init__(self,
                 pair_symbol: str,
                 state_file_path: Union[str, Path],
                 quote_asset_for_commission: str = DEFAULT_QUOTE_ASSET_FOR_COMMISSION):
        """
        Initialise le gestionnaire d'état.

        Args:
            pair_symbol (str): Le symbole de la paire de trading (ex: "BTCUSDT").
            state_file_path (Union[str, Path]): Chemin vers le fichier JSON de l'état.
            quote_asset_for_commission (str): L'actif de cotation utilisé pour calculer
                                              les commissions (ex: "USDC", "USDT").
        """
        self.pair_symbol: str = pair_symbol.upper()
        self.state_file_path: Path = Path(state_file_path)
        self.quote_asset_for_commission: str = quote_asset_for_commission.upper()
        self.log_prefix: str = f"[StateMgr][{self.pair_symbol}]"

        self.state: Dict[str, Any] = self._load_state()

        # Validation initiale après chargement ou création par défaut
        # S'assurer que l'état est pour la bonne paire et a un statut.
        # _load_state gère déjà la création par défaut si le fichier n'existe pas ou est invalide.
        # Cette vérification est une sécurité supplémentaire.
        if self.state.get("pair_symbol") != self.pair_symbol or not self.state.get("current_status"):
            logger.warning(
                f"{self.log_prefix} État initial chargé invalide ou pour une paire différente. "
                f"Paire dans état: {self.state.get('pair_symbol')}, Statut: {self.state.get('current_status')}. "
                "Réinitialisation forcée avec les valeurs par défaut."
            )
            self.state = self._default_state()
            self._save_state() # Sauvegarder l'état par défaut fraîchement initialisé

        logger.info(f"{self.log_prefix} LiveTradingState initialisé. Statut actuel : {self.get_current_status_name()}. "
                    f"Fichier d'état : {self.state_file_path}")

    def _default_state(self) -> Dict[str, Any]:
        """
        Retourne la structure et les valeurs par défaut pour un nouvel état de trading.
        Tous les timestamps sont en millisecondes UTC.
        """
        current_time_ms = int(time.time() * 1000)
        return {
            "pair_symbol": self.pair_symbol,
            "current_status": STATUT_1_NO_TRADE_NO_OCO,
            "current_trade_cycle_id": None, # ID unique pour un cycle complet entrée->sortie
            "last_status_update_timestamp_ms": current_time_ms,
            "last_error_message": None, # Dernier message d'erreur rencontré
            "last_error_timestamp_ms": None,
            "available_capital_at_last_check": 0.0, # Capital en actif de cotation (ex: USDC)

            # Détails de l'Ordre d'Entrée en Attente de Placement ou d'Exécution
            "pending_entry_order_id_api": None, # ID de l'ordre retourné par l'exchange
            "pending_entry_client_order_id": None, # ID client de l'ordre d'entrée
            "pending_entry_params_sent": {}, # Paramètres envoyés à l'API pour l'entrée
            "pending_sl_tp_raw_prices": {}, # Prix SL/TP bruts { "sl_price": float, "tp_price": float }

            # Détails de la Position Ouverte (après exécution de l'ordre d'entrée)
            "entry_order_details_api": {}, # Réponse complète de l'API pour l'ordre d'entrée rempli
            "position_side": None, # "BUY" (long) ou "SELL" (short)
            "position_quantity_base": 0.0, # Quantité en actif de base
            "position_entry_price_avg": 0.0, # Prix moyen d'exécution de l'entrée
            "position_entry_timestamp_ms": None, # Timestamp d'exécution de l'entrée
            "position_entry_commission_quote": 0.0, # Commission pour l'entrée (en actif de cotation)

            # Détails du Prêt sur Marge (si applicable)
            "margin_loan_details": {"asset": None, "amount": 0.0, "timestamp_ms": None},

            # Détails de l'Ordre OCO en Attente de Placement
            "oco_params_to_place_api": {}, # Paramètres construits pour l'API OCO
            "pending_oco_list_client_order_id": None, # Client Order ID pour la liste OCO

            # Détails de l'Ordre OCO Actif (après placement réussi)
            "active_oco_details_api": {}, # Réponse complète de l'API pour l'OCO placé
            "active_oco_order_list_id_api": None, # OrderListId de l'exchange pour l'OCO
            "active_sl_order_id_api": None,
            "active_tp_order_id_api": None,
            "active_sl_price_set": None, # Prix SL effectif de l'OCO
            "active_tp_price_set": None, # Prix TP effectif de l'OCO
            "oco_active_timestamp_ms": None,

            # Informations sur le Dernier Trade Clôturé
            "last_closed_trade_info": {}, # Résumé du dernier cycle de trade complet

            # Suivi de la synchronisation avec l'exchange
            "last_successful_sync_timestamp_ms": None,
            
            # Informations spécifiques à l'instance (optionnel, pour référence)
            "instance_context_label": None, # Ex: "5min_rsi_filter"
            "instance_account_alias": None  # Ex: "binance_margin_live"
        }

    def _load_state(self) -> Dict[str, Any]:
        """
        Charge l'état depuis le fichier JSON.
        Si le fichier n'existe pas, est corrompu, ou est pour une autre paire,
        retourne un état par défaut. Complète les champs manquants avec les
        valeurs par défaut si un état existant est chargé (pour la rétrocompatibilité).
        """
        default_s = self._default_state()
        if self.state_file_path.exists() and self.state_file_path.is_file() and self.state_file_path.stat().st_size > 0:
            try:
                # Utiliser load_json de file_utils si disponible
                loaded_data = load_json(self.state_file_path) if callable(globals().get('load_json')) else None
                if loaded_data is None and not callable(globals().get('load_json')): # Fallback si load_json n'est pas importé
                    with open(self.state_file_path, 'r', encoding='utf-8') as f:
                        loaded_data = json.load(f)
                
                loaded_state = cast(Dict[str, Any], loaded_data)

                if not isinstance(loaded_state, dict):
                    logger.warning(f"{self.log_prefix} Fichier d'état {self.state_file_path} ne contient pas un objet JSON valide. Réinitialisation.")
                    return default_s

                if loaded_state.get("pair_symbol") == self.pair_symbol and loaded_state.get("current_status"):
                    logger.info(f"{self.log_prefix} État chargé depuis {self.state_file_path}")
                    
                    # Fusionner avec l'état par défaut pour ajouter les nouveaux champs manquants
                    updated_during_load = False
                    final_state = default_s.copy() # Commencer avec une base de défauts
                    final_state.update(loaded_state) # Les valeurs chargées écrasent les défauts

                    # Vérifier si des clés du template par défaut manquaient dans final_state après update
                    # (ce qui signifie qu'elles n'étaient ni dans loaded_state ni dans default_s, ce qui est étrange)
                    # ou plutôt, vérifier si des clés de default_s ne sont pas dans loaded_state
                    # et les ajouter à loaded_state (ce que final_state.update(loaded_state) fait déjà si final_state=default_s.copy())
                    for key_default in default_s:
                        if key_default not in loaded_state:
                            logger.info(f"{self.log_prefix} Champ manquant '{key_default}' dans l'état chargé. Ajout de la valeur par défaut.")
                            # final_state[key_default] est déjà la valeur par défaut.
                            updated_during_load = True # Marquer pour une sauvegarde potentielle si on le souhaite ici

                    if updated_during_load:
                        logger.info(f"{self.log_prefix} L'état chargé a été complété avec des champs par défaut.")
                        # Optionnel : sauvegarder immédiatement si des champs ont été ajoutés.
                        # Pour l'instant, la sauvegarde se fera lors de la prochaine modification explicite.
                    return final_state
                else:
                    logger.warning(f"{self.log_prefix} Fichier d'état {self.state_file_path} pour une paire différente "
                                   f"(attendu: {self.pair_symbol}, trouvé: {loaded_state.get('pair_symbol')}) "
                                   "ou statut manquant. Réinitialisation.")
            except json.JSONDecodeError:
                logger.error(f"{self.log_prefix} Erreur de décodage JSON du fichier d'état {self.state_file_path}. Réinitialisation.", exc_info=True)
            except Exception as e: # pylint: disable=broad-except
                logger.error(f"{self.log_prefix} Erreur inattendue lors du chargement de l'état depuis {self.state_file_path}: {e}. Réinitialisation.", exc_info=True)
        else:
            logger.info(f"{self.log_prefix} Fichier d'état non trouvé à {self.state_file_path} ou vide. Initialisation avec l'état par défaut.")
        return default_s

    def _save_state(self) -> None:
        """Sauvegarde l'état actuel (self.state) dans le fichier JSON."""
        try:
            # Utiliser save_json de file_utils si disponible
            if callable(globals().get('save_json')):
                if not save_json(self.state_file_path, self.state, indent=4, default_serializer=str):
                    raise IOError("save_json a retourné False")
            else: # Fallback
                if not ensure_dir_exists(self.state_file_path.parent): # type: ignore
                     raise IOError(f"Impossible de créer le répertoire parent {self.state_file_path.parent}")
                with open(self.state_file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.state, f, indent=4, default=str)
            logger.debug(f"{self.log_prefix} État sauvegardé dans {self.state_file_path}")
        except IOError as e_io:
            logger.error(f"{self.log_prefix} Erreur d'IO lors de la sauvegarde de l'état dans {self.state_file_path}: {e_io}", exc_info=True)
        except Exception as e_gen: # pylint: disable=broad-except
            logger.error(f"{self.log_prefix} Erreur inattendue lors de la sauvegarde de l'état : {e_gen}", exc_info=True)

    def update_specific_fields(self, update_dict: Dict[str, Any]) -> None:
        """
        Met à jour des champs spécifiques dans l'état, met à jour le timestamp
        de dernière modification, et sauvegarde l'état.

        Args:
            update_dict (Dict[str, Any]): Dictionnaire des champs à mettre à jour.
        """
        if not isinstance(update_dict, dict):
            logger.error(f"{self.log_prefix} update_dict doit être un dictionnaire. Reçu : {type(update_dict)}")
            return

        logger.debug(f"{self.log_prefix} Mise à jour des champs d'état : {list(update_dict.keys())}")
        for key, value in update_dict.items():
            self.state[key] = value
        
        self.state["last_status_update_timestamp_ms"] = int(time.time() * 1000)
        self._save_state()
        logger.info(f"{self.log_prefix} Champs d'état mis à jour. Statut actuel : {self.get_current_status_name()}")

    def get_current_status(self) -> str:
        """Retourne le statut de trading actuel (chaîne complète)."""
        return self.state.get("current_status", STATUT_1_NO_TRADE_NO_OCO)

    def get_current_status_name(self) -> str:
        """Retourne le nom lisible du statut actuel (sans le préfixe "STATUT_")."""
        status_raw = self.state.get("current_status", "UNKNOWN_STATUS")
        return status_raw.replace("STATUT_", "")

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Retourne une copie de l'état actuel pour consultation."""
        return self.state.copy()

    def set_last_error(self, error_message: Optional[str]) -> None:
        """Enregistre un message d'erreur dans l'état."""
        timestamp_ms = int(time.time() * 1000) if error_message else None
        self.update_specific_fields({
            "last_error_message": error_message,
            "last_error_timestamp_ms": timestamp_ms
        })
        if error_message:
            logger.error(f"{self.log_prefix} Erreur enregistrée dans l'état : {error_message}")

    def clear_last_error(self) -> None:
        """Efface le dernier message d'erreur de l'état."""
        if self.state.get("last_error_message"):
            self.update_specific_fields({
                "last_error_message": None,
                "last_error_timestamp_ms": None
            })
            logger.info(f"{self.log_prefix} Erreur précédente effacée de l'état.")
            
    def update_last_successful_sync_timestamp(self) -> None:
        """Met à jour le timestamp de la dernière synchronisation réussie avec l'exchange."""
        self.update_specific_fields({"last_successful_sync_timestamp_ms": int(time.time() * 1000)})

    # --- Méthodes de Transition d'État ---

    def transition_to_status_1(self,
                               exit_reason: str = "REINITIALIZATION",
                               closed_trade_details_for_log: Optional[Dict[str, Any]] = None) -> None:
        """
        Transitionne vers STATUT_1_NO_TRADE_NO_OCO.
        Réinitialise la plupart des champs de l'état aux valeurs par défaut,
        tout en préservant certaines informations comme le capital disponible et
        les informations sur le dernier trade clôturé (sauf si écrasées).

        Args:
            exit_reason (str): La raison de la transition vers le statut 1.
            closed_trade_details_for_log (Optional[Dict[str, Any]]): Si la transition
                est due à la clôture d'un trade, ces détails seront utilisés pour mettre
                à jour `last_closed_trade_info`.
        """
        current_cycle_id = self.state.get("current_trade_cycle_id")
        logger.info(f"{self.log_prefix}[Cycle:{current_cycle_id or 'N/A'}] Transition vers STATUT_1_NO_TRADE_NO_OCO. Raison : {exit_reason}.")
        
        # Préserver certaines informations importantes avant la réinitialisation complète
        preserved_capital = self.state.get("available_capital_at_last_check", 0.0)
        # Si de nouveaux détails de clôture sont fournis, ils mettent à jour `last_closed_trade_info`.
        # Sinon, on conserve l'ancien `last_closed_trade_info`.
        last_trade_info_to_keep = self.state.get("last_closed_trade_info", {})
        if closed_trade_details_for_log: # Si on passe les détails d'un trade qui vient de se terminer
            last_trade_info_to_keep = closed_trade_details_for_log # Utiliser les nouveaux détails

        # Réinitialiser à l'état par défaut
        new_state_fields = self._default_state()
        
        # Restaurer les informations préservées/mises à jour
        new_state_fields["available_capital_at_last_check"] = preserved_capital
        new_state_fields["last_closed_trade_info"] = last_trade_info_to_keep
        # `current_trade_cycle_id` est déjà None dans `_default_state`.
        # `last_error_message` et `last_error_timestamp_ms` sont aussi None par défaut.

        self.state.update(new_state_fields) # Appliquer tous les champs par défaut et préservés
        self.state["last_status_update_timestamp_ms"] = int(time.time() * 1000)
        # Effacer les erreurs précédentes lors d'une réinitialisation vers un état "propre"
        if self.state.get("last_error_message"):
             self.state["last_error_message"] = None
             self.state["last_error_timestamp_ms"] = None
        
        self._save_state()
        logger.info(f"{self.log_prefix} Transition vers STATUT_1 terminée.")

    def prepare_for_entry_order(self,
                                entry_params_sent: Dict[str, Any],
                                sl_tp_raw_prices: Dict[str, float],
                                trade_cycle_id: str) -> None:
        """
        Prépare l'état pour le placement d'un ordre d'entrée.
        Met à jour les informations relatives à l'ordre d'entrée en attente.

        Args:
            entry_params_sent (Dict[str, Any]): Paramètres de l'ordre envoyés à l'API.
            sl_tp_raw_prices (Dict[str, float]): Prix SL/TP bruts (non ajustés) suggérés.
                                                 Format : `{"sl_price": float, "tp_price": float}`.
            trade_cycle_id (str): ID unique pour ce nouveau cycle de trade.
        """
        logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id}] Préparation pour l'ordre d'entrée. "
                    f"Params: {entry_params_sent.get('side')} {entry_params_sent.get('quantity')} @ {entry_params_sent.get('price', 'MARKET')}, "
                    f"SL/TP bruts: {sl_tp_raw_prices}")
        update_dict = {
            "current_status": STATUT_1_NO_TRADE_NO_OCO, # Reste en statut 1 jusqu'à confirmation du placement
            "pending_entry_params_sent": entry_params_sent,
            "pending_sl_tp_raw_prices": sl_tp_raw_prices,
            "current_trade_cycle_id": trade_cycle_id,
            "pending_entry_order_id_api": None, # Réinitialiser au cas où
            "pending_entry_client_order_id": entry_params_sent.get("newClientOrderId"), # Capturer le client ID envoyé
            "last_error_message": None, # Effacer les erreurs précédentes avant une nouvelle tentative
            "last_error_timestamp_ms": None
        }
        self.update_specific_fields(update_dict)

    def record_placed_entry_order(self, order_id_api: Union[str, int], client_order_id: str) -> None:
        """
        Enregistre les IDs d'un ordre d'entrée qui vient d'être placé avec succès sur l'exchange.

        Args:
            order_id_api (Union[str, int]): ID de l'ordre retourné par l'exchange.
            client_order_id (str): ID client de l'ordre (celui envoyé).
        """
        trade_cycle_id = self.state.get("current_trade_cycle_id", "UNKNOWN_CYCLE")
        logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id}] Ordre d'entrée placé sur l'exchange. "
                    f"ExchangeID: {order_id_api}, ClientID: {client_order_id}")
        self.update_specific_fields({
            "pending_entry_order_id_api": str(order_id_api), # S'assurer que c'est une chaîne
            "pending_entry_client_order_id": client_order_id # Devrait déjà être là mais confirmer
        })

    def transition_to_status_2(self,
                               filled_entry_details_api: Dict[str, Any],
                               margin_loan_info: Dict[str, Any]) -> None:
        """
        Transitionne vers STATUT_2_ENTRY_FILLED_OCO_PENDING après qu'un ordre d'entrée
        a été confirmé comme rempli (FILLED).

        Args:
            filled_entry_details_api (Dict[str, Any]): Réponse complète de l'API pour
                l'ordre d'entrée rempli (doit contenir executedQty, cummulativeQuoteQty, side, etc.).
            margin_loan_info (Dict[str, Any]): Informations sur le prêt sur marge contracté
                pour cette entrée (ex: `{"asset": "USDT", "amount": 1000.0, "timestamp_ms": ...}`).
        """
        trade_cycle_id = self.state.get("current_trade_cycle_id", "UNKNOWN_CYCLE")
        entry_order_id = filled_entry_details_api.get("orderId", "N/A")
        logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id}] Transition vers STATUT_2_ENTRY_FILLED_OCO_PENDING. "
                    f"Ordre d'entrée {entry_order_id} rempli. Prêt : {margin_loan_info.get('asset')} {margin_loan_info.get('amount')}")
        
        executed_qty_base = float(filled_entry_details_api.get("executedQty", 0.0))
        cummulative_quote_qty = float(filled_entry_details_api.get("cummulativeQuoteQty", 0.0))
        
        entry_price_avg_calc = 0.0
        if executed_qty_base > 1e-9: # Éviter division par zéro pour les quantités très petites
            entry_price_avg_calc = cummulative_quote_qty / executed_qty_base
        
        commission_total_quote = 0.0
        # Calculer la commission totale en se basant sur les 'fills' si disponibles
        if 'fills' in filled_entry_details_api and isinstance(filled_entry_details_api['fills'], list):
            for fill in filled_entry_details_api['fills']:
                if isinstance(fill, dict) and fill.get('commissionAsset', '').upper() == self.quote_asset_for_commission:
                    commission_total_quote += float(fill.get('commission', 0.0))
        # Fallback si 'fills' n'est pas là ou si la commission est au niveau de l'ordre principal
        elif filled_entry_details_api.get('commissionAsset', '').upper() == self.quote_asset_for_commission:
             commission_total_quote = float(filled_entry_details_api.get('commission', 0.0))

        update_dict = {
            "current_status": STATUT_2_ENTRY_FILLED_OCO_PENDING,
            "pending_entry_order_id_api": None, # L'ordre d'entrée n'est plus en attente
            "pending_entry_client_order_id": None,
            "pending_entry_params_sent": {}, # Effacer les params de l'entrée en attente
            # pending_sl_tp_raw_prices est conservé car nécessaire pour construire l'OCO

            "entry_order_details_api": filled_entry_details_api,
            "position_side": filled_entry_details_api.get("side"), # "BUY" ou "SELL"
            "position_quantity_base": executed_qty_base,
            "position_entry_price_avg": entry_price_avg_calc,
            "position_entry_timestamp_ms": filled_entry_details_api.get("transactTime") or \
                                         filled_entry_details_api.get("updateTime") or \
                                         int(time.time() * 1000), # Fallback
            "position_entry_commission_quote": commission_total_quote,
            
            "margin_loan_details": margin_loan_info,

            # Réinitialiser les champs OCO en préparation du placement
            "oco_params_to_place_api": {},
            "pending_oco_list_client_order_id": None,
            # "pending_oco_order_list_id_api": None, # Ce champ n'est pas dans _default_state, il est optionnel
            "active_oco_details_api": {},
            "active_oco_order_list_id_api": None,
            "active_sl_order_id_api": None,
            "active_tp_order_id_api": None,
            "active_sl_price_set": None,
            "active_tp_price_set": None,
            "oco_active_timestamp_ms": None,
            "last_error_message": None, # Effacer les erreurs précédentes
            "last_error_timestamp_ms": None
        }
        self.update_specific_fields(update_dict)

    def prepare_for_oco_order(self,
                              oco_params_api_to_send: Dict[str, Any],
                              list_client_order_id_for_oco: str) -> None:
        """
        Prépare l'état pour le placement d'un ordre OCO.
        Met à jour les informations relatives à l'ordre OCO en attente de placement.

        Args:
            oco_params_api_to_send (Dict[str, Any]): Paramètres complets à envoyer à l'API pour l'OCO.
            list_client_order_id_for_oco (str): L'ID client qui sera utilisé pour la liste OCO.
        """
        trade_cycle_id = self.state.get("current_trade_cycle_id", "UNKNOWN_CYCLE")
        logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id}] Préparation pour l'ordre OCO. "
                    f"ListClientOrderID: {list_client_order_id_for_oco}")
        update_dict = {
            # Le statut reste STATUT_2 jusqu'à confirmation du placement de l'OCO
            "oco_params_to_place_api": oco_params_api_to_send,
            "pending_oco_list_client_order_id": list_client_order_id_for_oco,
            
            # Réinitialiser les champs OCO actifs car on s'apprête à en placer un nouveau
            "active_oco_details_api": {},
            "active_oco_order_list_id_api": None, # Sera rempli si l'API le retourne avant confirmation
            "active_sl_order_id_api": None,
            "active_tp_order_id_api": None,
            "active_sl_price_set": None,
            "active_tp_price_set": None,
            "oco_active_timestamp_ms": None,
            "last_error_message": None, # Effacer les erreurs précédentes
            "last_error_timestamp_ms": None
        }
        self.update_specific_fields(update_dict)

    def transition_to_status_3(self, active_oco_api_response: Dict[str, Any]) -> None:
        """
        Transitionne vers STATUT_3_OCO_ACTIVE après qu'un ordre OCO a été placé
        et confirmé comme actif sur l'exchange.

        Args:
            active_oco_api_response (Dict[str, Any]): Réponse complète de l'API pour
                l'OCO placé (typiquement la réponse de `POST /sapi/v1/margin/oco` ou
                le résultat d'une requête de statut confirmant l'OCO).
                Doit contenir `orderListId` et la liste des `orderReports`.
        """
        trade_cycle_id = self.state.get("current_trade_cycle_id", "UNKNOWN_CYCLE")
        api_order_list_id = active_oco_api_response.get('orderListId')
        logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id}] Transition vers STATUT_3_OCO_ACTIVE. "
                    f"OrderListID API: {api_order_list_id}")
        
        # Utiliser le listClientOrderId de la réponse s'il existe, sinon celui en attente
        list_client_id_confirmed = active_oco_api_response.get("listClientOrderId", self.state.get("pending_oco_list_client_order_id"))
        
        sl_order_id_api, tp_order_id_api = None, None
        sl_price_effective, tp_price_effective = None, None
        
        # La réponse API pour un OCO contient une liste 'orders' ou 'orderReports'
        order_reports = active_oco_api_response.get("orderReports", active_oco_api_response.get("orders", []))

        if isinstance(order_reports, list):
            for report in order_reports:
                if not isinstance(report, dict): continue
                order_type_api = report.get("type", "").upper()
                # Le Stop est typiquement STOP_LOSS_LIMIT ou STOP_LOSS (qui devient MARKET)
                if "STOP_LOSS" in order_type_api or "STOP_MARKET" in order_type_api:
                    sl_order_id_api = str(report.get("orderId"))
                    sl_price_effective = float(report.get("stopPrice", 0.0)) # stopPrice est le prix de déclenchement
                # Le Take Profit est typiquement LIMIT_MAKER ou LIMIT
                elif "LIMIT" in order_type_api or "TAKE_PROFIT" in order_type_api : # LIMIT_MAKER pour le TP
                    tp_order_id_api = str(report.get("orderId"))
                    tp_price_effective = float(report.get("price", 0.0)) # price est le prix limite du TP
        
        update_dict = {
            "current_status": STATUT_3_OCO_ACTIVE,
            "oco_params_to_place_api": {}, # Effacer car l'OCO est maintenant placé
            "pending_oco_list_client_order_id": None, # Effacer car confirmé

            "active_oco_details_api": active_oco_api_response,
            "active_oco_list_client_order_id": list_client_id_confirmed,
            "active_oco_order_list_id_api": str(api_order_list_id) if api_order_list_id is not None else None,
            "active_sl_order_id_api": sl_order_id_api,
            "active_tp_order_id_api": tp_order_id_api,
            "active_sl_price_set": sl_price_effective,
            "active_tp_price_set": tp_price_effective,
            "oco_active_timestamp_ms": active_oco_api_response.get("transactionTime") or int(time.time() * 1000),
            "last_error_message": None, # Effacer les erreurs précédentes
            "last_error_timestamp_ms": None
        }
        self.update_specific_fields(update_dict)

    def record_closed_trade(self,
                            exit_reason: str,
                            exit_price_avg: Optional[float],
                            # pnl_usdc_estimate: Optional[float], # Le PnL sera recalculé ici
                            closed_order_details_api: Optional[Dict[str, Any]],
                            exit_commission_quote: Optional[float] = None
                           ) -> None:
        """
        Enregistre les informations complètes d'un trade qui vient d'être clôturé
        dans `self.state["last_closed_trade_info"]`.
        Cette méthode est typiquement appelée avant de transitionner vers STATUT_1.

        Args:
            exit_reason (str): Raison de la clôture (ex: "SL_FILLED", "TP_FILLED", "MANUAL_CLOSE").
            exit_price_avg (Optional[float]): Prix moyen d'exécution de la sortie.
            closed_order_details_api (Optional[Dict[str, Any]]): Réponse complète de l'API
                pour l'ordre de clôture (SL, TP, ou ordre manuel).
            exit_commission_quote (Optional[float]): Commission de sortie explicite si connue,
                sinon elle sera estimée à partir de closed_order_details_api.
        """
        trade_cycle_id_closed = self.state.get("current_trade_cycle_id", f"ORPHAN_CLOSURE_{int(time.time()*1000)}")
        logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id_closed}] Enregistrement du trade clôturé. "
                    f"Raison: {exit_reason}, Prix Sortie Avg: {exit_price_avg}")
        
        entry_details_snapshot = self.state.get("entry_order_details_api", {})
        entry_commission_quote_recorded = self.state.get("position_entry_commission_quote", 0.0)
        
        # Déterminer la commission de sortie
        actual_exit_commission_quote = 0.0
        if exit_commission_quote is not None:
            actual_exit_commission_quote = exit_commission_quote
        elif closed_order_details_api and isinstance(closed_order_details_api.get('fills'), list):
            for fill in closed_order_details_api['fills']:
                if isinstance(fill, dict) and fill.get('commissionAsset', '').upper() == self.quote_asset_for_commission:
                    actual_exit_commission_quote += float(fill.get('commission', 0.0))
        elif closed_order_details_api and closed_order_details_api.get('commissionAsset', '').upper() == self.quote_asset_for_commission:
            actual_exit_commission_quote = float(closed_order_details_api.get('commission', 0.0))

        # Calcul du PnL Net final
        pnl_net_final_quote: Optional[float] = None
        pnl_gross_final_quote: Optional[float] = None
        
        entry_price_recorded = self.state.get("position_entry_price_avg")
        qty_base_recorded = self.state.get("position_quantity_base")
        side_recorded = self.state.get("position_side")

        if exit_price_avg is not None and \
           isinstance(entry_price_recorded, (int, float)) and \
           isinstance(qty_base_recorded, (int, float)) and qty_base_recorded > 1e-9 and \
           side_recorded in ["BUY", "SELL"]:
            
            if side_recorded == "BUY": # Long
                pnl_gross_final_quote = (exit_price_avg - entry_price_recorded) * qty_base_recorded
            else: # Short
                pnl_gross_final_quote = (entry_price_recorded - exit_price_avg) * qty_base_recorded
            
            pnl_net_final_quote = pnl_gross_final_quote - entry_commission_quote_recorded - actual_exit_commission_quote
            logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id_closed}] PnL Calculé - Gross: {pnl_gross_final_quote:.4f}, "
                        f"Net: {pnl_net_final_quote:.4f} {self.quote_asset_for_commission}")
        else:
            logger.warning(f"{self.log_prefix}[Cycle:{trade_cycle_id_closed}] Données PnL incomplètes. "
                           f"ExitPx:{exit_price_avg}, EntryPx:{entry_price_recorded}, Qty:{qty_base_recorded}, Side:{side_recorded}")


        closed_trade_summary = {
            "trade_cycle_id": trade_cycle_id_closed,
            "timestamp_closure_utc_iso": datetime.now(timezone.utc).isoformat(),
            "pair_symbol": self.pair_symbol,
            "account_alias_used": self.state.get("instance_account_alias", "N/A"),
            "context_label_used": self.state.get("instance_context_label", "N/A"),

            "entry_order_id_api": entry_details_snapshot.get("orderId"),
            "entry_client_order_id": entry_details_snapshot.get("clientOrderId"),
            "entry_timestamp_ms": self.state.get("position_entry_timestamp_ms"),
            "position_side": side_recorded,
            "position_quantity_base": qty_base_recorded,
            "position_entry_price_avg": entry_price_recorded,
            "position_entry_commission_quote": entry_commission_quote_recorded,
            
            "margin_loan_details_at_entry": self.state.get("margin_loan_details", {}),
            
            "active_oco_order_list_id_api_at_closure": self.state.get("active_oco_order_list_id_api"),
            "active_sl_price_set_at_closure": self.state.get("active_sl_price_set"),
            "active_tp_price_set_at_closure": self.state.get("active_tp_price_set"),
            
            "exit_reason": exit_reason,
            "exit_order_id_api": closed_order_details_api.get("orderId") if closed_order_details_api else None,
            "exit_timestamp_ms": closed_order_details_api.get("transactTime") or \
                                 closed_order_details_api.get("updateTime") if closed_order_details_api else int(time.time() * 1000),
            "exit_price_avg_actual": exit_price_avg,
            "exit_commission_quote_actual": actual_exit_commission_quote,
            
            "pnl_gross_quote_final": pnl_gross_final_quote,
            "pnl_net_quote_final": pnl_net_final_quote,
            "closed_order_details_api_response": closed_order_details_api
        }
        
        # Mettre à jour `last_closed_trade_info` dans l'état.
        # La transition vers STATUT_1 se fera séparément et préservera cette information.
        self.update_specific_fields({"last_closed_trade_info": closed_trade_summary})
        logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id_closed}] Informations du trade clôturé enregistrées dans l'état.")

