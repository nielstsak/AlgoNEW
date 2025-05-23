import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, Union
from datetime import datetime, timezone # Pour ISO format timestamps

logger = logging.getLogger(__name__)

# --- Constantes de Statut ---
STATUT_1_NO_TRADE_NO_OCO = "STATUT_1_NO_TRADE_NO_OCO"
STATUT_2_ENTRY_FILLED_OCO_PENDING = "STATUT_2_ENTRY_FILLED_OCO_PENDING"
STATUT_3_OCO_ACTIVE = "STATUT_3_OCO_ACTIVE"
# STATUT_4_ERROR_REQUIRES_MANUAL_INTERVENTION = "STATUT_4_ERROR_REQUIRES_MANUAL_INTERVENTION" # Exemple si besoin

# --- Constantes d'Actifs ---
# Principalement utilisé pour identifier l'asset de commission.
# Peut être surchargé ou rendu configurable si d'autres assets de cotation sont utilisés.
USDC_ASSET = "USDC" # Ou BUSD, USDT, etc.

class LiveTradingState:
    def __init__(self, pair_symbol: str, state_file_path: Union[str, Path]):
        """
        Initialise le gestionnaire d'état pour une instance de trading.

        Args:
            pair_symbol (str): Le symbole de la paire de trading (ex: "BTCUSDC").
            state_file_path (Union[str, Path]): Chemin vers le fichier JSON de l'état.
        """
        self.pair_symbol = pair_symbol.upper()
        self.state_file_path = Path(state_file_path)
        self.log_prefix = f"[{self.pair_symbol}][State]" # Pour les logs spécifiques à cette instance d'état

        self.state: Dict[str, Any] = self._load_state()

        # Validation initiale après chargement ou création par défaut
        if not self.state.get("current_status") or self.state.get("pair_symbol") != self.pair_symbol:
            logger.warning(
                f"{self.log_prefix} État initial chargé invalide, pour une paire différente, ou fichier inexistant. "
                f"PairSymbol dans état: {self.state.get('pair_symbol')}. Forçage de la réinitialisation avec les valeurs par défaut."
            )
            # _load_state devrait déjà avoir appelé _default_state si le fichier n'existait pas
            # ou si la paire ne correspondait pas (selon l'implémentation de _load_state).
            # Si on arrive ici, c'est que _load_state a retourné un état potentiellement corrompu
            # ou qu'une vérification supplémentaire est souhaitée.
            self.state = self._default_state() # Assurer un état propre
            self._save_state() # Sauvegarder l'état par défaut fraîchement initialisé

        logger.info(f"{self.log_prefix} LiveTradingState initialisé. Statut actuel : {self.get_current_status_name()}. Fichier : {self.state_file_path}")

    def _default_state(self) -> Dict[str, Any]:
        """
        Retourne la structure et les valeurs par défaut pour un nouvel état.
        """
        current_time_ms = time.time() * 1000
        return {
            "pair_symbol": self.pair_symbol,
            "current_status": STATUT_1_NO_TRADE_NO_OCO,
            "current_trade_cycle_id": None, # ID unique pour un cycle complet entrée->sortie
            "last_status_update_timestamp": current_time_ms, # Timestamp en ms
            "last_error": None,
            "available_capital_at_last_check": 0.0, # Capital en USDC (ou équivalent)

            # Détails de l'Ordre d'Entrée en Attente
            "pending_entry_order_id": None, # ID de l'exchange
            "pending_entry_client_order_id": None,
            "pending_entry_params": {}, # Paramètres envoyés à l'API pour l'entrée
            "pending_sl_tp_raw": {}, # Prix SL/TP bruts { "sl_price": float, "tp_price": float }

            # Détails de la Position Ouverte
            "entry_order_details": {}, # Réponse complète de l'API pour l'ordre d'entrée rempli
            "position_side": None, # "BUY" ou "SELL"
            "position_quantity": 0.0,
            "position_entry_price": 0.0, # Prix moyen d'exécution
            "position_entry_timestamp": None, # Timestamp en ms de l'exécution
            "position_total_commission_usdc_equivalent": 0.0, # Commission pour l'entrée en USDC

            # Détails du Prêt sur Marge
            "loan_details": {"asset": None, "amount": 0.0, "timestamp": None}, # asset, amount, timestamp_ms

            # Détails de l'Ordre OCO en Attente de Placement
            "oco_params_to_place": {}, # Paramètres construits pour l'API OCO
            "pending_oco_list_client_order_id": None, # Client Order ID pour la liste OCO
            "pending_oco_order_list_id_api": None, # OrderListId de l'exchange pour l'OCO

            # Détails de l'Ordre OCO Actif
            "active_oco_details": {}, # Réponse complète de l'API pour l'OCO placé
            "active_oco_list_client_order_id": None,
            "active_oco_order_list_id": None, # OrderListId de l'exchange
            "active_sl_order_id": None,
            "active_tp_order_id": None,
            "active_sl_price": None,
            "active_tp_price": None,
            "oco_active_timestamp": None, # Timestamp en ms

            # Informations sur le Dernier Trade Clôturé
            "last_closed_trade_info": {}, # Résumé du dernier cycle de trade

            # Suivi de la synchronisation
            "last_successful_sync_timestamp": None, # Timestamp en ms de la dernière synchro réussie
        }

    def _load_state(self) -> Dict[str, Any]:
        """
        Charge l'état depuis le fichier JSON.
        Si le fichier n'existe pas, est corrompu, ou pour une autre paire, retourne un état par défaut.
        Complète les champs manquants avec les valeurs par défaut si un état existant est chargé.
        """
        default_s = self._default_state()
        if self.state_file_path.exists() and self.state_file_path.is_file():
            try:
                with open(self.state_file_path, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                
                if not isinstance(loaded_state, dict):
                    logger.warning(f"{self.log_prefix} Fichier d'état {self.state_file_path} ne contient pas un objet JSON valide. Réinitialisation.")
                    return default_s

                if loaded_state.get("pair_symbol") == self.pair_symbol and loaded_state.get("current_status"):
                    logger.info(f"{self.log_prefix} État chargé depuis {self.state_file_path}")
                    # Complétion des champs manquants pour la rétrocompatibilité
                    updated = False
                    for key, default_value in default_s.items():
                        if key not in loaded_state:
                            loaded_state[key] = default_value
                            logger.info(f"{self.log_prefix} Ajout du champ manquant '{key}' avec la valeur par défaut à l'état chargé.")
                            updated = True
                    if updated:
                        # Sauvegarder immédiatement si des champs ont été ajoutés
                        # self.state = loaded_state # Temporairement assigner pour que _save_state fonctionne
                        # self._save_state() # Cela pourrait être problématique si appelé dans __init__ avant que self.state soit pleinement assigné
                        # Il est préférable de laisser la sauvegarde se faire après l'appel à _load_state dans __init__ si nécessaire.
                        pass
                    return loaded_state
                else:
                    logger.warning(f"{self.log_prefix} Fichier d'état {self.state_file_path} pour une paire différente "
                                   f"(attendu: {self.pair_symbol}, trouvé: {loaded_state.get('pair_symbol')}) ou statut manquant. Réinitialisation.")
            except json.JSONDecodeError:
                logger.error(f"{self.log_prefix} Erreur de décodage JSON du fichier d'état {self.state_file_path}. Réinitialisation.", exc_info=True)
            except Exception as e:
                logger.error(f"{self.log_prefix} Erreur inattendue lors du chargement de l'état depuis {self.state_file_path}: {e}. Réinitialisation.", exc_info=True)
        else:
            logger.info(f"{self.log_prefix} Fichier d'état non trouvé à {self.state_file_path}. Initialisation avec l'état par défaut.")
        return default_s # Retourner l'état par défaut en cas d'échec ou de non-existence

    def _save_state(self):
        """Sauvegarde l'état actuel dans le fichier JSON."""
        try:
            self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=4, default=str) # default=str pour sérialiser les types non natifs JSON
            logger.debug(f"{self.log_prefix} État sauvegardé dans {self.state_file_path}")
        except IOError as e:
            logger.error(f"{self.log_prefix} Erreur d'IO lors de la sauvegarde de l'état dans {self.state_file_path}: {e}", exc_info=True)
        except Exception as e_gen:
            logger.error(f"{self.log_prefix} Erreur inattendue lors de la sauvegarde de l'état: {e_gen}", exc_info=True)


    def update_specific_fields(self, update_dict: Dict[str, Any]):
        """Met à jour des champs spécifiques dans l'état et sauvegarde."""
        if not isinstance(update_dict, dict):
            logger.error(f"{self.log_prefix} update_dict doit être un dictionnaire. Reçu: {type(update_dict)}")
            return

        for key, value in update_dict.items():
            self.state[key] = value
        self.state["last_status_update_timestamp"] = time.time() * 1000 # ms
        self._save_state()
        logger.info(f"{self.log_prefix} Champs d'état mis à jour : {list(update_dict.keys())}. Statut actuel : {self.get_current_status_name()}")

    def get_current_status(self) -> str:
        """Retourne le statut de trading actuel."""
        return self.state.get("current_status", STATUT_1_NO_TRADE_NO_OCO)

    def get_current_status_name(self) -> str:
        """Retourne le nom du statut actuel sans le préfixe "STATUT_"."""
        return self.state.get("current_status", "UNKNOWN").replace("STATUT_", "")

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Retourne une copie de l'état actuel."""
        return self.state.copy()

    def set_last_error(self, error_message: Optional[str]):
        """Enregistre un message d'erreur dans l'état."""
        self.update_specific_fields({"last_error": error_message})
        if error_message:
            logger.error(f"{self.log_prefix} Erreur enregistrée dans l'état : {error_message}")

    def clear_last_error(self):
        """Efface le dernier message d'erreur de l'état."""
        if self.state.get("last_error"):
            self.update_specific_fields({"last_error": None})
            logger.info(f"{self.log_prefix} Erreur précédente effacée de l'état.")
            
    def update_last_successful_sync_timestamp(self):
        """Met à jour le timestamp de la dernière synchronisation réussie."""
        self.update_specific_fields({"last_successful_sync_timestamp": time.time() * 1000})


    # --- Méthodes de Transition d'État ---
    def transition_to_status_1(self, exit_reason: Optional[str] = "REINITIALIZATION",
                                 closed_trade_details: Optional[Dict[str, Any]] = None):
        """Transitionne vers STATUT_1_NO_TRADE_NO_OCO, réinitialisant la plupart des champs."""
        logger.info(f"{self.log_prefix} Transition vers STATUT_1_NO_TRADE_NO_OCO. Raison : {exit_reason}. "
                    f"Cycle ID actuel (avant réinit) : {self.state.get('current_trade_cycle_id')}")
        
        # Préserver certaines informations importantes
        preserved_capital = self.state.get("available_capital_at_last_check", 0.0)
        last_trade_info = self.state.get("last_closed_trade_info", {})
        if closed_trade_details: # Si de nouveaux détails de clôture sont fournis, ils priment
            last_trade_info = closed_trade_details # Ceci devrait être le dict complet construit par record_closed_trade
        
        # Réinitialiser à l'état par défaut
        new_state_fields = self._default_state()
        
        # Restaurer les informations préservées
        new_state_fields["available_capital_at_last_check"] = preserved_capital
        new_state_fields["last_closed_trade_info"] = last_trade_info
        # current_trade_cycle_id est déjà None dans _default_state

        self.state.update(new_state_fields) # Appliquer tous les champs par défaut
        self.state["last_status_update_timestamp"] = time.time() * 1000
        self.clear_last_error() # Effacer les erreurs précédentes lors d'une réinitialisation propre
        self._save_state()
        logger.info(f"{self.log_prefix} Transition vers STATUT_1 terminée.")


    def prepare_for_entry_order(self, entry_params: Dict[str, Any], sl_tp_raw: Dict[str, float], trade_cycle_id: str):
        """Prépare l'état pour placer un ordre d'entrée."""
        logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id}] Préparation pour ordre d'entrée. Params: {entry_params}, SL/TP bruts: {sl_tp_raw}")
        update_dict = {
            "pending_entry_params": entry_params,
            "pending_sl_tp_raw": sl_tp_raw,
            "current_trade_cycle_id": trade_cycle_id,
            "pending_entry_order_id": None, # Réinitialiser au cas où
            "pending_entry_client_order_id": None, # Réinitialiser au cas où
            "last_error": None # Effacer les erreurs précédentes avant une nouvelle tentative
        }
        self.update_specific_fields(update_dict)

    def record_placed_entry_order(self, order_id: Union[str, int], client_order_id: str):
        """Enregistre les IDs d'un ordre d'entrée qui vient d'être placé."""
        trade_cycle_id = self.state.get("current_trade_cycle_id", "UNKNOWN_CYCLE")
        logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id}] Ordre d'entrée placé. ExchangeID: {order_id}, ClientID: {client_order_id}")
        self.update_specific_fields({
            "pending_entry_order_id": str(order_id), # S'assurer que c'est une chaîne
            "pending_entry_client_order_id": client_order_id
        })

    def transition_to_status_2(self, filled_entry_details: Dict[str, Any], loan_info: Dict[str, Any]):
        """Transitionne vers STATUT_2_ENTRY_FILLED_OCO_PENDING après qu'un ordre d'entrée a été rempli."""
        trade_cycle_id = self.state.get("current_trade_cycle_id", "UNKNOWN_CYCLE")
        logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id}] Transition vers STATUT_2_ENTRY_FILLED_OCO_PENDING. Détails entrée: {filled_entry_details.get('orderId')}, Prêt: {loan_info.get('asset')}")
        
        executed_qty = float(filled_entry_details.get("executedQty", 0.0))
        cummulative_quote_qty = float(filled_entry_details.get("cummulativeQuoteQty", 0.0))
        entry_price = 0.0
        if executed_qty > 1e-9: # Eviter division par zéro pour les très petites quantités
            entry_price = cummulative_quote_qty / executed_qty
        
        commission_total_usdc = 0.0
        if 'fills' in filled_entry_details and isinstance(filled_entry_details['fills'], list):
            for fill in filled_entry_details['fills']:
                if fill.get('commissionAsset', '').upper() == USDC_ASSET: # ou self.quote_asset si c'est la devise de commission
                    commission_total_usdc += float(fill.get('commission', 0.0))
        elif filled_entry_details.get('commissionAsset', '').upper() == USDC_ASSET: # Fallback si pas de 'fills' détaillés
             commission_total_usdc = float(filled_entry_details.get('commission', 0.0))

        update_dict = {
            "current_status": STATUT_2_ENTRY_FILLED_OCO_PENDING,
            "pending_entry_order_id": None, # L'ordre d'entrée n'est plus en attente
            "pending_entry_client_order_id": None,
            "pending_entry_params": {}, # Effacer les params de l'entrée
            # pending_sl_tp_raw est conservé car nécessaire pour construire l'OCO

            "entry_order_details": filled_entry_details,
            "position_side": filled_entry_details.get("side"), # "BUY" ou "SELL"
            "position_quantity": executed_qty,
            "position_entry_price": entry_price,
            "position_entry_timestamp": filled_entry_details.get("transactTime") or filled_entry_details.get("updateTime") or (time.time() * 1000),
            "position_total_commission_usdc_equivalent": commission_total_usdc,
            
            "loan_details": loan_info, # {"asset": str, "amount": float, "timestamp": float_ms}

            # Réinitialiser les champs OCO en préparation du placement
            "oco_params_to_place": {},
            "pending_oco_list_client_order_id": None,
            "pending_oco_order_list_id_api": None,
            "active_oco_details": {},
            "active_oco_list_client_order_id": None,
            "active_oco_order_list_id": None,
            "active_sl_order_id": None,
            "active_tp_order_id": None,
            "active_sl_price": None,
            "active_tp_price": None,
            "oco_active_timestamp": None,
            "last_error": None # Effacer les erreurs précédentes
        }
        self.update_specific_fields(update_dict)

    def prepare_for_oco_order(self, oco_params: Dict[str, Any], list_client_order_id: str, order_list_id_api: Optional[str] = None):
        """Prépare l'état pour placer un ordre OCO."""
        trade_cycle_id = self.state.get("current_trade_cycle_id", "UNKNOWN_CYCLE")
        logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id}] Préparation pour ordre OCO. ListClientOrderID: {list_client_order_id}, API OrderListID (si connu): {order_list_id_api}")
        update_dict = {
            "oco_params_to_place": oco_params,
            "pending_oco_list_client_order_id": list_client_order_id,
            "pending_oco_order_list_id_api": order_list_id_api, # Peut être None si non retourné immédiatement
            
            # Réinitialiser les champs OCO actifs car on s'apprête à en placer un nouveau
            "active_oco_details": {},
            "active_oco_list_client_order_id": None,
            "active_oco_order_list_id": None,
            "active_sl_order_id": None,
            "active_tp_order_id": None,
            "active_sl_price": None,
            "active_tp_price": None,
            "oco_active_timestamp": None,
            "last_error": None # Effacer les erreurs précédentes
        }
        self.update_specific_fields(update_dict)

    def transition_to_status_3(self, active_oco_api_response: Dict[str, Any]):
        """Transitionne vers STATUT_3_OCO_ACTIVE après qu'un ordre OCO a été placé et confirmé."""
        trade_cycle_id = self.state.get("current_trade_cycle_id", "UNKNOWN_CYCLE")
        logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id}] Transition vers STATUT_3_OCO_ACTIVE. OrderListID API: {active_oco_api_response.get('orderListId')}")
        
        list_client_order_id = active_oco_api_response.get("listClientOrderId", self.state.get("pending_oco_list_client_order_id"))
        order_list_id_api = active_oco_api_response.get("orderListId")
        
        sl_order_id, tp_order_id = None, None
        sl_price, tp_price = None, None
        
        # La réponse API pour un OCO contient une liste 'orders' ou 'orderReports'
        order_reports = active_oco_api_response.get("orders", active_oco_api_response.get("orderReports", []))

        if isinstance(order_reports, list):
            for report in order_reports:
                if not isinstance(report, dict): continue
                order_type = report.get("type", "").upper()
                if order_type in ["STOP_LOSS_LIMIT", "STOP_LOSS", "STOP_MARKET"]:
                    sl_order_id = str(report.get("orderId"))
                    sl_price = float(report.get("stopPrice", 0.0)) # stopPrice est le prix de déclenchement du SL
                elif order_type in ["LIMIT_MAKER", "TAKE_PROFIT_LIMIT", "LIMIT", "TAKE_PROFIT"]: # LIMIT_MAKER pour le TP
                    tp_order_id = str(report.get("orderId"))
                    tp_price = float(report.get("price", 0.0)) # price est le prix limite du TP
        
        update_dict = {
            "current_status": STATUT_3_OCO_ACTIVE,
            "oco_params_to_place": {}, # Effacer car l'OCO est placé
            "pending_oco_list_client_order_id": None,
            "pending_oco_order_list_id_api": None,

            "active_oco_details": active_oco_api_response,
            "active_oco_list_client_order_id": list_client_order_id,
            "active_oco_order_list_id": str(order_list_id_api) if order_list_id_api is not None else None,
            "active_sl_order_id": sl_order_id,
            "active_tp_order_id": tp_order_id,
            "active_sl_price": sl_price,
            "active_tp_price": tp_price,
            "oco_active_timestamp": active_oco_api_response.get("transactionTime") or (time.time() * 1000),
            "last_error": None
        }
        self.update_specific_fields(update_dict)

    def record_closed_trade(self, exit_reason: str, exit_price: Optional[float], 
                              pnl_usdc_estimate: Optional[float], # PnL net estimé (avant frais de clôture exacts si non connus)
                              closed_order_details: Optional[Dict[str, Any]]):
        """Enregistre les informations d'un trade clôturé dans last_closed_trade_info."""
        trade_cycle_id = self.state.get("current_trade_cycle_id", f"ORPHAN_CLOSURE_{int(time.time()*1000)}")
        logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id}] Enregistrement du trade clôturé. Raison: {exit_reason}, Prix Sortie: {exit_price}, PnL Est.: {pnl_usdc_estimate}")
        
        entry_details_snapshot = self.state.get("entry_order_details", {})
        commission_entry_usdc = self.state.get("position_total_commission_usdc_equivalent", 0.0)
        
        commission_exit_usdc = 0.0
        if closed_order_details and isinstance(closed_order_details.get('fills'), list):
            for fill in closed_order_details['fills']:
                if fill.get('commissionAsset', '').upper() == USDC_ASSET: # ou self.quote_asset
                    commission_exit_usdc += float(fill.get('commission', 0.0))
        elif closed_order_details and closed_order_details.get('commissionAsset', '').upper() == USDC_ASSET:
            commission_exit_usdc = float(closed_order_details.get('commission', 0.0))

        # Calcul du PnL net final si possible
        pnl_net_final = pnl_usdc_estimate # Utiliser l'estimation si pas d'autres infos
        if exit_price is not None and \
           isinstance(self.state.get("position_entry_price"), (int, float)) and \
           isinstance(self.state.get("position_quantity"), (int, float)) and \
           self.state.get("position_side"):
            
            entry_p = self.state["position_entry_price"]
            qty = self.state["position_quantity"]
            side = self.state["position_side"]
            
            pnl_gross_calc = (exit_price - entry_p) * qty if side == "BUY" else (entry_p - exit_price) * qty
            pnl_net_final = pnl_gross_calc - commission_entry_usdc - commission_exit_usdc


        closed_trade_summary = {
            "trade_cycle_id": trade_cycle_id,
            "timestamp_closure_utc": datetime.now(timezone.utc).isoformat(),
            "pair_symbol": self.pair_symbol,
            "account_alias": self.state.get("account_alias_for_this_instance", "N/A"), # A ajouter à _default_state si besoin
            "context_label": self.state.get("context_label_for_this_instance", "N/A"), # A ajouter

            "entry_order_id_api": entry_details_snapshot.get("orderId"),
            "entry_timestamp_ms": self.state.get("position_entry_timestamp"),
            "position_side": self.state.get("position_side"),
            "position_quantity": self.state.get("position_quantity"),
            "position_entry_price": self.state.get("position_entry_price"),
            "commission_entry_usdc": commission_entry_usdc,
            
            "loan_details_at_entry": self.state.get("loan_details", {}),
            
            "oco_order_list_id_api": self.state.get("active_oco_order_list_id"),
            "oco_sl_price_set": self.state.get("active_sl_price"),
            "oco_tp_price_set": self.state.get("active_tp_price"),
            
            "exit_reason": exit_reason,
            "exit_order_id_api": closed_order_details.get("orderId") if closed_order_details else None,
            "exit_timestamp_ms": closed_order_details.get("transactTime") or closed_order_details.get("updateTime") if closed_order_details else (time.time() * 1000),
            "exit_price_actual": exit_price,
            "commission_exit_usdc": commission_exit_usdc,
            
            "pnl_usdc_net_final_estimate": pnl_net_final, # PnL Net final estimé
            "closed_order_api_response": closed_order_details # Réponse complète de l'ordre de clôture
        }
        
        # Utiliser update_specific_fields pour sauvegarder et mettre à jour le timestamp
        self.update_specific_fields({"last_closed_trade_info": closed_trade_summary})
        logger.info(f"{self.log_prefix}[Cycle:{trade_cycle_id}] Informations du trade clôturé enregistrées.")
