# src/utils/simulation_elements/fees_simulator.py
"""
Ce module définit la classe FeeSimulator, responsable de la simulation
des frais de transaction pour les opérations de trading.
"""
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

class FeeSimulator:
    """
    Simule les frais de transaction pour les ordres de trading.
    """
    def __init__(self,
                 fee_bps: Union[float, int],
                 min_fee_quote: Optional[Union[float, int]] = None, # Frais minimum par transaction en actif de cotation
                 fee_asset_is_base: bool = False): # Si True, les frais sont payés en actif de base
        """
        Initialise le simulateur de frais.

        Args:
            fee_bps (Union[float, int]): Les frais de transaction en points de base (BPS).
                                         Par exemple, 7.5 pour 0.075%.
            min_fee_quote (Optional[Union[float, int]]): Frais minimum par transaction, exprimés
                                                       dans l'actif de cotation.
                                                       Si None, aucun frais minimum n'est appliqué.
            fee_asset_is_base (bool): Si True, simule que les frais sont payés dans l'actif de base
                                      (nécessite une conversion au moment du calcul si la valeur du trade
                                      est en actif de cotation). Si False (par défaut), les frais sont
                                      supposés être déduits de l'actif de cotation.
                                      Cette fonctionnalité est plus complexe et pour l'instant,
                                      le simulateur se concentre sur les frais en actif de cotation.
        
        Raises:
            ValueError: Si fee_bps est négatif.
        """
        if not isinstance(fee_bps, (int, float)) or fee_bps < 0:
            msg = f"fee_bps ({fee_bps}) doit être un nombre non négatif."
            logger.error(f"[FeeSimulator] {msg}")
            raise ValueError(msg)

        self.fee_bps: float = float(fee_bps)
        self.fee_rate: float = self.fee_bps / 10000.0  # Conversion BPS en taux décimal (ex: 7.5 BPS -> 0.00075)
        
        self.min_fee_quote: Optional[float] = None
        if min_fee_quote is not None:
            if not isinstance(min_fee_quote, (int, float)) or min_fee_quote < 0:
                msg = f"min_fee_quote ({min_fee_quote}) doit être un nombre non négatif s'il est fourni."
                logger.error(f"[FeeSimulator] {msg}")
                raise ValueError(msg)
            self.min_fee_quote = float(min_fee_quote)

        self.fee_asset_is_base: bool = fee_asset_is_base
        if self.fee_asset_is_base:
            # La logique pour les frais en actif de base est plus complexe et non entièrement
            # implémentée ici car elle nécessiterait le prix pour convertir la valeur du trade.
            # Le BacktestRunner devra gérer cela s'il supporte ce mode.
            logger.warning("[FeeSimulator] Le mode fee_asset_is_base=True est noté mais le calcul actuel "
                           "suppose que trade_value_quote est la base pour les frais en quote_asset. "
                           "Une logique de conversion supplémentaire serait nécessaire pour les frais en base_asset.")

        logger.info(f"[FeeSimulator] Initialisé. Taux de frais: {self.fee_rate:.6f} ({self.fee_bps} BPS), "
                    f"Min Fee (Quote): {self.min_fee_quote if self.min_fee_quote is not None else 'N/A'}, "
                    f"Frais en Base Asset: {self.fee_asset_is_base}")

    def calculate_fee(self,
                      trade_value_quote: float,
                      # Les arguments suivants seraient nécessaires pour des frais en actif de base
                      # quantity_base: Optional[float] = None,
                      # price: Optional[float] = None
                     ) -> float:
        """
        Calcule les frais pour une transaction donnée, en supposant que les frais sont
        prélevés sur l'actif de cotation et basés sur la valeur du trade en actif de cotation.

        Args:
            trade_value_quote (float): La valeur totale de la transaction dans l'actif de cotation
                                       (ex: pour un achat de 0.1 BTC à 50000 USDT/BTC,
                                       trade_value_quote = 5000 USDT).
                                       Doit être une valeur positive.

        Returns:
            float: Le montant des frais calculés, dans l'actif de cotation.
                   Retourne 0.0 si trade_value_quote est non positif.
        """
        if trade_value_quote <= 1e-9: # Utiliser un petit seuil pour éviter les valeurs négatives ou nulles
            logger.debug(f"[FeeSimulator] Valeur du trade ({trade_value_quote:.8f}) non positive. "
                           "Aucun frais calculé.")
            return 0.0

        calculated_fee = trade_value_quote * self.fee_rate

        if self.min_fee_quote is not None and calculated_fee < self.min_fee_quote:
            logger.debug(f"[FeeSimulator] Frais calculés {calculated_fee:.8f} < min_fee_quote {self.min_fee_quote}. "
                         f"Application du min_fee_quote.")
            final_fee = self.min_fee_quote
        else:
            final_fee = calculated_fee
        
        # La logique pour fee_asset_is_base nécessiterait de convertir `final_fee` (qui est en quote)
        # en une quantité équivalente de base_asset, si `price` est fourni.
        # Ou, si les frais sont nativement en base_asset, le calcul initial serait différent.
        # Pour l'instant, on retourne toujours les frais en quote_asset.
        if self.fee_asset_is_base:
            logger.debug("[FeeSimulator] Calcul des frais en actif de base non implémenté dans cette version. "
                        f"Retour des frais calculés en actif de cotation: {final_fee:.8f}")

        logger.debug(f"[FeeSimulator] Frais calculés pour valeur de trade {trade_value_quote:.2f} : {final_fee:.8f} (en quote asset)")
        return final_fee

