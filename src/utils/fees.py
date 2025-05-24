# src/utils/fees.py
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class FeeSimulator:
    """
    Simule les frais de transaction pour les ordres de trading.
    """
    def __init__(self,
                 fee_bps: float, # Frais en points de base (par exemple, 7.5 pour 0.075%)
                 min_fee: Optional[float] = None, # Frais minimum par ordre (non utilisé actuellement)
                 fee_asset: Optional[str] = None): # Actif dans lequel les frais sont payés (non utilisé actuellement)
        """
        Initialise le simulateur de frais.

        Args:
            fee_bps (float): Les frais de transaction en points de base (BPS).
                             1 BPS = 0.01%. Donc 7.5 BPS = 0.075%.
            min_fee (Optional[float]): Frais minimum par transaction (non implémenté).
            fee_asset (Optional[str]): L'actif dans lequel les frais sont payés (non implémenté,
                                     on suppose que les frais sont déduits de l'actif de cotation).
        """
        if not isinstance(fee_bps, (int, float)) or fee_bps < 0:
            raise ValueError("fee_bps doit être un nombre non négatif.")

        self.fee_rate = fee_bps / 10000.0  # Convertir BPS en taux décimal (ex: 7.5 BPS -> 0.00075)
        self.min_fee = min_fee
        self.fee_asset = fee_asset # Non utilisé pour le moment, mais conservé pour une future extension

        logger.info(f"FeeSimulator initialized. Fee rate: {self.fee_rate:.5f} ({fee_bps} BPS)")

    def calculate_fee(self, trade_value_quote: float) -> float:
        """
        Calcule les frais pour une transaction donnée.

        Args:
            trade_value_quote (float): La valeur totale de la transaction dans l'actif de cotation.
                                       (par exemple, pour un achat de 0.1 BTC à 50000 USDT/BTC, trade_value_quote = 5000 USDT).

        Returns:
            float: Le montant des frais calculés, dans l'actif de cotation.
        """
        if trade_value_quote < 0:
            logger.warning(f"Trade value ({trade_value_quote}) is negative. Fee calculation will use absolute value.")
            trade_value_quote = abs(trade_value_quote)

        calculated_fee = trade_value_quote * self.fee_rate

        # Logique pour min_fee (si implémentée à l'avenir)
        # if self.min_fee is not None and calculated_fee < self.min_fee:
        #     logger.debug(f"Calculated fee {calculated_fee} is less than min_fee {self.min_fee}. Using min_fee.")
        #     return self.min_fee

        # logger.debug(f"Calculated fee for trade value {trade_value_quote:.2f} is {calculated_fee:.8f}")
        return calculated_fee

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Test avec des frais de 7.5 BPS (0.075%)
    fee_sim_standard = FeeSimulator(fee_bps=7.5)
    trade_val1 = 10000.0 # e.g., USDT
    fee1 = fee_sim_standard.calculate_fee(trade_val1)
    logger.debug(f"Pour une valeur de trade de {trade_val1}, les frais (7.5 BPS) sont: {fee1:.4f} (Attendu: 7.5)")

    trade_val2 = 500.0
    fee2 = fee_sim_standard.calculate_fee(trade_val2)
    logger.debug(f"Pour une valeur de trade de {trade_val2}, les frais (7.5 BPS) sont: {fee2:.4f} (Attendu: 0.375)")

    # Test avec des frais de 0 BPS
    fee_sim_zero = FeeSimulator(fee_bps=0.0)
    fee_zero = fee_sim_zero.calculate_fee(trade_val1)
    logger.debug(f"Pour une valeur de trade de {trade_val1}, les frais (0 BPS) sont: {fee_zero:.4f} (Attendu: 0.0)")

    # Test avec des frais de 100 BPS (1%)
    fee_sim_high = FeeSimulator(fee_bps=100.0)
    fee_high = fee_sim_high.calculate_fee(trade_val1)
    logger.debug(f"Pour une valeur de trade de {trade_val1}, les frais (100 BPS) sont: {fee_high:.4f} (Attendu: 100.0)")

    try:
        FeeSimulator(fee_bps=-1.0)
    except ValueError as e:
        logger.debug(f"Test d'erreur pour fee_bps négatif réussi: {e}")
