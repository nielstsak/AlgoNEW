# src/utils/simulation_elements/fees_simulator.py
"""
Ce module définit la classe FeeSimulator, responsable de la simulation
des frais de transaction pour les opérations de trading.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class FeeSimulator:
    """
    Simule les frais de transaction pour les ordres de trading.
    """
    def __init__(self,
                 fee_bps: float,
                 min_fee: Optional[float] = None,
                 fee_asset: Optional[str] = None):
        """
        Initialise le simulateur de frais.

        Args:
            fee_bps (float): Les frais de transaction en points de base (BPS).
                             Par exemple, 7.5 pour 0.075%.
            min_fee (Optional[float]): Frais minimum par transaction.
                                       Actuellement non utilisé mais conservé pour une future extension.
            fee_asset (Optional[str]): L'actif dans lequel les frais sont payés.
                                        Actuellement non utilisé, on suppose que les frais
                                        sont déduits de l'actif de cotation.
        
        Raises:
            ValueError: Si fee_bps est négatif.
        """
        if not isinstance(fee_bps, (int, float)) or fee_bps < 0:
            logger.error(f"FeeSimulator: fee_bps doit être un nombre non négatif. Reçu : {fee_bps}")
            raise ValueError("fee_bps doit être un nombre non négatif.")

        self.fee_bps: float = fee_bps
        self.fee_rate: float = fee_bps / 10000.0  # Conversion BPS en taux décimal (ex: 7.5 BPS -> 0.00075)
        self.min_fee: Optional[float] = min_fee
        self.fee_asset: Optional[str] = fee_asset

        logger.info(f"FeeSimulator initialisé. Taux de frais : {self.fee_rate:.6f} ({self.fee_bps} BPS)")

    def calculate_fee(self, trade_value_quote: float) -> float:
        """
        Calcule les frais pour une transaction donnée.

        Args:
            trade_value_quote (float): La valeur totale de la transaction dans l'actif de cotation
                                       (ex: pour un achat de 0.1 BTC à 50000 USDT/BTC,
                                       trade_value_quote = 5000 USDT).

        Returns:
            float: Le montant des frais calculés, dans l'actif de cotation.
        """
        value_for_fee_calc = trade_value_quote
        if trade_value_quote < 0:
            logger.warning(f"FeeSimulator: La valeur du trade ({trade_value_quote}) est négative. "
                           "Le calcul des frais utilisera la valeur absolue.")
            value_for_fee_calc = abs(trade_value_quote)

        calculated_fee = value_for_fee_calc * self.fee_rate

        # La logique pour min_fee pourrait être implémentée ici si nécessaire à l'avenir.
        # if self.min_fee is not None and calculated_fee < self.min_fee:
        #     logger.debug(f"FeeSimulator: Frais calculés {calculated_fee:.8f} inférieurs au min_fee {self.min_fee}. "
        #                  "Utilisation du min_fee.")
        #     return self.min_fee

        logger.debug(f"FeeSimulator: Frais calculés pour une valeur de trade de {value_for_fee_calc:.2f} : {calculated_fee:.8f}")
        return calculated_fee

if __name__ == '__main__':
    # Configuration du logging pour les tests directs de ce module
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Test de FeeSimulator ---")

    # Test avec des frais standards (ex: 7.5 BPS = 0.075%)
    fee_sim_standard = FeeSimulator(fee_bps=7.5)
    trade_value1 = 10000.0  # ex: 10000 USDC
    fee1 = fee_sim_standard.calculate_fee(trade_value1)
    expected_fee1 = 10000.0 * (7.5 / 10000.0)
    logger.info(f"Pour une valeur de trade de {trade_value1}, les frais (7.5 BPS) sont : {fee1:.4f} (Attendu: {expected_fee1:.4f})")
    assert abs(fee1 - expected_fee1) < 1e-9, "Test standard échoué"

    # Test avec une valeur de trade plus petite
    trade_value2 = 500.0
    fee2 = fee_sim_standard.calculate_fee(trade_value2)
    expected_fee2 = 500.0 * (7.5 / 10000.0)
    logger.info(f"Pour une valeur de trade de {trade_value2}, les frais (7.5 BPS) sont : {fee2:.4f} (Attendu: {expected_fee2:.4f})")
    assert abs(fee2 - expected_fee2) < 1e-9, "Test petite valeur échoué"

    # Test avec des frais nuls
    fee_sim_zero = FeeSimulator(fee_bps=0.0)
    fee_zero = fee_sim_zero.calculate_fee(trade_value1)
    logger.info(f"Pour une valeur de trade de {trade_value1}, les frais (0 BPS) sont : {fee_zero:.4f} (Attendu: 0.0)")
    assert abs(fee_zero - 0.0) < 1e-9, "Test frais nuls échoué"

    # Test avec des frais plus élevés (ex: 100 BPS = 1%)
    fee_sim_high = FeeSimulator(fee_bps=100.0)
    fee_high = fee_sim_high.calculate_fee(trade_value1)
    expected_fee_high = 10000.0 * (100.0 / 10000.0)
    logger.info(f"Pour une valeur de trade de {trade_value1}, les frais (100 BPS) sont : {fee_high:.4f} (Attendu: {expected_fee_high:.4f})")
    assert abs(fee_high - expected_fee_high) < 1e-9, "Test frais élevés échoué"
    
    # Test avec une valeur de trade négative
    trade_value_neg = -2000.0
    fee_neg = fee_sim_standard.calculate_fee(trade_value_neg)
    expected_fee_neg = abs(trade_value_neg) * (7.5 / 10000.0)
    logger.info(f"Pour une valeur de trade de {trade_value_neg}, les frais (7.5 BPS) sont : {fee_neg:.4f} (Attendu: {expected_fee_neg:.4f})")
    assert abs(fee_neg - expected_fee_neg) < 1e-9, "Test valeur négative échoué"

    # Test d'initialisation avec fee_bps négatif (devrait lever ValueError)
    try:
        FeeSimulator(fee_bps=-1.0)
        logger.error("Test d'erreur pour fee_bps négatif A ÉCHOUÉ : ValueError non levée.")
        assert False, "ValueError non levée pour fee_bps négatif"
    except ValueError as e:
        logger.info(f"Test d'erreur pour fee_bps négatif RÉUSSI : {e}")
    
    logger.info("--- Tests de FeeSimulator terminés ---")
