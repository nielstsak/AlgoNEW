# src/utils/slippage.py
import logging
import numpy as np
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SlippageSimulator:
    """
    Simule le slippage pour les ordres de trading.
    """
    def __init__(self,
                 method: str = "percentage",
                 percentage_max_bps: Optional[float] = None, # Slippage max en points de base
                 volume_factor: Optional[float] = None,    # Facteur pour le slippage basé sur le volume
                 volatility_factor: Optional[float] = None, # Facteur pour le slippage basé sur la volatilité
                 min_slippage_bps: float = 0.0,           # Slippage minimum à appliquer en BPS
                 max_slippage_bps: float = 100.0):        # Plafond de slippage en BPS (100 BPS = 1%)
        """
        Initialise le simulateur de slippage.

        Args:
            method (str): Méthode de calcul du slippage ('percentage', 'volume_based', 'volatility_based', 'fixed_bps', 'none').
            percentage_max_bps (Optional[float]): Pour la méthode 'percentage', le slippage maximum en points de base.
                                                  Un slippage aléatoire sera choisi entre min_slippage_bps et ce max.
            volume_factor (Optional[float]): Facteur pour la méthode 'volume_based'.
            volatility_factor (Optional[float]): Facteur pour la méthode 'volatility_based'.
            min_slippage_bps (float): Slippage minimum à appliquer en points de base (0.0 par défaut).
            max_slippage_bps (float): Plafond absolu du slippage en points de base (100.0 par défaut, soit 1%).
        """
        self.method = method.lower()
        self.percentage_max_bps = percentage_max_bps
        self.volume_factor = volume_factor
        self.volatility_factor = volatility_factor
        self.min_slippage_bps = max(0.0, min_slippage_bps) # Assurer que min_slippage_bps >= 0
        self.max_slippage_bps = max(self.min_slippage_bps, max_slippage_bps) # Assurer max >= min

        if self.method == "percentage" and self.percentage_max_bps is None:
            logger.warning("Slippage method 'percentage' selected but 'percentage_max_bps' not provided. Defaulting to 0 BPS max.")
            self.percentage_max_bps = 0.0
        if self.percentage_max_bps is not None:
            self.percentage_max_bps = max(self.min_slippage_bps, self.percentage_max_bps) # Assurer que le max du % est >= min global

        logger.info(f"SlippageSimulator initialized. Method: {self.method}, MinBPS: {self.min_slippage_bps}, MaxBPS: {self.max_slippage_bps}, PercMaxBPS: {self.percentage_max_bps}")

    def simulate_slippage(self,
                          price: float,
                          direction: int, # 1 pour achat (le prix augmente), -1 pour vente (le prix baisse)
                          volume_at_price: Optional[float] = None, # Volume du carnet d'ordres au niveau de prix
                          volatility: Optional[float] = None # Mesure de volatilité (ex: ATR en pourcentage)
                         ) -> float:
        """
        Simule le slippage sur un prix donné.

        Args:
            price (float): Le prix théorique d'exécution avant slippage.
            direction (int): Direction du trade initiant le slippage.
                             1 si l'ordre est un achat (susceptible de faire monter le prix d'exécution).
                             -1 si l'ordre est une vente (susceptible de faire baisser le prix d'exécution).
            volume_at_price (Optional[float]): Volume disponible au prix `price` (pour la méthode `volume_based`).
            volatility (Optional[float]): Mesure de la volatilité récente (pour la méthode `volatility_based`).

        Returns:
            float: Le prix ajusté après simulation du slippage.
        """
        if self.method == "none" or price <= 0:
            return price

        slippage_bps = 0.0

        if self.method == "percentage":
            # Slippage aléatoire entre min_slippage_bps et percentage_max_bps (ou max_slippage_bps si plus petit)
            upper_bound_bps = self.percentage_max_bps if self.percentage_max_bps is not None else self.max_slippage_bps
            # Assurer que l'upper_bound pour le random est au moins min_slippage_bps
            effective_upper_bound_bps = max(self.min_slippage_bps, upper_bound_bps)
            
            if effective_upper_bound_bps < self.min_slippage_bps: # Devrait pas arriver avec les gardes au-dessus
                slippage_bps = self.min_slippage_bps
            else:
                slippage_bps = np.random.uniform(self.min_slippage_bps, effective_upper_bound_bps)

        elif self.method == "fixed_bps":
            # Utilise percentage_max_bps comme la valeur fixe si fournie, sinon min_slippage_bps
            slippage_bps = self.percentage_max_bps if self.percentage_max_bps is not None else self.min_slippage_bps

        elif self.method == "volume_based":
            if volume_at_price is not None and self.volume_factor is not None and volume_at_price > 1e-9:
                # Exemple simple: slippage inversement proportionnel au volume disponible, avec un facteur
                # Plus de volume -> moins de slippage.
                # Ceci est une simplification. Un modèle réel serait plus complexe.
                slippage_bps = (1 / volume_at_price) * self.volume_factor * 10000 # *10000 pour convertir en BPS
            else:
                slippage_bps = self.min_slippage_bps # Fallback si volume non dispo

        elif self.method == "volatility_based":
            if volatility is not None and self.volatility_factor is not None:
                # Exemple: slippage proportionnel à la volatilité
                slippage_bps = volatility * self.volatility_factor * 10000 # Volatility est un % (0.01 pour 1%), *10000 pour BPS
            else:
                slippage_bps = self.min_slippage_bps # Fallback

        else: # Méthode inconnue ou non implémentée
            logger.warning(f"Unknown slippage method: '{self.method}'. Applying min_slippage_bps: {self.min_slippage_bps} BPS.")
            slippage_bps = self.min_slippage_bps

        # Plafonner le slippage calculé par max_slippage_bps et s'assurer qu'il est au moins min_slippage_bps
        slippage_bps = max(self.min_slippage_bps, min(slippage_bps, self.max_slippage_bps))

        slippage_decimal = slippage_bps / 10000.0  # Convertir BPS en décimal (ex: 10 BPS = 0.001)

        # Le slippage est toujours défavorable
        if direction == 1:  # Achat, le prix d'exécution augmente
            slipped_price = price * (1 + slippage_decimal)
        elif direction == -1:  # Vente, le prix d'exécution baisse
            slipped_price = price * (1 - slippage_decimal)
        else: # Direction neutre ou inconnue, pas de slippage directionnel
            slipped_price = price

        # logger.debug(f"Slippage: PriceIn={price:.4f}, Direction={direction}, Method='{self.method}', SlippageBPS={slippage_bps:.2f}, PriceOut={slipped_price:.4f}")
        return slipped_price

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    # Test avec la méthode 'percentage'
    sim_percentage = SlippageSimulator(method="percentage", percentage_max_bps=10.0, min_slippage_bps=1.0, max_slippage_bps=50.0)
    price_in = 100.0
    for _ in range(5):
        slipped_buy = sim_percentage.simulate_slippage(price_in, 1)
        slipped_sell = sim_percentage.simulate_slippage(price_in, -1)
        logger.debug(f"Percentage: Buy Price: {slipped_buy:.4f} (Expected > {price_in}), Sell Price: {slipped_sell:.4f} (Expected < {price_in})")

    # Test avec la méthode 'fixed_bps'
    sim_fixed = SlippageSimulator(method="fixed_bps", percentage_max_bps=5.0) # percentage_max_bps est utilisé comme la valeur fixe
    slipped_buy_fixed = sim_fixed.simulate_slippage(price_in, 1)
    slipped_sell_fixed = sim_fixed.simulate_slippage(price_in, -1)
    logger.debug(f"Fixed BPS: Buy Price: {slipped_buy_fixed:.4f}, Sell Price: {slipped_sell_fixed:.4f}")

    # Test avec min_slippage_bps > 0
    sim_min_guaranteed = SlippageSimulator(method="percentage", percentage_max_bps=0.0, min_slippage_bps=2.0, max_slippage_bps=5.0)
    slipped_buy_min = sim_min_guaranteed.simulate_slippage(price_in, 1)
    logger.debug(f"Min Guaranteed Slippage (2 BPS): Buy Price: {slipped_buy_min:.4f}") # Devrait être 100 * (1 + 2/10000) = 100.02

    # Test avec max_slippage_bps
    sim_max_capped = SlippageSimulator(method="percentage", percentage_max_bps=200.0, min_slippage_bps=1.0, max_slippage_bps=10.0)
    slipped_buy_capped_list = [sim_max_capped.simulate_slippage(price_in, 1) for _ in range(10)]
    logger.debug(f"Max Capped Slippage (10 BPS): Buy Prices: {[f'{p:.4f}' for p in slipped_buy_capped_list]}")
    # Tous devraient être <= 100 * (1 + 10/10000) = 100.10

    # Test avec 'none'
    sim_none = SlippageSimulator(method="none")
    slipped_none = sim_none.simulate_slippage(price_in, 1)
    logger.debug(f"No Slippage: Buy Price: {slipped_none:.4f} (Expected = {price_in})")
