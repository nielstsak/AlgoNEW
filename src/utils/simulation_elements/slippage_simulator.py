# src/utils/simulation_elements/slippage_simulator.py
"""
Ce module définit la classe SlippageSimulator, responsable de la simulation
du slippage sur les prix d'exécution des ordres de trading.
"""
import logging
import numpy as np # Pour np.random.uniform et autres opérations numériques
from typing import Optional, Union

logger = logging.getLogger(__name__)

class SlippageSimulator:
    """
    Simule le slippage pour les ordres de trading en se basant sur différentes méthodes.
    """
    def __init__(self,
                 method: str = "percentage",
                 percentage_max_bps: Optional[Union[float, int]] = None,
                 volume_factor: Optional[Union[float, int]] = None,
                 volatility_factor: Optional[Union[float, int]] = None,
                 min_slippage_bps: Union[float, int] = 0.0,
                 max_slippage_bps: Union[float, int] = 100.0): # Binance max slippage is often around 50-100 BPS for liquid pairs.
        """
        Initialise le simulateur de slippage.

        Args:
            method (str): Méthode de calcul du slippage. Options valides :
                          "percentage", "fixed_bps", "volume_based", "volatility_based", "none".
                          Par défaut "percentage".
            percentage_max_bps (Optional[Union[float, int]]): Pour la méthode 'percentage', le slippage maximum
                                                  en points de base (BPS) pour la plage aléatoire.
                                                  Pour la méthode 'fixed_bps', cette valeur est utilisée
                                                  comme le BPS fixe si fournie.
            volume_factor (Optional[Union[float, int]]): Facteur utilisé si method="volume_based".
                                               Un facteur plus élevé signifie plus de slippage pour un volume donné.
            volatility_factor (Optional[Union[float, int]]): Facteur utilisé si method="volatility_based".
                                                   Un facteur plus élevé signifie plus de slippage pour une volatilité donnée.
            min_slippage_bps (Union[float, int]): Slippage minimum garanti à appliquer, en BPS.
                                      Doit être >= 0. Par défaut 0.0.
            max_slippage_bps (Union[float, int]): Plafond absolu pour le slippage calculé, en BPS.
                                      Doit être >= min_slippage_bps. Par défaut 100.0.
        
        Raises:
            ValueError: Si les paramètres d'initialisation sont invalides.
        """
        self.method = method.lower().strip()
        self.log_prefix = f"[SlippageSim][{self.method}]"

        if self.method not in ["percentage", "fixed_bps", "volume_based", "volatility_based", "none"]:
            msg = f"Méthode de slippage '{method}' non supportée. Valides: 'percentage', 'fixed_bps', 'volume_based', 'volatility_based', 'none'."
            logger.error(f"{self.log_prefix} {msg}")
            raise ValueError(msg)

        if not isinstance(min_slippage_bps, (int, float)) or min_slippage_bps < 0:
            msg = f"min_slippage_bps ({min_slippage_bps}) doit être un nombre non négatif."
            logger.error(f"{self.log_prefix} {msg}")
            raise ValueError(msg)
        self.min_slippage_bps = float(min_slippage_bps)

        if not isinstance(max_slippage_bps, (int, float)) or max_slippage_bps < self.min_slippage_bps:
            msg = f"max_slippage_bps ({max_slippage_bps}) doit être un nombre >= min_slippage_bps ({self.min_slippage_bps})."
            logger.error(f"{self.log_prefix} {msg}")
            raise ValueError(msg)
        self.max_slippage_bps = float(max_slippage_bps)

        self.percentage_max_bps: Optional[float] = None
        if percentage_max_bps is not None:
            if not isinstance(percentage_max_bps, (int, float)) or percentage_max_bps < 0:
                msg = f"percentage_max_bps ({percentage_max_bps}) doit être un nombre non négatif s'il est fourni."
                logger.error(f"{self.log_prefix} {msg}")
                raise ValueError(msg)
            self.percentage_max_bps = float(percentage_max_bps)

        if self.method == "percentage":
            if self.percentage_max_bps is None:
                msg = "percentage_max_bps doit être fourni pour la méthode 'percentage'."
                logger.error(f"{self.log_prefix} {msg}")
                raise ValueError(msg)
            if self.percentage_max_bps < self.min_slippage_bps:
                logger.warning(f"{self.log_prefix} percentage_max_bps ({self.percentage_max_bps}) est < min_slippage_bps ({self.min_slippage_bps}). "
                               f"Ajustement de percentage_max_bps à {self.min_slippage_bps} pour assurer une plage valide.")
                self.percentage_max_bps = self.min_slippage_bps

        self.volume_factor: Optional[float] = None
        if volume_factor is not None:
            if not isinstance(volume_factor, (int, float)) or volume_factor < 0:
                msg = f"volume_factor ({volume_factor}) doit être un nombre non négatif s'il est fourni."
                logger.error(f"{self.log_prefix} {msg}")
                raise ValueError(msg)
            self.volume_factor = float(volume_factor)
        
        self.volatility_factor: Optional[float] = None
        if volatility_factor is not None:
            if not isinstance(volatility_factor, (int, float)) or volatility_factor < 0:
                msg = f"volatility_factor ({volatility_factor}) doit être un nombre non négatif s'il est fourni."
                logger.error(f"{self.log_prefix} {msg}")
                raise ValueError(msg)
            self.volatility_factor = float(volatility_factor)

        logger.info(f"{self.log_prefix} Initialisé. MinBPS: {self.min_slippage_bps:.2f}, MaxBPS: {self.max_slippage_bps:.2f}, "
                    f"PercMaxBPS: {self.percentage_max_bps if self.percentage_max_bps is not None else 'N/A'}, "
                    f"VolFactor: {self.volume_factor if self.volume_factor is not None else 'N/A'}, "
                    f"VolaFactor: {self.volatility_factor if self.volatility_factor is not None else 'N/A'}")

    def simulate_slippage(self,
                          price: float,
                          direction: int, # 1 pour achat (le prix slippé augmente), -1 pour vente (le prix slippé baisse)
                          order_book_depth_at_price: Optional[float] = None, # Volume disponible au niveau de `price`
                          market_volatility_pct: Optional[float] = None # Ex: ATR en % du prix, ou écart-type des rendements
                         ) -> float:
        """
        Simule le slippage sur un prix donné. Le slippage est toujours défavorable.

        Args:
            price (float): Le prix théorique d'exécution avant slippage.
            direction (int): Direction de l'ordre initiant le slippage.
                             1 si l'ordre est un achat (le prix d'exécution augmente).
                             -1 si l'ordre est une vente (le prix d'exécution baisse).
            order_book_depth_at_price (Optional[float]): Volume disponible au niveau de `price`
                                               (utilisé pour la méthode `volume_based`).
            market_volatility_pct (Optional[float]): Mesure de la volatilité récente en pourcentage
                                          (ex: ATR / prix, doit être > 0).
                                          (utilisé pour la méthode `volatility_based`).

        Returns:
            float: Le prix ajusté après simulation du slippage.
        """
        current_log_prefix = f"{self.log_prefix}[Simulate(P:{price:.4f},D:{direction})]"

        if self.method == "none" or price <= 1e-9: # 1e-9 pour éviter les prix nuls/négatifs
            logger.debug(f"{current_log_prefix} Pas de slippage appliqué (méthode 'none' ou prix invalide).")
            return price
        
        if direction not in [1, -1]:
            logger.warning(f"{current_log_prefix} Direction d'ordre invalide ({direction}). Doit être 1 (achat) ou -1 (vente). "
                           "Aucun slippage ne sera appliqué.")
            return price

        slippage_bps_calculated: float = 0.0

        if self.method == "percentage":
            if self.percentage_max_bps is None: # Devrait être attrapé par __init__
                logger.error(f"{current_log_prefix} percentage_max_bps est None pour méthode 'percentage'. Utilisation de min_slippage_bps.")
                slippage_bps_calculated = self.min_slippage_bps
            else:
                # La borne supérieure pour la distribution est min(config_perc_max, global_max_cap)
                # et doit être >= min_slippage_bps (assuré par __init__)
                upper_bound_for_random_dist = min(self.percentage_max_bps, self.max_slippage_bps)
                slippage_bps_calculated = np.random.uniform(self.min_slippage_bps, upper_bound_for_random_dist)
                logger.debug(f"{current_log_prefix} Méthode 'percentage'. Range: [{self.min_slippage_bps:.2f}, {upper_bound_for_random_dist:.2f}]. Slippage BPS brut: {slippage_bps_calculated:.2f}")
        
        elif self.method == "fixed_bps":
            fixed_val_to_use = self.percentage_max_bps if self.percentage_max_bps is not None else self.min_slippage_bps
            slippage_bps_calculated = fixed_val_to_use
            logger.debug(f"{current_log_prefix} Méthode 'fixed_bps'. Valeur BPS fixe utilisée: {slippage_bps_calculated:.2f}")

        elif self.method == "volume_based":
            if self.volume_factor is not None and order_book_depth_at_price is not None and order_book_depth_at_price > 1e-9:
                # Modèle simple : slippage inversement proportionnel à la profondeur, ajusté par un facteur.
                # Ex: si volume_factor = 1000 et order_book_depth = 100 (unités de base) => 10 BPS.
                # Le facteur doit être calibré pour donner des BPS raisonnables.
                # Slippage BPS = (1 / profondeur) * facteur_volume
                slippage_bps_calculated = (1.0 / order_book_depth_at_price) * self.volume_factor
                logger.debug(f"{current_log_prefix} Méthode 'volume_based'. Profondeur: {order_book_depth_at_price}, Factor: {self.volume_factor}. Slippage BPS brut: {slippage_bps_calculated:.2f}")
            else:
                logger.debug(f"{current_log_prefix} Méthode 'volume_based'. Données de profondeur ou facteur manquants/invalides. Utilisation de min_slippage_bps.")
                slippage_bps_calculated = self.min_slippage_bps

        elif self.method == "volatility_based":
            if self.volatility_factor is not None and market_volatility_pct is not None and market_volatility_pct > 1e-9:
                # Modèle simple : slippage proportionnel à la volatilité (en pourcentage) et au facteur.
                # Ex: volatilité = 0.01 (1%), factor = 50 => 0.01 * 50 = 0.5% = 50 BPS.
                # Slippage BPS = volatilité_pct * facteur_volatilité * 100 (pour convertir % en BPS si volatilité_pct est ex: 0.01 pour 1%)
                # Si market_volatility_pct est déjà en BPS (ex: ATR en BPS), alors pas besoin de *100.
                # Supposons market_volatility_pct est un ratio (ex: 0.01 pour 1%).
                slippage_bps_calculated = market_volatility_pct * self.volatility_factor * 100.0 # Convertir le produit en BPS
                logger.debug(f"{current_log_prefix} Méthode 'volatility_based'. Volatilité %: {market_volatility_pct:.4f}, Factor: {self.volatility_factor}. Slippage BPS brut: {slippage_bps_calculated:.2f}")
            else:
                logger.debug(f"{current_log_prefix} Méthode 'volatility_based'. Données de volatilité ou facteur manquants/invalides. Utilisation de min_slippage_bps.")
                slippage_bps_calculated = self.min_slippage_bps
        
        else: # Méthode inconnue (ne devrait pas arriver si __init__ valide bien)
            logger.warning(f"{current_log_prefix} Méthode de slippage inconnue '{self.method}'. Application de min_slippage_bps.")
            slippage_bps_calculated = self.min_slippage_bps

        # Plafonner le slippage calculé par max_slippage_bps et s'assurer qu'il est au moins min_slippage_bps.
        final_slippage_bps = max(self.min_slippage_bps, min(slippage_bps_calculated, self.max_slippage_bps))
        if abs(final_slippage_bps - slippage_bps_calculated) > 1e-9 : # Si un clamping a eu lieu
            logger.debug(f"{current_log_prefix} Slippage BPS clampé. Brut: {slippage_bps_calculated:.2f}, MinConfig: {self.min_slippage_bps:.2f}, MaxConfig: {self.max_slippage_bps:.2f}. Final BPS: {final_slippage_bps:.2f}")

        # Convertir BPS en taux décimal
        slippage_decimal_rate = final_slippage_bps / 10000.0  # Ex: 10 BPS = 0.001

        # Appliquer le slippage de manière défavorable
        slipped_price: float
        if direction == 1:  # Achat, le prix d'exécution augmente (défavorable)
            slipped_price = price * (1 + slippage_decimal_rate)
        elif direction == -1:  # Vente, le prix d'exécution baisse (défavorable)
            slipped_price = price * (1 - slippage_decimal_rate)
        else: # Ne devrait pas arriver si direction est validée
            slipped_price = price 
        
        logger.debug(f"{current_log_prefix} Prix original: {price:.4f}, Slippage BPS appliqué: {final_slippage_bps:.2f} (Taux: {slippage_decimal_rate:.6f}), "
                     f"Prix avec slippage: {slipped_price:.4f}")
        return slipped_price

