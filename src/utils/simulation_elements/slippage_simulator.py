# src/utils/simulation_elements/slippage_simulator.py
"""
Ce module définit la classe SlippageSimulator, responsable de la simulation
du slippage sur les prix d'exécution des ordres de trading.
"""
import logging
import numpy as np # Pour np.random.uniform et autres opérations numériques
from typing import Optional

logger = logging.getLogger(__name__)

class SlippageSimulator:
    """
    Simule le slippage pour les ordres de trading en se basant sur différentes méthodes.
    """
    def __init__(self,
                 method: str = "percentage",
                 percentage_max_bps: Optional[float] = None,
                 volume_factor: Optional[float] = None,
                 volatility_factor: Optional[float] = None,
                 min_slippage_bps: float = 0.0,
                 max_slippage_bps: float = 0.10):
        """
        Initialise le simulateur de slippage.

        Args:
            method (str): Méthode de calcul du slippage. Options valides :
                          "percentage", "fixed_bps", "volume_based", "volatility_based", "none".
                          Par défaut "percentage".
            percentage_max_bps (Optional[float]): Pour la méthode 'percentage', le slippage maximum
                                                  en points de base (BPS) pour la plage aléatoire.
                                                  Pour la méthode 'fixed_bps', cette valeur est utilisée
                                                  comme le BPS fixe si fournie.
            volume_factor (Optional[float]): Facteur utilisé si method="volume_based".
            volatility_factor (Optional[float]): Facteur utilisé si method="volatility_based".
            min_slippage_bps (float): Slippage minimum garanti à appliquer, en BPS.
                                      Doit être >= 0. Par défaut 0.0.
            max_slippage_bps (float): Plafond absolu pour le slippage calculé, en BPS.
                                      Doit être >= min_slippage_bps. Par défaut 100.0.
        
        Raises:
            ValueError: Si les paramètres d'initialisation sont invalides.
        """
        self.method = method.lower().strip()
        self.log_prefix = f"[SlippageSimulator][{self.method}]"

        if self.method not in ["percentage", "fixed_bps", "volume_based", "volatility_based", "none"]:
            logger.error(f"{self.log_prefix} Méthode de slippage '{method}' non supportée.")
            raise ValueError(f"Méthode de slippage '{method}' non supportée.")

        if not isinstance(min_slippage_bps, (int, float)) or min_slippage_bps < 0:
            logger.error(f"{self.log_prefix} min_slippage_bps ({min_slippage_bps}) doit être un nombre non négatif.")
            raise ValueError("min_slippage_bps doit être un nombre non négatif.")
        self.min_slippage_bps = float(min_slippage_bps)

        if not isinstance(max_slippage_bps, (int, float)) or max_slippage_bps < self.min_slippage_bps:
            logger.error(f"{self.log_prefix} max_slippage_bps ({max_slippage_bps}) doit être un nombre >= min_slippage_bps ({self.min_slippage_bps}).")
            raise ValueError(f"max_slippage_bps ({max_slippage_bps}) doit être >= min_slippage_bps ({self.min_slippage_bps}).")
        self.max_slippage_bps = float(max_slippage_bps)

        # Validation et stockage de percentage_max_bps
        self.percentage_max_bps: Optional[float] = None
        if percentage_max_bps is not None:
            if not isinstance(percentage_max_bps, (int, float)) or percentage_max_bps < 0:
                logger.error(f"{self.log_prefix} percentage_max_bps ({percentage_max_bps}) doit être un nombre non négatif s'il est fourni.")
                raise ValueError("percentage_max_bps doit être un nombre non négatif s'il est fourni.")
            self.percentage_max_bps = float(percentage_max_bps)

        if self.method == "percentage":
            if self.percentage_max_bps is None:
                logger.error(f"{self.log_prefix} percentage_max_bps doit être fourni pour la méthode 'percentage'.")
                raise ValueError("percentage_max_bps doit être fourni pour la méthode 'percentage'.")
            if self.percentage_max_bps < self.min_slippage_bps:
                logger.warning(f"{self.log_prefix} percentage_max_bps ({self.percentage_max_bps}) est inférieur à min_slippage_bps ({self.min_slippage_bps}) "
                               f"pour la méthode 'percentage'. Ajustement de percentage_max_bps à {self.min_slippage_bps}.")
                self.percentage_max_bps = self.min_slippage_bps # Assure que la plage aléatoire est valide

        # Validation et stockage des facteurs optionnels
        self.volume_factor: Optional[float] = None
        if volume_factor is not None:
            if not isinstance(volume_factor, (int, float)) or volume_factor < 0:
                logger.error(f"{self.log_prefix} volume_factor ({volume_factor}) doit être un nombre non négatif s'il est fourni.")
                raise ValueError("volume_factor doit être un nombre non négatif s'il est fourni.")
            self.volume_factor = float(volume_factor)
        
        self.volatility_factor: Optional[float] = None
        if volatility_factor is not None:
            if not isinstance(volatility_factor, (int, float)) or volatility_factor < 0:
                logger.error(f"{self.log_prefix} volatility_factor ({volatility_factor}) doit être un nombre non négatif s'il est fourni.")
                raise ValueError("volatility_factor doit être un nombre non négatif s'il est fourni.")
            self.volatility_factor = float(volatility_factor)

        logger.info(f"{self.log_prefix} Initialisé. Méthode: {self.method}, "
                    f"MinBPS: {self.min_slippage_bps}, MaxBPS: {self.max_slippage_bps}, "
                    f"PercMaxBPS: {self.percentage_max_bps}, VolFactor: {self.volume_factor}, VolaFactor: {self.volatility_factor}")

    def simulate_slippage(self,
                          price: float,
                          direction: int,
                          volume_at_price: Optional[float] = None,
                          volatility: Optional[float] = None) -> float:
        """
        Simule le slippage sur un prix donné.

        Args:
            price (float): Le prix théorique d'exécution avant slippage.
            direction (int): Direction du trade initiant le slippage.
                             1 si l'ordre est un achat (le prix d'exécution augmente).
                             -1 si l'ordre est une vente (le prix d'exécution baisse).
            volume_at_price (Optional[float]): Volume disponible au niveau de `price`
                                               (utilisé pour la méthode `volume_based`).
            volatility (Optional[float]): Mesure de la volatilité récente (ex: ATR en pourcentage du prix)
                                          (utilisé pour la méthode `volatility_based`).

        Returns:
            float: Le prix ajusté après simulation du slippage.
        """
        current_log_prefix = f"{self.log_prefix}[Simulate(P:{price:.4f},D:{direction})]"

        if self.method == "none" or price <= 0:
            logger.debug(f"{current_log_prefix} Pas de slippage appliqué (méthode 'none' ou prix invalide).")
            return price

        slippage_bps_calculated: float = 0.0

        if self.method == "percentage":
            if self.percentage_max_bps is None: # Devrait être attrapé par __init__ mais par sécurité
                logger.error(f"{current_log_prefix} percentage_max_bps est None pour la méthode 'percentage'. Application de min_slippage_bps.")
                slippage_bps_calculated = self.min_slippage_bps
            else:
                # La borne supérieure pour la distribution uniforme est le minimum entre
                # le max configuré pour le pourcentage et le plafond global de slippage.
                # Cette borne doit aussi être au moins égale à min_slippage_bps (assuré par __init__).
                upper_bound_for_dist = min(self.percentage_max_bps, self.max_slippage_bps)
                # min_slippage_bps est déjà <= percentage_max_bps (par __init__)
                # et min_slippage_bps est déjà <= max_slippage_bps (par __init__)
                # donc min_slippage_bps <= upper_bound_for_dist est garanti.
                slippage_bps_calculated = np.random.uniform(self.min_slippage_bps, upper_bound_for_dist)
                logger.debug(f"{current_log_prefix} Méthode 'percentage'. Range: [{self.min_slippage_bps}, {upper_bound_for_dist}]. Slippage BPS brut: {slippage_bps_calculated:.2f}")
        
        elif self.method == "fixed_bps":
            # Utilise percentage_max_bps comme valeur fixe, ou min_slippage_bps si non fourni.
            fixed_value_to_use = self.percentage_max_bps if self.percentage_max_bps is not None else self.min_slippage_bps
            slippage_bps_calculated = fixed_value_to_use
            logger.debug(f"{current_log_prefix} Méthode 'fixed_bps'. Valeur BPS fixe utilisée: {slippage_bps_calculated:.2f}")

        elif self.method == "volume_based":
            if self.volume_factor is not None and volume_at_price is not None and volume_at_price > 1e-9: # Éviter division par zéro
                # Modèle simple : slippage inversement proportionnel au volume, ajusté par un facteur.
                # Le facteur doit être calibré pour donner des BPS raisonnables.
                # Ex: si volume_factor = 1000 et volume_at_price = 100 => 10 BPS.
                slippage_bps_calculated = (1.0 / volume_at_price) * self.volume_factor * 100.0 # Ajustement pour BPS (facteur * 10000 / 100)
                logger.debug(f"{current_log_prefix} Méthode 'volume_based'. Vol@{price}: {volume_at_price}, Factor: {self.volume_factor}. Slippage BPS brut: {slippage_bps_calculated:.2f}")
            else:
                logger.debug(f"{current_log_prefix} Méthode 'volume_based'. Données de volume ou facteur manquants/invalides. Utilisation de min_slippage_bps.")
                slippage_bps_calculated = self.min_slippage_bps

        elif self.method == "volatility_based":
            if self.volatility_factor is not None and volatility is not None and volatility > 0:
                # Modèle simple : slippage proportionnel à la volatilité (en pourcentage) et au facteur.
                # Ex: volatilité = 0.01 (1%), factor = 50 => 0.01 * 50 = 0.5% = 50 BPS.
                slippage_bps_calculated = volatility * self.volatility_factor * 10000.0 # volatility (ex: 0.01) * factor * 10000 (pour BPS)
                logger.debug(f"{current_log_prefix} Méthode 'volatility_based'. Volatilité: {volatility:.4f}, Factor: {self.volatility_factor}. Slippage BPS brut: {slippage_bps_calculated:.2f}")
            else:
                logger.debug(f"{current_log_prefix} Méthode 'volatility_based'. Données de volatilité ou facteur manquants/invalides. Utilisation de min_slippage_bps.")
                slippage_bps_calculated = self.min_slippage_bps
        
        else: # Méthode inconnue (ne devrait pas arriver si __init__ valide bien)
            logger.warning(f"{current_log_prefix} Méthode de slippage inconnue '{self.method}'. Application de min_slippage_bps.")
            slippage_bps_calculated = self.min_slippage_bps

        # Plafonner le slippage calculé par max_slippage_bps et s'assurer qu'il est au moins min_slippage_bps.
        final_slippage_bps = max(self.min_slippage_bps, min(slippage_bps_calculated, self.max_slippage_bps))
        if abs(final_slippage_bps - slippage_bps_calculated) > 1e-9 : # Si un clamping a eu lieu
            logger.debug(f"{current_log_prefix} Slippage BPS clampé. Brut: {slippage_bps_calculated:.2f}, MinConfig: {self.min_slippage_bps}, MaxConfig: {self.max_slippage_bps}. Final BPS: {final_slippage_bps:.2f}")


        # Convertir BPS en décimal
        slippage_decimal = final_slippage_bps / 10000.0  # Ex: 10 BPS = 0.001

        # Appliquer le slippage de manière défavorable
        slipped_price: float
        if direction == 1:  # Achat, le prix d'exécution augmente (défavorable)
            slipped_price = price * (1 + slippage_decimal)
        elif direction == -1:  # Vente, le prix d'exécution baisse (défavorable)
            slipped_price = price * (1 - slippage_decimal)
        else: # Direction neutre ou inconnue (0), pas de slippage directionnel
            logger.warning(f"{current_log_prefix} Direction de trade invalide ou neutre ({direction}). Aucun slippage directionnel appliqué.")
            slipped_price = price
        
        logger.debug(f"{current_log_prefix} Prix original: {price:.4f}, Slippage BPS appliqué: {final_slippage_bps:.2f} ({slippage_decimal:.6f}), "
                     f"Prix avec slippage: {slipped_price:.4f}")
        return slipped_price

if __name__ == '__main__':
    # Configuration du logging pour les tests directs de ce module
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("--- Test de SlippageSimulator ---")
    prix_initial = 100.0

    # Test méthode "percentage"
    sim_percent = SlippageSimulator(method="percentage", percentage_max_bps=20.0, min_slippage_bps=5.0, max_slippage_bps=50.0)
    logger.info(f"\nTest 'percentage' (min:5, perc_max:20, global_max:50) sur prix {prix_initial}:")
    for i in range(3):
        achat_p = sim_percent.simulate_slippage(prix_initial, 1)
        vente_p = sim_percent.simulate_slippage(prix_initial, -1)
        logger.info(f"  Essai {i+1}: Achat -> {achat_p:.4f} (attendu > {prix_initial}), Vente -> {vente_p:.4f} (attendu < {prix_initial})")
        assert prix_initial * (1 + 5/10000) <= achat_p <= prix_initial * (1 + 20/10000)
        assert prix_initial * (1 - 20/10000) <= vente_p <= prix_initial * (1 - 5/10000)


    # Test méthode "fixed_bps"
    sim_fixed_1 = SlippageSimulator(method="fixed_bps", percentage_max_bps=10.0, min_slippage_bps=0.0, max_slippage_bps=50.0)
    achat_f1 = sim_fixed_1.simulate_slippage(prix_initial, 1)
    expected_f1 = prix_initial * (1 + 10.0/10000.0)
    logger.info(f"\nTest 'fixed_bps' (perc_max_bps=10.0) -> Achat: {achat_f1:.4f} (Attendu: {expected_f1:.4f})")
    assert abs(achat_f1 - expected_f1) < 1e-9

    sim_fixed_2 = SlippageSimulator(method="fixed_bps", percentage_max_bps=None, min_slippage_bps=2.0, max_slippage_bps=50.0)
    achat_f2 = sim_fixed_2.simulate_slippage(prix_initial, 1)
    expected_f2 = prix_initial * (1 + 2.0/10000.0) # Doit utiliser min_slippage_bps
    logger.info(f"Test 'fixed_bps' (perc_max_bps=None, min_bps=2.0) -> Achat: {achat_f2:.4f} (Attendu: {expected_f2:.4f})")
    assert abs(achat_f2 - expected_f2) < 1e-9
    
    sim_fixed_3 = SlippageSimulator(method="fixed_bps", percentage_max_bps=1.0, min_slippage_bps=5.0, max_slippage_bps=50.0)
    achat_f3 = sim_fixed_3.simulate_slippage(prix_initial, 1)
    expected_f3 = prix_initial * (1 + 5.0/10000.0) # Doit être clampé par min_slippage_bps
    logger.info(f"Test 'fixed_bps' (perc_max_bps=1.0, min_bps=5.0) -> Achat: {achat_f3:.4f} (Attendu: {expected_f3:.4f})")
    assert abs(achat_f3 - expected_f3) < 1e-9

    sim_fixed_4 = SlippageSimulator(method="fixed_bps", percentage_max_bps=60.0, min_slippage_bps=0.0, max_slippage_bps=50.0)
    achat_f4 = sim_fixed_4.simulate_slippage(prix_initial, 1)
    expected_f4 = prix_initial * (1 + 50.0/10000.0) # Doit être clampé par max_slippage_bps
    logger.info(f"Test 'fixed_bps' (perc_max_bps=60.0, max_bps=50.0) -> Achat: {achat_f4:.4f} (Attendu: {expected_f4:.4f})")
    assert abs(achat_f4 - expected_f4) < 1e-9


    # Test méthode "volume_based"
    sim_vol = SlippageSimulator(method="volume_based", volume_factor=2000.0, min_slippage_bps=1.0, max_slippage_bps=25.0) # factor * 100
    achat_v1 = sim_vol.simulate_slippage(prix_initial, 1, volume_at_price=100.0) # 2000/100 * 100 = 2000 BPS -> clampé à 25 BPS
    expected_v1 = prix_initial * (1 + 25.0/10000.0)
    logger.info(f"\nTest 'volume_based' (vol=100, factor=2000, max=25) -> Achat: {achat_v1:.4f} (Attendu: {expected_v1:.4f} car clampé par max_slippage_bps)")
    assert abs(achat_v1 - expected_v1) < 1e-9
    
    achat_v2 = sim_vol.simulate_slippage(prix_initial, 1, volume_at_price=100000.0) # 2000/100000 * 100 = 2 BPS. min_slippage_bps est 1.0
    expected_v2 = prix_initial * (1 + 2.0/10000.0)
    logger.info(f"Test 'volume_based' (vol=100000, factor=2000, min=1.0) -> Achat: {achat_v2:.4f} (Attendu: {expected_v2:.4f})")
    assert abs(achat_v2 - expected_v2) < 1e-9

    # Test méthode "volatility_based"
    sim_vola = SlippageSimulator(method="volatility_based", volatility_factor=100.0, min_slippage_bps=2.0, max_slippage_bps=30.0) # factor * 10000
    achat_vola1 = sim_vola.simulate_slippage(prix_initial, 1, volatility=0.001) # 0.001 * 100 * 10000 = 1000 BPS -> clampé à 30 BPS
    expected_vola1 = prix_initial * (1 + 30.0/10000.0)
    logger.info(f"\nTest 'volatility_based' (vola=0.1%, factor=100, max=30) -> Achat: {achat_vola1:.4f} (Attendu: {expected_vola1:.4f} car clampé par max_slippage_bps)")
    assert abs(achat_vola1 - expected_vola1) < 1e-9

    achat_vola2 = sim_vola.simulate_slippage(prix_initial, 1, volatility=0.00001) # 0.00001 * 100 * 10000 = 1 BPS -> clampé par min à 2 BPS
    expected_vola2 = prix_initial * (1 + 2.0/10000.0)
    logger.info(f"Test 'volatility_based' (vola=0.001%, factor=100, min=2.0) -> Achat: {achat_vola2:.4f} (Attendu: {expected_vola2:.4f} car clampé par min_slippage_bps)")
    assert abs(achat_vola2 - expected_vola2) < 1e-9

    # Test méthode "none"
    sim_none = SlippageSimulator(method="none")
    achat_none = sim_none.simulate_slippage(prix_initial, 1)
    logger.info(f"\nTest 'none' -> Achat: {achat_none:.4f} (Attendu: {prix_initial:.4f})")
    assert abs(achat_none - prix_initial) < 1e-9

    # Test d'initialisation invalide
    try:
        SlippageSimulator(min_slippage_bps=-1)
        assert False, "ValueError non levée pour min_slippage_bps négatif"
    except ValueError: logger.info("Test d'erreur (min_slippage_bps < 0) RÉUSSI.")
    try:
        SlippageSimulator(max_slippage_bps=5, min_slippage_bps=10)
        assert False, "ValueError non levée pour max_slippage_bps < min_slippage_bps"
    except ValueError: logger.info("Test d'erreur (max_slippage_bps < min_slippage_bps) RÉUSSI.")
    try:
        SlippageSimulator(method="percentage", percentage_max_bps=None)
        assert False, "ValueError non levée pour percentage_max_bps=None avec méthode 'percentage'"
    except ValueError: logger.info("Test d'erreur (percentage_max_bps=None pour méthode 'percentage') RÉUSSI.")
    
    logger.info("--- Tests de SlippageSimulator terminés ---")

