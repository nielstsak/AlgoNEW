# test_strategy_format.py
"""
Script CLI pour tester le format et la conformité des classes de stratégies de trading
par rapport à src.strategies.base.BaseStrategy.
"""

import argparse
import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, cast

# --- Configuration initiale du logging et du PYTHONPATH ---
# Le logging sera reconfiguré par load_all_configs si nécessaire pour l'instanciation.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path(".").resolve()

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
    logger.debug(f"Ajouté {SRC_PATH} au PYTHONPATH par test_strategy_format.py")

# --- Imports des modules de l'application ---
try:
    from src.strategies.base import BaseStrategy
    from src.config.loader import load_all_configs, AppConfig
    # Importer pandas pour vérifier le type pd.DataFrame dans les signatures
    import pandas as pd
except ImportError as e:
    logger.critical(f"ÉCHEC CRITIQUE (test_strategy_format.py): Impossible d'importer les modules de base: {e}. "
                    f"Vérifiez PYTHONPATH et les installations. CWD: {Path.cwd()}, sys.path: {sys.path}", exc_info=True)
    sys.exit(2) # Code de sortie différent pour les erreurs d'import critiques

# Définition des signatures attendues pour les méthodes abstraites
# Utilisera inspect.signature pour comparer. Les annotations de type sont importantes.
# Note: typing.Dict et typing.List ne sont pas directement comparables avec isinstance,
# on vérifiera la présence des annotations plutôt que l'égalité exacte des types complexes.
EXPECTED_METHOD_SIGNATURES = {
    "_validate_params": inspect.Signature(parameters=[
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ], return_annotation=None), # ou inspect.Signature.empty si on ne type pas le retour None
    "get_required_indicator_configs": inspect.Signature(parameters=[
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ], return_annotation=List[Dict[str, Any]]),
    "_calculate_indicators": inspect.Signature(parameters=[
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("data_feed", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=pd.DataFrame)
    ], return_annotation=pd.DataFrame),
    "_generate_signals": inspect.Signature(parameters=[
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("data_with_indicators", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=pd.DataFrame),
        inspect.Parameter("current_position_open", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=bool),
        inspect.Parameter("current_position_direction", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
        inspect.Parameter("current_entry_price", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=float)
    ], return_annotation=Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]),
    "generate_order_request": inspect.Signature(parameters=[
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("data", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=pd.DataFrame),
        inspect.Parameter("current_position", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
        inspect.Parameter("available_capital", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=float),
        inspect.Parameter("symbol_info", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=dict) # ou Dict[str, Any]
    ], return_annotation=Optional[Tuple[Dict[str, Any], Dict[str, float]]])
}


def compare_signatures(actual_sig: inspect.Signature, expected_sig: inspect.Signature, method_name: str) -> List[str]:
    """
    Compare deux signatures de méthode.
    Pour l'instant, vérifie le nombre de paramètres et leur nom.
    La vérification des annotations de type peut être complexe et est simplifiée.
    """
    errors: List[str] = []
    actual_params = list(actual_sig.parameters.values())
    expected_params = list(expected_sig.parameters.values())

    if len(actual_params) != len(expected_params):
        errors.append(f"Nombre d'arguments incorrect. Attendu: {len(expected_params)}, Reçu: {len(actual_params)}")
    else:
        for i, (actual_p, expected_p) in enumerate(zip(actual_params, expected_params)):
            if actual_p.name != expected_p.name:
                errors.append(f"Nom de l'argument {i} incorrect. Attendu: '{expected_p.name}', Reçu: '{actual_p.name}'")
            # Vérification simplifiée des annotations (présence plutôt qu'égalité exacte pour les types complexes)
            if expected_p.annotation is not inspect.Parameter.empty and \
               actual_p.annotation is inspect.Parameter.empty:
                errors.append(f"Annotation de type manquante pour l'argument '{actual_p.name}'. Attendu (ou similaire): {expected_p.annotation}")
            # Note: Comparer les annotations de type complexes (List[Dict[...]]) est difficile dynamiquement.

    # Vérification de l'annotation de retour (présence)
    if expected_sig.return_annotation is not inspect.Signature.empty and \
       actual_sig.return_annotation is inspect.Signature.empty and \
       expected_sig.return_annotation is not None: # Ne pas se plaindre si l'attendu est None et le reçu est empty
        errors.append(f"Annotation de retour manquante. Attendu (ou similaire): {expected_sig.return_annotation}")
    
    if errors:
        logger.debug(f"Détails de la signature pour {method_name}:")
        logger.debug(f"  Attendue: {expected_sig}")
        logger.debug(f"  Reçue   : {actual_sig}")
    return errors

def find_strategy_class_in_module(module: Any) -> Optional[Type[BaseStrategy]]:
    """
    Trouve la première classe dans un module qui hérite de BaseStrategy.
    """
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
            return cast(Type[BaseStrategy], obj)
    return None

def get_default_params_for_strategy(
    app_config: AppConfig,
    strategy_class_name: str,
    strategy_key_in_config: Optional[str] = None
) -> Dict[str, Any]:
    """
    Tente de récupérer les default_params pour une stratégie depuis AppConfig.
    """
    if not strategy_key_in_config:
        # Essayer de déduire la clé de config à partir du nom de la classe
        # Ex: MaCrossoverStrategy -> ma_crossover_strategy
        parts = re.findall('[A-Z][^A-Z]*', strategy_class_name.replace("Strategy", ""))
        strategy_key_in_config = "_".join(parts).lower() if parts else strategy_class_name.lower()

    if app_config.strategies_config and app_config.strategies_config.strategies:
        strat_cfg_entry = app_config.strategies_config.strategies.get(strategy_key_in_config)
        if strat_cfg_entry and strat_cfg_entry.default_params:
            logger.info(f"Utilisation des default_params trouvés dans config_strategies.json pour '{strategy_key_in_config}'.")
            return strat_cfg_entry.default_params.copy()
        else:
            logger.warning(f"Aucun default_params trouvé pour la clé de stratégie '{strategy_key_in_config}' dans config_strategies.json.")
    else:
        logger.warning("strategies_config non trouvé ou vide dans AppConfig.")
    return {}


def main():
    """
    Point d'entrée principal pour le script de test de format de stratégie.
    """
    parser = argparse.ArgumentParser(description="Teste le format et la conformité d'une classe de stratégie.")
    parser.add_argument(
        "--strategy_file",
        type=str,
        required=True,
        help="Chemin relatif (depuis la racine du projet) vers le fichier Python de la stratégie (ex: src/strategies/ma_crossover_strategy.py)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Chemin vers la racine du projet si différent du répertoire parent de ce script."
    )
    args = parser.parse_args()

    project_root_path = Path(args.root).resolve() if args.root else PROJECT_ROOT
    strategy_file_rel_path = Path(args.strategy_file)
    
    if not strategy_file_rel_path.is_absolute():
        strategy_file_abs_path = (project_root_path / strategy_file_rel_path).resolve()
    else:
        strategy_file_abs_path = strategy_file_rel_path

    logger.info(f"--- Test de Conformité de Stratégie ---")
    logger.info(f"Racine du projet utilisée : {project_root_path}")
    logger.info(f"Fichier de stratégie à tester : {strategy_file_abs_path}")

    if not strategy_file_abs_path.exists():
        logger.critical(f"Le fichier de stratégie spécifié n'existe pas : {strategy_file_abs_path}")
        sys.exit(1)

    # Convertir le chemin du fichier en nom de module importable
    # ex: C:\...\project\src\strategies\my_strat.py -> src.strategies.my_strat
    try:
        module_import_str = str(strategy_file_abs_path.relative_to(project_root_path).with_suffix('')).replace(os.sep, '.')
    except ValueError:
        logger.critical(f"Le fichier de stratégie {strategy_file_abs_path} ne semble pas être sous la racine du projet {project_root_path}. "
                        "Impossible de déterminer le chemin d'import.")
        sys.exit(1)

    test_results: Dict[str, Union[str, List[str]]] = {}
    exit_code = 0

    # 2. Chargement Dynamique
    strategy_module: Optional[Any] = None
    strategy_class: Optional[Type[BaseStrategy]] = None
    try:
        logger.info(f"Tentative d'importation du module : {module_import_str}")
        strategy_module = importlib.import_module(module_import_str)
        test_results["Chargement du Module"] = "OK"
        
        strategy_class = find_strategy_class_in_module(strategy_module)
        if strategy_class:
            logger.info(f"Classe de stratégie trouvée : {strategy_class.__name__}")
            test_results["Détection de la Classe"] = f"OK ({strategy_class.__name__})"
        else:
            logger.error(f"Aucune classe héritant de BaseStrategy trouvée dans {module_import_str}.")
            test_results["Détection de la Classe"] = "ÉCHEC : Aucune classe BaseStrategy trouvée."
            exit_code = 1
    except ImportError as e_import:
        logger.critical(f"Échec de l'importation du module de stratégie '{module_import_str}': {e_import}", exc_info=True)
        test_results["Chargement du Module"] = f"ÉCHEC : {e_import}"
        exit_code = 1
    except Exception as e_load: # pylint: disable=broad-except
        logger.critical(f"Erreur inattendue lors du chargement du module ou de la classe : {e_load}", exc_info=True)
        test_results["Chargement Dynamique"] = f"ÉCHEC : {e_load}"
        exit_code = 1

    if not strategy_class:
        logger.error("Impossible de continuer les tests sans une classe de stratégie valide.")
        # Afficher les résultats partiels et sortir
        print("\n--- Résultats des Tests de Conformité ---")
        for check_name, result_val in test_results.items():
            print(f"- {check_name}: {result_val}")
        sys.exit(exit_code if exit_code != 0 else 1)

    # 3. Vérifications de Conformité
    # Héritage
    if issubclass(strategy_class, BaseStrategy):
        test_results["Héritage de BaseStrategy"] = "OK"
    else:
        test_results["Héritage de BaseStrategy"] = "ÉCHEC : N'hérite pas de BaseStrategy."
        exit_code = 1

    # Méthodes Abstraites
    for method_name, expected_sig in EXPECTED_METHOD_SIGNATURES.items():
        check_name_method = f"Méthode '{method_name}'"
        if not hasattr(strategy_class, method_name):
            test_results[check_name_method] = "ÉCHEC : Manquante."
            exit_code = 1
            continue
        
        method_obj = getattr(strategy_class, method_name)
        if not callable(method_obj):
            test_results[check_name_method] = f"ÉCHEC : '{method_name}' n'est pas appelable (callable)."
            exit_code = 1
            continue
            
        try:
            actual_method_sig = inspect.signature(method_obj)
            signature_errors = compare_signatures(actual_method_sig, expected_sig, method_name)
            if not signature_errors:
                test_results[check_name_method] = "Signature OK"
            else:
                test_results[check_name_method] = ["ÉCHEC : Signature incorrecte."] + [f"  - {err}" for err in signature_errors]
                exit_code = 1
        except ValueError: # Peut arriver si la signature ne peut pas être déterminée (ex: built-in)
            test_results[check_name_method] = "ÉCHEC : Impossible d'inspecter la signature."
            exit_code = 1
        except Exception as e_sig_inspect: # pylint: disable=broad-except
            test_results[check_name_method] = f"ÉCHEC : Erreur inspection signature: {e_sig_inspect}"
            exit_code = 1


    # Attribut REQUIRED_PARAMS
    if hasattr(strategy_class, "REQUIRED_PARAMS") and isinstance(getattr(strategy_class, "REQUIRED_PARAMS"), list):
        test_results["Attribut 'REQUIRED_PARAMS'"] = "OK (présent et est une liste)"
    else:
        test_results["Attribut 'REQUIRED_PARAMS'"] = "AVERTISSEMENT : Manquant ou n'est pas une liste. Recommandé."
        # Pas un échec bloquant, mais un avertissement.

    # Tentative d'Instanciation
    # Charger AppConfig pour obtenir les default_params
    app_config_for_defaults: Optional[AppConfig] = None
    try:
        app_config_for_defaults = load_all_configs(project_root=str(project_root_path))
    except Exception as e_conf_load_inst: # pylint: disable=broad-except
        logger.warning(f"Impossible de charger AppConfig pour récupérer les default_params : {e_conf_load_inst}. "
                       "Tentative d'instanciation avec des paramètres vides.")

    strategy_default_params: Dict[str, Any] = {}
    if app_config_for_defaults:
        strategy_default_params = get_default_params_for_strategy(app_config_for_defaults, strategy_class.__name__)
    
    if not strategy_default_params:
        logger.warning(f"Aucun paramètre par défaut trouvé pour {strategy_class.__name__}. "
                       "L'instanciation sera tentée avec un dictionnaire de paramètres vide. "
                       "La méthode _validate_params de la stratégie pourrait échouer si des paramètres sont requis.")
        # Si REQUIRED_PARAMS est défini et non vide, l'instanciation avec params={} échouera probablement
        # dans _validate_params, ce qui est un bon test.
    
    try:
        instance = strategy_class(strategy_name="TestStrategyInstance", symbol="TESTSYMBOL", params=strategy_default_params)
        test_results["Instanciation"] = f"OK (avec params: {strategy_default_params if strategy_default_params else '{}'})"
        
        # Tester l'appel à get_required_indicator_configs sur l'instance
        try:
            indicator_configs = instance.get_required_indicator_configs()
            if isinstance(indicator_configs, list):
                test_results["Appel get_required_indicator_configs"] = "OK (retourne une liste)"
                # On pourrait ajouter plus de validations sur le contenu de la liste ici
            else:
                test_results["Appel get_required_indicator_configs"] = f"ÉCHEC : Doit retourner List[Dict[str, Any]], reçu {type(indicator_configs)}"
                exit_code = 1
        except Exception as e_get_indic: # pylint: disable=broad-except
            test_results["Appel get_required_indicator_configs"] = f"ÉCHEC : Erreur lors de l'appel : {e_get_indic}"
            exit_code = 1

    except ValueError as e_val_inst: # Typiquement levé par _validate_params
        test_results["Instanciation"] = f"ÉCHEC (ValueError lors de _validate_params probable) : {e_val_inst}"
        exit_code = 1
    except TypeError as e_type_inst: # Typiquement si __init__ a des arguments manquants/incorrects
        test_results["Instanciation"] = f"ÉCHEC (TypeError lors de __init__ probable) : {e_type_inst}"
        exit_code = 1
    except Exception as e_inst: # pylint: disable=broad-except
        test_results["Instanciation"] = f"ÉCHEC : Erreur inattendue lors de l'instanciation : {e_inst}"
        exit_code = 1


    # 4. Rapport de Test
    print("\n--- Résultats des Tests de Conformité de la Stratégie ---")
    for check_name, result_val in test_results.items():
        if isinstance(result_val, list): # Pour les erreurs de signature multiples
            print(f"- {check_name}:")
            for item_err in result_val:
                print(f"  {item_err}")
        else:
            print(f"- {check_name}: {result_val}")

    if exit_code == 0:
        print("\nCONCLUSION : La stratégie semble conforme aux exigences de BaseStrategy.")
    else:
        print("\nCONCLUSION : La stratégie présente des problèmes de conformité.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
