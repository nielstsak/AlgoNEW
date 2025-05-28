# src/backtesting/indicator_calculator.py
"""
Ce module est responsable du calcul dynamique des séries d'indicateurs techniques
(ex: EMA, RSI, PSAR) sur un DataFrame donné, en utilisant les paramètres
spécifiques d'un "trial" d'optimisation ou d'une configuration de stratégie.
Il intègre désormais un CacheManager pour la mise en cache des résultats,
la gestion des dépendances d'indicateurs, et des optimisations de calcul.
"""
import hashlib
import json
import logging
import time # Pour le logging de performance
from typing import Dict, Any, List, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import pandas_ta as ta # type: ignore
import networkx as nx # Pour le graphe de dépendances
# from joblib import hash as joblib_hash # Alternative pour le hashing d'objets complexes
# import numba # Pour @numba.jit sur des fonctions custom si besoin

# Tentative d'importation de l'interface ICacheManager
try:
    from src.core.interfaces import ICacheManager
except ImportError:
    class ICacheManager: # type: ignore
        def get_or_compute(self, key: str, compute_func: Callable[[], Any], ttl: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> Any: return compute_func()
        def get_statistics(self) -> Dict[str, Any]: return {} # Placeholder
    logging.getLogger(__name__).warning(
        "ICacheManager interface not found in indicator_calculator. Using a placeholder."
    )

logger = logging.getLogger(__name__)

# Colonnes OHLCV de base qui doivent être présentes dans le DataFrame de sortie final
# avec les indicateurs _strat. Ces colonnes sont extraites de df_source_enriched.
BASE_OHLCV_OUTPUT_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

# Compteurs pour les statistiques du cache (spécifiques à ce module si le CacheManager est externe)
# Si CacheManager gère ses propres stats, ces compteurs pourraient être pour le logging local.
_calculator_cache_requests = 0
_calculator_cache_hits = 0
_calculator_cache_misses = 0
_calculator_stats_log_interval = 100 # Logger les stats toutes les N requêtes

def _generate_data_fingerprint(df: pd.DataFrame, relevant_cols: Optional[List[str]] = None) -> str:
    """
    Génère une empreinte (fingerprint) pour un DataFrame basée sur les premières et dernières
    lignes, la forme, et les statistiques descriptives des colonnes pertinentes.
    Ceci est moins coûteux que de hasher tout le DataFrame.
    """
    if df.empty:
        return "empty_df"
    
    fingerprint_parts = [str(df.shape)]
    if not df.index.empty:
        fingerprint_parts.append(str(df.index.min()))
        fingerprint_parts.append(str(df.index.max()))

    cols_to_describe = relevant_cols if relevant_cols else df.columns.tolist()
    cols_present = [col for col in cols_to_describe if col in df.columns]

    if not cols_present: # Si aucune colonne pertinente n'est trouvée
        return "_".join(fingerprint_parts)

    # Utiliser un sous-ensemble pour la description pour éviter de surcharger
    subset_size = min(len(df), 5) # Utiliser les 5 premières et dernières lignes
    if subset_size > 0:
        df_sample_head = df[cols_present].head(subset_size)
        df_sample_tail = df[cols_present].tail(subset_size)
        
        try:
            # Convertir en numérique, ignorer les erreurs pour les colonnes non numériques
            desc_head = df_sample_head.apply(pd.to_numeric, errors='coerce').describe().to_json(orient="split", date_format="iso", default_handler=str)
            desc_tail = df_sample_tail.apply(pd.to_numeric, errors='coerce').describe().to_json(orient="split", date_format="iso", default_handler=str)
            fingerprint_parts.append(desc_head)
            fingerprint_parts.append(desc_tail)
        except Exception as e_desc:
            logger.warning(f"[_generate_data_fingerprint] Erreur lors de la description du DataFrame pour fingerprint: {e_desc}")
            # Fallback si describe échoue (ex: types mixtes non gérés)
            fingerprint_parts.append(str(df_sample_head.iloc[0].to_dict() if not df_sample_head.empty else {}))
            fingerprint_parts.append(str(df_sample_tail.iloc[-1].to_dict() if not df_sample_tail.empty else {}))
            
    return "_".join(fingerprint_parts)


def _get_cache_key(
    indicator_configs: List[Dict[str, Any]],
    df_source_fingerprint: str # Utiliser un fingerprint du DataFrame source
    # df_shape_str: str,
    # df_index_min_str: str,
    # df_index_max_str: str
) -> str:
    """
    Génère une clé de cache SHA256 basée sur les configurations d'indicateurs
    et les caractéristiques du DataFrame source.
    """
    # Sérialiser les configurations d'indicateurs de manière stable (triées)
    # On s'assure que les dictionnaires internes sont aussi triés si possible
    # pour une meilleure stabilité du hash.
    def sort_dict_recursive(d):
        if isinstance(d, dict):
            return {k: sort_dict_recursive(d[k]) for k in sorted(d.keys())}
        elif isinstance(d, list):
            return sorted([sort_dict_recursive(x) for x in d], key=lambda x: str(x))
        return d

    try:
        # Trier la liste externe de dictionnaires par un élément stable (ex: indicator_name)
        sorted_configs = sorted(indicator_configs, key=lambda cfg: cfg.get('indicator_name', ''))
        # Appliquer le tri récursif à chaque dictionnaire de configuration
        stable_configs_repr_list = [sort_dict_recursive(cfg) for cfg in sorted_configs]
        
        config_str = json.dumps(stable_configs_repr_list, sort_keys=True, default=str)
    except Exception as e_json_dump:
        logger.warning(f"[_get_cache_key] Erreur lors de la sérialisation JSON des configs: {e_json_dump}. "
                       "Utilisation de repr() comme fallback (moins stable).")
        config_str = repr(sorted_configs) # Fallback moins stable

    # Combiner avec les caractéristiques du DataFrame
    # key_material = f"{config_str}_{df_shape_str}_{df_index_min_str}_{df_index_max_str}"
    key_material = f"{config_str}_{df_source_fingerprint}"
    
    return hashlib.sha256(key_material.encode('utf-8')).hexdigest()


def _build_dependency_graph(indicator_configs: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Construit un graphe de dépendances pour les indicateurs.
    Les nœuds sont les noms de colonnes de sortie des indicateurs.
    Une arête de A vers B signifie que B dépend de A.
    """
    graph = nx.DiGraph()
    output_to_config_map: Dict[str, Dict[str, Any]] = {}

    # Ajouter tous les nœuds de sortie possibles et mapper les sorties à leurs configs
    for config in indicator_configs:
        outputs = config.get('outputs')
        if isinstance(outputs, str):
            graph.add_node(outputs, config=config)
            output_to_config_map[outputs] = config
        elif isinstance(outputs, dict):
            for conceptual_key, output_col_name in outputs.items():
                graph.add_node(output_col_name, config=config, conceptual_key=conceptual_key)
                output_to_config_map[output_col_name] = config
    
    # Ajouter les arêtes de dépendance
    for output_node_name, config_data_for_node in list(graph.nodes(data=True)): # Utiliser list() pour copier
        node_config = config_data_for_node.get('config')
        if not node_config: continue

        inputs_map = node_config.get('inputs', {})
        for input_conceptual_key, source_col_name_or_dependency in inputs_map.items():
            # Si la source est une autre colonne de sortie d'indicateur (déjà dans le graphe)
            if source_col_name_or_dependency in graph:
                graph.add_edge(source_col_name_or_dependency, output_node_name)
            # Sinon, on suppose que c'est une colonne source du DataFrame original (ex: 'close', 'Klines_5min_high')
            # et elle n'a pas de dépendance entrante dans ce contexte de calcul d'indicateurs.

    # Vérifier les cycles (ne devrait pas arriver avec pandas_ta, mais bon pour la robustesse)
    try:
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            logger.error(f"[_build_dependency_graph] Cycles détectés dans le graphe de dépendances des indicateurs: {cycles}")
            # Gérer l'erreur de cycle (ex: lever une exception ou tenter de casser le cycle)
            # Pour l'instant, on logue et on continue, mais le tri topologique échouera.
    except Exception as e_cycle:
         logger.error(f"[_build_dependency_graph] Erreur lors de la détection de cycles: {e_cycle}")


    logger.debug(f"[_build_dependency_graph] Graphe de dépendances construit. Nœuds: {graph.number_of_nodes()}, Arêtes: {graph.number_of_edges()}")
    return graph

def _map_conceptual_output_keys_to_ta_df_cols(
    ta_df_columns: List[str],
    conceptual_key: str,
    indicator_name_for_log: str,
    indicator_params_for_log: Dict[str, Any]
) -> Optional[str]:
    """
    Tente de mapper une clé de sortie conceptuelle (ex: 'lower' pour bbands)
    à un nom de colonne réel dans le DataFrame retourné par pandas-ta.
    """
    log_prefix_map = f"[_mapConceptualKeys][{indicator_name_for_log}({conceptual_key})]"
    
    param_str_parts = []
    # Ordre de préférence pour la construction du nom de colonne pandas-ta
    param_order_preference = ['length', 'fast', 'slow', 'signal', 'std', 'af', 'max_af', 'mamode']
    
    # Construire une chaîne de paramètres triée selon la préférence, puis les autres
    sorted_params_for_naming_dict = {
        k: indicator_params_for_log[k]
        for k in param_order_preference
        if k in indicator_params_for_log and indicator_params_for_log[k] is not None
    }
    for k_param, v_param in indicator_params_for_log.items():
        if k_param not in sorted_params_for_naming_dict and v_param is not None:
            sorted_params_for_naming_dict[k_param] = v_param

    for _, v_param_val in sorted_params_for_naming_dict.items():
        if isinstance(v_param_val, float):
            # Éviter ".0" pour les entiers formatés en float, ex: 20.0 -> "20"
            param_str_parts.append(str(int(v_param_val)) if v_param_val.is_integer() else str(v_param_val))
        elif isinstance(v_param_val, (int, str, bool)):
            param_str_parts.append(str(v_param_val))
        # Ignorer les autres types pour la construction du nom de colonne

    # Mappings connus des préfixes de colonnes pandas-ta pour des clés conceptuelles
    known_mappings_prefix = {
        ('bbands', 'lower'): 'bbl', ('bbands', 'middle'): 'bbm', ('bbands', 'upper'): 'bbu',
        ('bbands', 'bandwidth'): 'bbb', ('bbands', 'percent'): 'bbp',
        ('psar', 'long'): 'psarl', ('psar', 'short'): 'psars',
        ('psar', 'af'): 'psaraf', ('psar', 'reversal'): 'psarr',
        ('macd', 'macd'): 'macd', ('macd', 'histogram'): 'macdh', ('macd', 'signal'): 'macds',
        ('stoch', 'stoch_k'): 'stochk', ('stoch', 'stoch_d'): 'stochd',
        ('slope', 'slope'): 'slope', ('atr', 'true_range'): 'tr', ('atr', 'atr'): 'atr'
        # Ajouter d'autres mappings connus au besoin
    }

    map_key_tuple = (indicator_name_for_log.lower(), conceptual_key.lower())
    
    if map_key_tuple in known_mappings_prefix:
        expected_col_prefix_ta = known_mappings_prefix[map_key_tuple]
        # Construire le nom de colonne complet attendu par pandas-ta (ex: "BBL_5_2.0", "PSARl", "MACD_12_26_9")
        # Pour PSAR, pandas-ta ne met pas toujours les params dans le nom.
        # Pour ATR, c'est souvent juste ATR_period.
        expected_full_col_name_base = expected_col_prefix_ta.upper()
        if param_str_parts and indicator_name_for_log.lower() not in ['psar']: # PSAR a un nommage spécial
            if indicator_name_for_log.lower() == 'atr' and conceptual_key.lower() == 'atr': # ATR_period
                 expected_full_col_name_base = f"ATR_{param_str_parts[0]}" if param_str_parts else expected_col_prefix_ta.upper()
            else:
                 expected_full_col_name_base += f"_{'_'.join(param_str_parts)}"
        
        # Chercher une correspondance exacte ou commençant par
        for col_name_ta in ta_df_columns:
            if col_name_ta.lower() == expected_full_col_name_base.lower():
                logger.debug(f"{log_prefix_map} Colonne TA '{col_name_ta}' trouvée par correspondance exacte construite '{expected_full_col_name_base}'.")
                return col_name_ta
        # Fallback: chercher si la colonne TA commence par le nom construit (sans les params pour certains indicateurs)
        for col_name_ta in ta_df_columns:
            if col_name_ta.lower().startswith(expected_full_col_name_base.lower()):
                logger.debug(f"{log_prefix_map} Colonne TA '{col_name_ta}' trouvée pour préfixe construit '{expected_full_col_name_base}'.")
                return col_name_ta
        # Fallback plus générique: chercher si la colonne TA contient le préfixe de base (ex: "BBL" dans "BBL_longueur_std")
        for col_name_ta in ta_df_columns:
            if expected_col_prefix_ta.lower() in col_name_ta.lower():
                logger.debug(f"{log_prefix_map} Colonne TA '{col_name_ta}' trouvée pour sous-chaîne de base '{expected_col_prefix_ta}' (fallback).")
                return col_name_ta
        
        logger.warning(f"{log_prefix_map} Préfixe attendu '{expected_col_prefix_ta}' (ou avec params '{expected_full_col_name_base}') "
                       f"non trouvé dans les colonnes TA: {ta_df_columns}")
        return None # Non trouvé par mapping connu

    # Si pas de mapping connu, chercher une correspondance directe de la clé conceptuelle
    for col_name_ta in ta_df_columns:
        if conceptual_key.lower() == col_name_ta.lower():
            logger.debug(f"{log_prefix_map} Colonne TA '{col_name_ta}' trouvée par correspondance directe avec clé conceptuelle.")
            return col_name_ta
    
    logger.warning(f"{log_prefix_map} Clé conceptuelle '{conceptual_key}' non mappée et non trouvée directement dans les colonnes TA: {ta_df_columns}")
    return None

def _calculate_single_indicator_set(
    indicator_config: Dict[str, Any],
    df_input_data_view: pd.DataFrame, # Vue du DataFrame source avec les colonnes nécessaires
    log_prefix_context: str = ""
) -> Optional[Union[pd.Series, pd.DataFrame]]:
    """
    Calcule un ensemble d'indicateurs (potentiellement plusieurs sorties) basé sur une config.
    Utilise pandas_ta.
    """
    indicator_name_orig = indicator_config.get('indicator_name')
    indicator_params = indicator_config.get('params', {})
    input_mapping = indicator_config.get('inputs', {})
    # output_config = indicator_config.get('outputs') # Le nommage des sorties est géré par l'appelant

    current_indicator_log_prefix = f"{log_prefix_context}[CalcSet:{indicator_name_orig}]"
    
    if not isinstance(indicator_name_orig, str) or not indicator_name_orig:
        logger.error(f"{current_indicator_log_prefix} Nom d'indicateur manquant ou invalide dans la config.")
        return None

    indicator_name_lower = indicator_name_orig.lower()
    
    ta_input_kwargs: Dict[str, pd.Series] = {}
    valid_inputs_for_ta_call = True

    for ta_expected_arg_name, source_col_name_in_df_input in input_mapping.items():
        if source_col_name_in_df_input not in df_input_data_view.columns:
            logger.error(f"{current_indicator_log_prefix} Colonne source '{source_col_name_in_df_input}' pour TA arg '{ta_expected_arg_name}' non trouvée. Indicateur ignoré.")
            valid_inputs_for_ta_call = False
            break
        
        series_data_from_source = df_input_data_view[source_col_name_in_df_input]
        if series_data_from_source.isnull().all():
            logger.warning(f"{current_indicator_log_prefix} Colonne source '{source_col_name_in_df_input}' est entièrement NaN.")
        
        try:
            ta_input_kwargs[ta_expected_arg_name.lower()] = series_data_from_source.astype(float)
        except ValueError as e_astype:
            logger.error(f"{current_indicator_log_prefix} Conversion de '{source_col_name_in_df_input}' en float échouée: {e_astype}. Indicateur ignoré.")
            valid_inputs_for_ta_call = False
            break
    
    if not valid_inputs_for_ta_call:
        return None

    # Ajout des entrées par défaut (open, high, low, close, volume) si non spécifiées et disponibles
    default_ta_inputs_map = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
    for ta_default_arg, df_col_name in default_ta_inputs_map.items():
        if ta_default_arg not in ta_input_kwargs and df_col_name in df_input_data_view.columns:
            try:
                ta_input_kwargs[ta_default_arg] = df_input_data_view[df_col_name].astype(float)
            except ValueError: pass # Ignorer si la conversion échoue pour une colonne optionnelle

    indicator_function: Optional[Any] = getattr(ta, indicator_name_lower, None)
    if indicator_function is None: # Chercher dans les sous-modules de ta
        for category_module_name in ['trend', 'momentum', 'overlap', 'volume', 'volatility', 'cycles', 'statistics', 'transform', 'performance', 'candles']:
            category_module = getattr(ta, category_module_name, None)
            if category_module and hasattr(category_module, indicator_name_lower):
                indicator_function = getattr(category_module, indicator_name_lower)
                break
    
    if indicator_function is None:
        logger.error(f"{current_indicator_log_prefix} Fonction pandas_ta '{indicator_name_lower}' non trouvée.")
        return None

    try:
        logger.debug(f"{current_indicator_log_prefix} Appel de pandas_ta.{indicator_name_lower} avec "
                     f"kwargs: {list(ta_input_kwargs.keys())}, params: {indicator_params}")
        # Utiliser .copy() sur les séries d'entrée pour éviter les modifications in-place si pandas_ta le fait
        # (bien que ce soit rare pour les fonctions qui retournent de nouvelles séries/DF).
        # Cependant, pour la performance, on peut l'omettre si on est sûr que pandas_ta ne modifie pas.
        # Pour la robustesse, une copie est plus sûre, mais peut impacter la mémoire/temps.
        # df_input_data_view est déjà une vue, donc les séries extraites sont aussi des vues.
        # On ne copie que si on passe à une fonction qui pourrait modifier.
        # pandas-ta retourne généralement de nouveaux objets.
        
        # Nettoyer les paramètres pour pandas-ta (ex: certains attendent des types spécifiques)
        cleaned_params = {}
        for k, v in indicator_params.items():
            if isinstance(v, np.integer): cleaned_params[k] = int(v)
            elif isinstance(v, np.floating): cleaned_params[k] = float(v)
            else: cleaned_params[k] = v
            
        indicator_result_ta = indicator_function(**ta_input_kwargs, **cleaned_params, append=False)
        
        if indicator_result_ta is None:
            logger.warning(f"{current_indicator_log_prefix} pandas_ta.{indicator_name_lower} a retourné None.")
        elif isinstance(indicator_result_ta, (pd.Series, pd.DataFrame)) and indicator_result_ta.empty:
            logger.warning(f"{current_indicator_log_prefix} pandas_ta.{indicator_name_lower} a retourné un Series/DataFrame vide.")
        
        return indicator_result_ta
    except Exception as e_call_ta:
        logger.error(f"{current_indicator_log_prefix} Erreur lors de l'appel à pandas_ta.{indicator_name_lower} "
                     f"avec params {indicator_params}: {e_call_ta}", exc_info=True)
        return None


def _execute_indicator_calculations_in_order(
    df_source_data_for_calc: pd.DataFrame, # Doit contenir toutes les colonnes sources nécessaires
    ordered_configs_to_process: List[Dict[str, Any]], # Configs triées par dépendance
    log_prefix_context: str = ""
) -> pd.DataFrame:
    """
    Exécute le calcul des indicateurs dans l'ordre spécifié (par dépendance).
    Les résultats sont ajoutés au DataFrame de résultats.
    """
    main_exec_log_prefix = f"[ExecInOrder]{log_prefix_context}"
    # df_results_intermediate commence avec les colonnes OHLCV de base et les colonnes sources
    # nécessaires aux indicateurs qui ne sont pas OHLCV (ex: Klines_5min_close).
    # On s'assure que toutes les colonnes potentiellement sources sont là.
    # Pour simplifier, on part d'une copie du df_source qui contient déjà tout.
    df_results_intermediate = df_source_data_for_calc.copy()
    
    # Initialiser toutes les colonnes de sortie attendues avec NaN pour éviter les KeyError
    # si un calcul échoue ou ne produit pas toutes les sorties attendues.
    all_final_output_names_from_configs: List[str] = []
    for cfg_out_init in ordered_configs_to_process:
        outputs_val = cfg_out_init.get('outputs')
        if isinstance(outputs_val, str): all_final_output_names_from_configs.append(outputs_val)
        elif isinstance(outputs_val, dict): all_final_output_names_from_configs.extend(list(outputs_val.values()))
    
    for out_col_name_to_init in all_final_output_names_from_configs:
        if out_col_name_to_init not in df_results_intermediate.columns:
            df_results_intermediate[out_col_name_to_init] = np.nan
            logger.debug(f"{main_exec_log_prefix} Colonne de sortie pré-initialisée à NaN: {out_col_name_to_init}")


    # TODO: Implémenter la parallélisation des groupes indépendants ici avec ThreadPoolExecutor
    # Pour l'instant, exécution séquentielle pour la simplicité de cette refactorisation.
    # La parallélisation nécessiterait d'identifier des "niveaux" dans le graphe topologique.

    for config_item in ordered_configs_to_process:
        indic_name_item = config_item.get('indicator_name', 'N/A')
        indic_params_item = config_item.get('params', {})
        outputs_config_item = config_item.get('outputs')
        
        # Les inputs pour _calculate_single_indicator_set doivent venir de df_results_intermediate
        # car il contient les résultats des indicateurs précédents.
        # df_source_data_for_calc est la source "brute" (enrichie mais avant ces indicateurs _strat).
        # On doit s'assurer que les colonnes sources référencées par config_item['inputs']
        # sont bien dans df_results_intermediate (soit des colonnes originales, soit des indicateurs déjà calculés).
        
        # Créer une vue ou une copie des données nécessaires pour ce calcul spécifique
        # pour éviter de passer tout df_results_intermediate à chaque fois.
        # Colonnes nécessaires = celles listées dans config_item['inputs'].values()
        # + les colonnes OHLCV de base si implicitement utilisées par pandas-ta.
        input_source_cols_needed = list(config_item.get('inputs', {}).values())
        # Ajouter les colonnes OHLCV de base si elles ne sont pas déjà listées et sont présentes
        for base_col in ['open', 'high', 'low', 'close', 'volume']:
            if base_col not in input_source_cols_needed and base_col in df_results_intermediate.columns:
                input_source_cols_needed.append(base_col)
        
        # S'assurer que les colonnes existent avant de faire le slice
        missing_src_for_slice = [c for c in input_source_cols_needed if c not in df_results_intermediate.columns]
        if missing_src_for_slice:
            logger.error(f"{main_exec_log_prefix}[{indic_name_item}] Colonnes sources pour slice manquantes: {missing_src_for_slice}. Calcul ignoré.")
            continue
            
        df_input_slice_for_calc = df_results_intermediate[input_source_cols_needed]

        calculated_data = _calculate_single_indicator_set(
            indicator_config=config_item,
            df_input_data_view=df_input_slice_for_calc, # Passer la vue/copie avec seulement les colonnes nécessaires
            log_prefix_context=f"{log_prefix_context}[SeqExec]"
        )

        if calculated_data is None:
            logger.warning(f"{main_exec_log_prefix}[{indic_name_item}] Le calcul n'a retourné aucune donnée.")
            continue

        # Fusionner les résultats dans df_results_intermediate
        if isinstance(calculated_data, pd.Series):
            if isinstance(outputs_config_item, str):
                # S'assurer que l'index correspond avant l'assignation
                df_results_intermediate[outputs_config_item] = calculated_data.reindex(df_results_intermediate.index)
            else:
                logger.error(f"{main_exec_log_prefix}[{indic_name_item}] Résultat Series mais config 'outputs' n'est pas str.")
        elif isinstance(calculated_data, pd.DataFrame):
            if isinstance(outputs_config_item, dict):
                for conceptual_key, target_col_name_strat in outputs_config_item.items():
                    # Mapper la clé conceptuelle au nom de colonne réel du DataFrame de pandas-ta
                    actual_col_from_ta = _map_conceptual_output_keys_to_ta_df_cols(
                        calculated_data.columns.tolist(),
                        conceptual_key,
                        indic_name_item,
                        indic_params_item
                    )
                    if actual_col_from_ta and actual_col_from_ta in calculated_data.columns:
                        df_results_intermediate[target_col_name_strat] = calculated_data[actual_col_from_ta].reindex(df_results_intermediate.index)
                    else:
                        logger.warning(f"{main_exec_log_prefix}[{indic_name_item}] Colonne TA pour clé '{conceptual_key}' "
                                       f"(attendue: {actual_col_from_ta}) non trouvée dans résultat. '{target_col_name_strat}' restera NaN.")
            else:
                logger.error(f"{main_exec_log_prefix}[{indic_name_item}] Résultat DataFrame mais config 'outputs' n'est pas dict.")
    
    # Retourner seulement les colonnes de base et les colonnes _strat finales attendues
    # pour éviter de propager des colonnes intermédiaires de pandas-ta.
    final_cols_to_return = BASE_OHLCV_OUTPUT_COLUMNS + all_final_output_names_from_configs
    # S'assurer que les colonnes existent dans df_results_intermediate avant de les sélectionner
    existing_final_cols = [col for col in final_cols_to_return if col in df_results_intermediate.columns]
    
    return df_results_intermediate[existing_final_cols]


def calculate_indicators_for_trial(
    df_source_enriched: pd.DataFrame,
    required_indicator_configs: List[Dict[str, Any]],
    cache_manager: Optional[ICacheManager] = None, # Injection du CacheManager
    log_prefix_context: Optional[str] = "" # Pour le contexte de logging (ex: Trial ID)
) -> pd.DataFrame:
    """
    Calcule dynamiquement une liste d'indicateurs techniques sur un DataFrame source enrichi.
    Utilise un CacheManager pour la mise en cache des résultats.
    """
    global _calculator_cache_requests, _calculator_cache_hits, _calculator_cache_misses
    
    main_calc_log_prefix = f"[IndicatorCalcForTrial]{log_prefix_context or ''}"
    logger.info(f"{main_calc_log_prefix} Démarrage du calcul des indicateurs pour le trial.")
    start_time_overall = time.perf_counter()

    if df_source_enriched.empty:
        logger.warning(f"{main_calc_log_prefix} df_source_enriched est vide. Retour d'un DataFrame vide.")
        return pd.DataFrame(columns=BASE_OHLCV_OUTPUT_COLUMNS) # Retourner un DF avec au moins les colonnes de base

    # 1. Préparer le fingerprint du DataFrame source pour la clé de cache
    # Sélectionner les colonnes pertinentes pour le fingerprint (OHLCV + colonnes sources des indicateurs)
    source_cols_for_fingerprint = set(BASE_OHLCV_OUTPUT_COLUMNS)
    for cfg in required_indicator_configs:
        inputs = cfg.get('inputs', {})
        if isinstance(inputs, dict):
            source_cols_for_fingerprint.update(inputs.values())
    
    df_source_fingerprint_str = _generate_data_fingerprint(
        df_source_enriched,
        relevant_cols=sorted(list(s_col for s_col in source_cols_for_fingerprint if isinstance(s_col, str)))
    )

    # 2. Générer la clé de cache
    cache_key = _get_cache_key(
        indicator_configs=required_indicator_configs,
        df_source_fingerprint=df_source_fingerprint_str
    )
    logger.debug(f"{main_calc_log_prefix} Clé de cache générée : {cache_key}")

    # 3. Logique de Cache et Calcul
    if cache_manager:
        _calculator_cache_requests += 1
        
        # Définir la fonction de calcul à passer à get_or_compute
        def _compute_indicators_logic() -> pd.DataFrame:
            logger.info(f"{main_calc_log_prefix}[CacheKey:{cache_key}] Exécution de _compute_indicators_logic (cache miss).")
            graph = _build_dependency_graph(required_indicator_configs)
            try:
                # Obtenir l'ordre d'exécution topologique
                # Les nœuds sans dépendances peuvent être calculés en premier (ou en parallèle)
                # Pour l'instant, on prend un ordre topologique simple.
                # La parallélisation serait sur des groupes de nœuds indépendants à chaque "niveau".
                # nx.topological_sort est un générateur.
                ordered_nodes = list(nx.topological_sort(graph))
            except nx.NetworkXUnfeasible: # Si le graphe a un cycle
                logger.error(f"{main_calc_log_prefix}[CacheKey:{cache_key}] Cycle détecté dans le graphe de dépendances. "
                             "Impossible de déterminer l'ordre de calcul. Calcul séquentiel des configs brutes.")
                # Fallback: utiliser l'ordre original des configs (moins optimal)
                # ou lever une erreur. Pour l'instant, on essaie l'ordre original.
                # Cela suppose que les configs sont déjà à peu près dans le bon ordre
                # ou que les dépendances sont gérées par des colonnes sources du DF original.
                return _execute_indicator_calculations_in_order(
                    df_source_enriched, required_indicator_configs, log_prefix_context
                )

            # Mapper les noms de nœuds (colonnes de sortie) à leurs configurations
            # en utilisant les données stockées dans le graphe.
            configs_in_exec_order: List[Dict[str, Any]] = []
            processed_configs_for_nodes = set() # Pour éviter de traiter la même config plusieurs fois si elle produit plusieurs outputs

            for node_name in ordered_nodes:
                node_data = graph.nodes[node_name]
                config_for_node = node_data.get('config')
                if config_for_node:
                    # Utiliser l'ID de l'objet config pour vérifier si on l'a déjà traitée
                    config_id = id(config_for_node)
                    if config_id not in processed_configs_for_nodes:
                        configs_in_exec_order.append(config_for_node)
                        processed_configs_for_nodes.add(config_id)
            
            return _execute_indicator_calculations_in_order(
                df_source_enriched, configs_in_exec_order, log_prefix_context
            )

        # Utiliser le CacheManager
        try:
            start_time_get_or_compute = time.perf_counter()
            # Le TTL pourrait être configuré globalement ou par type de donnée
            df_with_indicators = cache_manager.get_or_compute(
                key=cache_key,
                compute_func=_compute_indicators_logic,
                ttl=7*24*3600 # Exemple: cache pour 7 jours
            )
            duration_get_or_compute = time.perf_counter() - start_time_get_or_compute
            
            # Vérifier si c'était un hit ou un miss (le CacheManager pourrait le loguer,
            # mais on peut le déduire si _compute_indicators_logic a été appelé)
            # Pour cela, _compute_indicators_logic devrait retourner un flag ou on compare
            # le temps d'exécution. Pour l'instant, on se fie aux stats du CacheManager.
            
            # Loguer les stats du cache périodiquement
            if _calculator_cache_requests % _calculator_stats_log_interval == 0:
                stats = cache_manager.get_statistics()
                logger.info(f"{main_calc_log_prefix} Stats CacheManager (après {_calculator_cache_requests} requêtes à IndicatorCalculator): {stats}")

            logger.info(f"{main_calc_log_prefix} Indicateurs obtenus via CacheManager en {duration_get_or_compute:.4f}s.")

        except Exception as e_cache_op:
            logger.error(f"{main_calc_log_prefix}[CacheKey:{cache_key}] Erreur avec CacheManager: {e_cache_op}. "
                         "Tentative de calcul direct sans cache.", exc_info=True)
            # Fallback sur calcul direct si le cache échoue
            df_with_indicators = _compute_indicators_logic() # Appel direct
    else:
        # Pas de CacheManager fourni, calcul direct
        logger.warning(f"{main_calc_log_prefix} CacheManager non fourni. Calcul direct des indicateurs.")
        # La logique de _compute_indicators_logic est essentiellement ce qu'on ferait ici.
        graph = _build_dependency_graph(required_indicator_configs)
        try:
            ordered_nodes = list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            logger.error(f"{main_calc_log_prefix} Cycle détecté (sans cache). Calcul séquentiel des configs brutes.")
            df_with_indicators = _execute_indicator_calculations_in_order(
                df_source_enriched, required_indicator_configs, log_prefix_context
            )
        else:
            configs_in_exec_order_no_cache: List[Dict[str, Any]] = []
            processed_configs_no_cache = set()
            for node_name_nc in ordered_nodes:
                node_data_nc = graph.nodes[node_name_nc]
                config_nc = node_data_nc.get('config')
                if config_nc and id(config_nc) not in processed_configs_no_cache:
                    configs_in_exec_order_no_cache.append(config_nc)
                    processed_configs_no_cache.add(id(config_nc))
            df_with_indicators = _execute_indicator_calculations_in_order(
                df_source_enriched, configs_in_exec_order_no_cache, log_prefix_context
            )

    duration_overall = time.perf_counter() - start_time_overall
    logger.info(f"{main_calc_log_prefix} Calcul des indicateurs pour le trial terminé en {duration_overall:.4f}s. "
                f"Shape final du DataFrame: {df_with_indicators.shape if df_with_indicators is not None else 'None'}")
    
    if df_with_indicators is None: # Si une erreur majeure s'est produite
        return pd.DataFrame(columns=BASE_OHLCV_OUTPUT_COLUMNS)
        
    return df_with_indicators

   