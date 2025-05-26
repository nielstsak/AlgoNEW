# src/backtesting/indicator_calculator.py
"""
Ce module est responsable du calcul dynamique des séries d'indicateurs techniques
(ex: EMA, RSI, PSAR) sur un DataFrame donné, en utilisant les paramètres
spécifiques d'un "trial" d'optimisation ou d'une configuration de stratégie.
Il est principalement appelé par ObjectiveFunctionEvaluator ou lors de la préparation
des données pour le backtesting/live.
"""
import logging
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import numpy as np
import pandas_ta as ta # type: ignore

logger = logging.getLogger(__name__)

# Colonnes OHLCV de base qui doivent être présentes dans le DataFrame de sortie final
# avec les indicateurs _strat. Ces colonnes sont extraites de df_source_enriched.
BASE_OHLCV_OUTPUT_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

def _map_conceptual_output_keys_to_ta_df_cols(
    ta_df_columns: List[str],
    conceptual_key: str,
    indicator_name_for_log: str
) -> Optional[str]:
    """
    Tente de mapper une clé de sortie conceptuelle (ex: 'lower' pour bbands)
    à un nom de colonne réel dans le DataFrame retourné par pandas-ta.
    """
    # Mappings connus pour les indicateurs courants
    # Clé: (nom_indicateur_lower, clé_conceptuelle_lower)
    # Valeur: sous-chaîne attendue dans le nom de colonne de pandas-ta
    known_mappings = {
        ('bbands', 'lower'): 'bbl',
        ('bbands', 'middle'): 'bbm',
        ('bbands', 'upper'): 'bbu',
        ('bbands', 'bandwidth'): 'bbb',
        ('bbands', 'percent'): 'bbp',
        ('psar', 'long'): 'psarl', # Support PSAR
        ('psar', 'short'): 'psars', # Resistance PSAR
        ('psar', 'af'): 'psaraf',  # Acceleration Factor
        ('psar', 'reversal'): 'psarr', # Reversal points (0, 1, -1)
        ('macd', 'macd'): 'macd_', # MACD line itself (souvent MACD_fast_slow_signal)
        ('macd', 'histogram'): 'macdh_', # MACD Histogram
        ('macd', 'signal'): 'macds_', # MACD Signal line
        ('stoch', 'stoch_k'): 'stochk', # %K line
        ('stoch', 'stoch_d'): 'stochd', # %D line (signal of %K)
        # Ajouter d'autres mappings connus si nécessaire
    }

    map_key = (indicator_name_for_log.lower(), conceptual_key.lower())
    if map_key in known_mappings:
        substring_to_find = known_mappings[map_key]
        for col_name in ta_df_columns:
            if substring_to_find in col_name.lower():
                return col_name
        logger.warning(f"[_mapConceptualKeys] Pour indicateur '{indicator_name_for_log}', clé conceptuelle '{conceptual_key}', "
                       f"la sous-chaîne attendue '{substring_to_find}' n'a pas été trouvée dans les colonnes TA: {ta_df_columns}")
        return None # Non trouvé via mapping connu

    # Si pas de mapping connu, essayer une correspondance directe (insensible à la casse)
    for col_name in ta_df_columns:
        if conceptual_key.lower() == col_name.lower():
            return col_name
    
    logger.warning(f"[_mapConceptualKeys] Pour indicateur '{indicator_name_for_log}', clé conceptuelle '{conceptual_key}' "
                   f"non mappée et non trouvée directement dans les colonnes TA: {ta_df_columns}")
    return None


def calculate_indicators_for_trial(
    df_source_enriched: pd.DataFrame,
    required_indicator_configs: List[Dict[str, Any]],
    log_prefix_context: Optional[str] = ""
) -> pd.DataFrame:
    """
    Calcule dynamiquement une liste d'indicateurs techniques sur un DataFrame source enrichi.

    Args:
        df_source_enriched (pd.DataFrame): Le DataFrame source contenant les données OHLCV
            de base (1-minute) et potentiellement des K-lines agrégées pré-calculées.
            L'index doit être un DatetimeIndex (UTC attendu).
        required_indicator_configs (List[Dict[str, Any]]): Une liste de configurations d'indicateurs.
            Chaque config doit contenir: 'indicator_name', 'params', 'inputs', 'outputs'.
        log_prefix_context (Optional[str]): Préfixe de log optionnel.

    Returns:
        pd.DataFrame: Un DataFrame avec les colonnes OHLCV de base et les indicateurs calculés.
    """
    main_log_prefix = f"[IndicatorCalc]{log_prefix_context if log_prefix_context else ''}"
    logger.info(f"{main_log_prefix} Démarrage du calcul des indicateurs.")

    if df_source_enriched.empty:
        logger.warning(f"{main_log_prefix} df_source_enriched est vide. Retour d'un DataFrame vide.")
        return pd.DataFrame(columns=BASE_OHLCV_OUTPUT_COLUMNS)

    if not isinstance(df_source_enriched.index, pd.DatetimeIndex):
        logger.error(f"{main_log_prefix} L'index de df_source_enriched doit être un DatetimeIndex.")
        return pd.DataFrame(index=df_source_enriched.index, columns=BASE_OHLCV_OUTPUT_COLUMNS) # Garder l'index si possible
    
    # Assurer UTC pour l'index
    if df_source_enriched.index.tz is None:
        logger.warning(f"{main_log_prefix} L'index de df_source_enriched n'a pas de timezone. Localisation en UTC.")
        try:
            df_source_enriched.index = df_source_enriched.index.tz_localize('UTC')
        except Exception as e_tz_loc:
            logger.error(f"{main_log_prefix} Échec de la localisation de l'index en UTC: {e_tz_loc}. Indicateurs non calculés.")
            return pd.DataFrame(index=df_source_enriched.index, columns=BASE_OHLCV_OUTPUT_COLUMNS)
    elif str(df_source_enriched.index.tz).upper() != 'UTC': # Comparaison plus robuste
        logger.warning(f"{main_log_prefix} L'index de df_source_enriched n'est pas en UTC (actuel: {df_source_enriched.index.tz}). Conversion en UTC.")
        try:
            df_source_enriched.index = df_source_enriched.index.tz_convert('UTC')
        except Exception as e_tz_conv:
            logger.error(f"{main_log_prefix} Échec de la conversion de l'index en UTC: {e_tz_conv}. Indicateurs non calculés.")
            return pd.DataFrame(index=df_source_enriched.index, columns=BASE_OHLCV_OUTPUT_COLUMNS)


    missing_base_cols = [col for col in BASE_OHLCV_OUTPUT_COLUMNS if col not in df_source_enriched.columns]
    if missing_base_cols:
        logger.error(f"{main_log_prefix} Colonnes OHLCV de base manquantes dans df_source_enriched : {missing_base_cols}. "
                     "Impossible de continuer sans ces colonnes de base.")
        return pd.DataFrame(index=df_source_enriched.index, columns=BASE_OHLCV_OUTPUT_COLUMNS)

    df_results = df_source_enriched[BASE_OHLCV_OUTPUT_COLUMNS].copy()
    logger.debug(f"{main_log_prefix} DataFrame de résultats initialisé avec OHLCV de base. Shape: {df_results.shape}")

    # S'assurer que les colonnes OHLCV de base sont numériques
    for col_base_ohlcv in BASE_OHLCV_OUTPUT_COLUMNS:
        try:
            df_results[col_base_ohlcv] = pd.to_numeric(df_results[col_base_ohlcv], errors='raise')
        except ValueError as e_conv_base:
            logger.error(f"{main_log_prefix} Colonne de base '{col_base_ohlcv}' non convertible en numérique: {e_conv_base}. "
                         "Le calcul des indicateurs pourrait échouer.")
            df_results[col_base_ohlcv] = np.nan # Remplacer par NaN si la conversion échoue

    all_expected_output_strat_cols: List[str] = []
    for cfg in required_indicator_configs: # Collecter tous les noms de sortie attendus
        outputs_cfg_val = cfg.get('outputs')
        if isinstance(outputs_cfg_val, str): all_expected_output_strat_cols.append(outputs_cfg_val)
        elif isinstance(outputs_cfg_val, dict): all_expected_output_strat_cols.extend(outputs_cfg_val.values())
    
    for expected_col_init in all_expected_output_strat_cols: # Initialiser les colonnes de sortie à NaN
        if expected_col_init not in df_results.columns:
            df_results[expected_col_init] = np.nan


    for idx_cfg in required_indicator_configs:
        indicator_name_orig = idx_cfg.get('indicator_name')
        indicator_params = idx_cfg.get('params', {})
        input_mapping = idx_cfg.get('inputs', {})
        output_config = idx_cfg.get('outputs')

        # Validation de la configuration de l'indicateur
        if not all([
            isinstance(indicator_name_orig, str) and indicator_name_orig,
            isinstance(indicator_params, dict),
            isinstance(input_mapping, dict),
            output_config and (isinstance(output_config, str) or isinstance(output_config, dict))
        ]):
            logger.warning(f"{main_log_prefix} Configuration d'indicateur invalide ou incomplète : {idx_cfg}. Indicateur ignoré.")
            continue

        indicator_name_lower = indicator_name_orig.lower()
        current_indicator_log_prefix = f"{main_log_prefix}[{indicator_name_orig}]"
        logger.info(f"{current_indicator_log_prefix} Calcul avec params: {indicator_params}, entrées: {input_mapping}, sorties: {output_config}")

        ta_input_kwargs: Dict[str, pd.Series] = {} # pandas-ta attend des kwargs avec les séries
        valid_inputs_for_ta = True

        # Préparer les séries d'entrée pour pandas-ta
        for ta_expected_arg_name, source_col_name_in_df in input_mapping.items():
            if source_col_name_in_df not in df_source_enriched.columns:
                logger.error(f"{current_indicator_log_prefix} Colonne source '{source_col_name_in_df}' (pour entrée TA '{ta_expected_arg_name}') "
                             "non trouvée dans df_source_enriched. Indicateur ignoré.")
                valid_inputs_for_ta = False
                break
            
            series_data = df_source_enriched[source_col_name_in_df]
            if series_data.isnull().all():
                logger.warning(f"{current_indicator_log_prefix} Colonne source '{source_col_name_in_df}' (pour '{ta_expected_arg_name}') est entièrement NaN.")
            
            try:
                # L'argument pour pandas-ta doit être le nom attendu par la fonction (ex: 'close', 'high')
                ta_input_kwargs[ta_expected_arg_name.lower()] = series_data.astype(float)
            except ValueError as e_astype:
                logger.error(f"{current_indicator_log_prefix} Impossible de convertir la colonne source '{source_col_name_in_df}' en float pour l'entrée TA '{ta_expected_arg_name}': {e_astype}. Indicateur ignoré.")
                valid_inputs_for_ta = False
                break
        
        if not valid_inputs_for_ta:
            continue

        # Compléter avec les inputs TA par défaut si non spécifiés dans input_mapping
        default_ta_inputs_map = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
        for ta_default_arg_name, base_col_name_df in default_ta_inputs_map.items():
            if ta_default_arg_name not in ta_input_kwargs and base_col_name_df in df_source_enriched.columns:
                try:
                    ta_input_kwargs[ta_default_arg_name] = df_source_enriched[base_col_name_df].astype(float)
                    logger.debug(f"{current_indicator_log_prefix} Utilisation de la colonne de base 1-min '{base_col_name_df}' pour l'entrée TA '{ta_default_arg_name}'.")
                except ValueError:
                    logger.warning(f"{current_indicator_log_prefix} Échec conversion colonne de base '{base_col_name_df}' pour entrée TA '{ta_default_arg_name}'.")

        # Vérifier si l'entrée principale (souvent 'close' ou la première de ta_input_kwargs) a des données valides
        primary_input_series_for_check: Optional[pd.Series] = None
        if 'close' in ta_input_kwargs: primary_input_series_for_check = ta_input_kwargs['close']
        elif ta_input_kwargs: primary_input_series_for_check = next(iter(ta_input_kwargs.values()))

        if primary_input_series_for_check is None or not primary_input_series_for_check.notna().any():
            logger.warning(f"{current_indicator_log_prefix} L'entrée principale pour l'indicateur est vide ou entièrement NaN. "
                           "Le résultat de l'indicateur sera probablement NaN.")
            # Ne pas arrêter, laisser pandas-ta retourner NaN.
        
        try:
            indicator_function = getattr(ta, indicator_name_lower, None)
            if indicator_function is None:
                # Tentative de recherche dans les sous-modules communs de pandas-ta
                for category_module_name in ['trend', 'momentum', 'overlap', 'volume', 'volatility', 'cycles', 'statistics', 'transform', 'utils', 'performance']:
                    category_module = getattr(ta, category_module_name, None)
                    if category_module and hasattr(category_module, indicator_name_lower):
                        indicator_function = getattr(category_module, indicator_name_lower)
                        logger.debug(f"{current_indicator_log_prefix} Fonction indicateur '{indicator_name_lower}' trouvée dans ta.{category_module_name}.")
                        break
            
            if indicator_function is None:
                logger.error(f"{current_indicator_log_prefix} Fonction indicateur '{indicator_name_lower}' non trouvée dans pandas_ta.")
                continue

            indicator_result = indicator_function(**ta_input_kwargs, **indicator_params, append=False)

            if isinstance(indicator_result, pd.Series):
                if isinstance(output_config, str):
                    df_results[output_config] = indicator_result.reindex(df_results.index)
                    logger.debug(f"{current_indicator_log_prefix} Indicateur (Series) '{indicator_name_orig}' ajouté comme colonne '{output_config}'.")
                else:
                    logger.error(f"{current_indicator_log_prefix} L'indicateur '{indicator_name_orig}' a retourné une Series, mais 'outputs' n'est pas une chaîne : {output_config}")
            
            elif isinstance(indicator_result, pd.DataFrame):
                if isinstance(output_config, dict):
                    for conceptual_key, desired_output_col_name_strat in output_config.items():
                        # Tenter de mapper la clé conceptuelle à une colonne réelle du DataFrame de pandas-ta
                        actual_col_from_ta_df = _map_conceptual_output_keys_to_ta_df_cols(
                            indicator_result.columns.tolist(), conceptual_key, indicator_name_orig
                        )
                        
                        if actual_col_from_ta_df and actual_col_from_ta_df in indicator_result.columns:
                            df_results[desired_output_col_name_strat] = indicator_result[actual_col_from_ta_df].reindex(df_results.index)
                            logger.debug(f"{current_indicator_log_prefix} Colonne TA '{actual_col_from_ta_df}' (de '{indicator_name_orig}') "
                                         f"mappée depuis clé conceptuelle '{conceptual_key}' et ajoutée comme '{desired_output_col_name_strat}'.")
                        else:
                            logger.warning(f"{current_indicator_log_prefix} Colonne de sortie attendue via clé conceptuelle '{conceptual_key}' "
                                           f"non trouvée ou non mappable dans le DataFrame retourné par '{indicator_name_orig}'. "
                                           f"Colonnes TA disponibles: {indicator_result.columns.tolist()}. "
                                           f"Colonne _strat '{desired_output_col_name_strat}' restera NaN (ou sa valeur initiale).")
                            # df_results[desired_output_col_name_strat] est déjà initialisée à NaN
                else:
                    logger.error(f"{current_indicator_log_prefix} L'indicateur '{indicator_name_orig}' a retourné un DataFrame, "
                                 f"mais 'outputs' n'est pas un dictionnaire de mapping : {output_config}")
            
            elif indicator_result is None:
                 logger.warning(f"{current_indicator_log_prefix} Calcul de '{indicator_name_orig}' a retourné None. "
                                "Vérifiez les données d'entrée et les paramètres.")
                 # Les colonnes de sortie correspondantes resteront NaN (initialisées au début)
            else:
                logger.warning(f"{current_indicator_log_prefix} Résultat inattendu pour '{indicator_name_orig}'. Type: {type(indicator_result)}. "
                               "Attendu Series ou DataFrame.")

        except Exception as e_calc_indic:
            logger.error(f"{current_indicator_log_prefix} Erreur lors du calcul de '{indicator_name_orig}' avec params {indicator_params}: {e_calc_indic}", exc_info=True)
            # Les colonnes de sortie correspondantes resteront NaN (initialisées au début)
    
    # S'assurer que toutes les colonnes _strat attendues existent, même si elles sont restées NaN
    # Cette étape est maintenant faite au début du traitement des configs.

    logger.info(f"{main_log_prefix} Calcul de tous les indicateurs terminé. Shape final df_results: {df_results.shape}")
    logger.debug(f"{main_log_prefix} Colonnes dans df_results: {df_results.columns.tolist()}")
    
    return df_results
