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

import pandas as pd # Assurer l'import pour Pylance
import numpy as np
import pandas_ta as ta # type: ignore

logger = logging.getLogger(__name__)

# Colonnes OHLCV de base qui doivent être présentes dans le DataFrame de sortie final
# avec les indicateurs _strat. Ces colonnes sont extraites de df_source_enriched.
BASE_OHLCV_OUTPUT_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

def _map_conceptual_output_keys_to_ta_df_cols(
    ta_df_columns: List[str],
    conceptual_key: str,
    indicator_name_for_log: str,
    indicator_params_for_log: Dict[str, Any] 
) -> Optional[str]:
    """
    Tente de mapper une clé de sortie conceptuelle (ex: 'lower' pour bbands)
    à un nom de colonne réel dans le DataFrame retourné par pandas-ta.
    Prend en compte les paramètres de l'indicateur pour un mappage plus précis.
    """
    log_prefix_map = f"[_mapConceptualKeys][{indicator_name_for_log}({conceptual_key})]"
    
    param_str_parts = []
    param_order_preference = ['length', 'fast', 'slow', 'signal', 'std', 'af', 'max_af'] 
    
    sorted_params_for_naming = {
        k: indicator_params_for_log[k] 
        for k in param_order_preference 
        if k in indicator_params_for_log
    }
    for k, v in indicator_params_for_log.items():
        if k not in sorted_params_for_naming:
            sorted_params_for_naming[k] = v

    for k_param, v_param in sorted_params_for_naming.items():
        if isinstance(v_param, float):
            param_str_parts.append(f"{v_param:.1f}" if v_param.is_integer() else str(v_param))
        elif isinstance(v_param, (int, str, bool)):
            param_str_parts.append(str(v_param))

    known_mappings_prefix = {
        ('bbands', 'lower'): 'bbl', ('bbands', 'middle'): 'bbm', ('bbands', 'upper'): 'bbu',
        ('bbands', 'bandwidth'): 'bbb', ('bbands', 'percent'): 'bbp',
        ('psar', 'long'): 'psarl', ('psar', 'short'): 'psars',
        ('psar', 'af'): 'psaraf', ('psar', 'reversal'): 'psarr',
        ('macd', 'macd'): 'macd', ('macd', 'histogram'): 'macdh', ('macd', 'signal'): 'macds',
        ('stoch', 'stoch_k'): 'stochk', ('stoch', 'stoch_d'): 'stochd',
        ('slope', 'slope'): 'slope' 
    }

    map_key_tuple = (indicator_name_for_log.lower(), conceptual_key.lower())
    
    if map_key_tuple in known_mappings_prefix:
        expected_col_prefix_ta = known_mappings_prefix[map_key_tuple]
        expected_full_col_name_base = f"{expected_col_prefix_ta.upper()}"
        if param_str_parts and indicator_name_for_log.lower() not in ['psar']:
            expected_full_col_name_base += f"_{'_'.join(param_str_parts)}"
        
        for col_name_ta in ta_df_columns:
            if col_name_ta.lower().startswith(expected_full_col_name_base.lower()):
                logger.debug(f"{log_prefix_map} Colonne TA '{col_name_ta}' trouvée pour préfixe construit '{expected_full_col_name_base}'.")
                return col_name_ta
        
        for col_name_ta in ta_df_columns:
            if expected_col_prefix_ta in col_name_ta.lower():
                logger.debug(f"{log_prefix_map} Colonne TA '{col_name_ta}' trouvée pour sous-chaîne de base '{expected_col_prefix_ta}' (fallback).")
                return col_name_ta
        
        logger.warning(f"{log_prefix_map} Préfixe attendu '{expected_col_prefix_ta}' (ou avec params '{expected_full_col_name_base}') "
                       f"non trouvé dans les colonnes TA: {ta_df_columns}")
        return None

    for col_name_ta in ta_df_columns:
        if conceptual_key.lower() == col_name_ta.lower():
            logger.debug(f"{log_prefix_map} Colonne TA '{col_name_ta}' trouvée par correspondance directe avec clé conceptuelle.")
            return col_name_ta
    
    logger.warning(f"{log_prefix_map} Clé conceptuelle '{conceptual_key}' non mappée et non trouvée directement dans les colonnes TA: {ta_df_columns}")
    return None


def calculate_indicators_for_trial(
    df_source_enriched: pd.DataFrame,
    required_indicator_configs: List[Dict[str, Any]],
    log_prefix_context: Optional[str] = ""
) -> pd.DataFrame:
    """
    Calcule dynamiquement une liste d'indicateurs techniques sur un DataFrame source enrichi.
    """
    # Sourcery: Replace if-expression with `or`
    main_log_prefix = f"[IndicatorCalc]{log_prefix_context or ''}"
    logger.info(f"{main_log_prefix} Démarrage du calcul des indicateurs.")

    if df_source_enriched.empty:
        logger.warning(f"{main_log_prefix} df_source_enriched est vide. Retour d'un DataFrame vide.")
        return pd.DataFrame(columns=BASE_OHLCV_OUTPUT_COLUMNS)

    if not isinstance(df_source_enriched.index, pd.DatetimeIndex):
        logger.error(f"{main_log_prefix} L'index de df_source_enriched doit être un DatetimeIndex.")
        return pd.DataFrame(index=df_source_enriched.index, columns=BASE_OHLCV_OUTPUT_COLUMNS) 
    
    df_working_copy = df_source_enriched.copy() 
    if df_working_copy.index.tz is None:
        logger.debug(f"{main_log_prefix} Index de df_source_enriched sans timezone. Localisation en UTC.")
        df_working_copy.index = df_working_copy.index.tz_localize('UTC') # type: ignore
    elif str(df_working_copy.index.tz).upper() != 'UTC':
        logger.debug(f"{main_log_prefix} Index de df_source_enriched non UTC ({df_working_copy.index.tz}). Conversion en UTC.")
        df_working_copy.index = df_working_copy.index.tz_convert('UTC') # type: ignore

    missing_base_cols = [col for col in BASE_OHLCV_OUTPUT_COLUMNS if col not in df_working_copy.columns]
    if missing_base_cols:
        logger.error(f"{main_log_prefix} Colonnes OHLCV de base manquantes dans df_source_enriched : {missing_base_cols}. "
                     "Impossible de continuer.")
        return pd.DataFrame(index=df_working_copy.index, columns=BASE_OHLCV_OUTPUT_COLUMNS)

    df_results = df_working_copy[BASE_OHLCV_OUTPUT_COLUMNS].copy()
    logger.debug(f"{main_log_prefix} DataFrame de résultats initialisé avec OHLCV de base. Shape: {df_results.shape}")

    for col_base_ohlcv in BASE_OHLCV_OUTPUT_COLUMNS:
        try:
            df_results[col_base_ohlcv] = pd.to_numeric(df_results[col_base_ohlcv], errors='raise')
        except ValueError as e_conv_base:
            logger.error(f"{main_log_prefix} Colonne de base '{col_base_ohlcv}' non convertible en numérique: {e_conv_base}. "
                         "Le calcul des indicateurs pourrait échouer ou produire des NaNs.")
            df_results[col_base_ohlcv] = np.nan

    all_expected_output_strat_cols: List[str] = []
    for cfg_init_out in required_indicator_configs:
        # Sourcery: Replace calls to `dict.items` with `dict.values` when the keys are not used
        if outputs_cfg_val_init := cfg_init_out.get('outputs'): # Sourcery: Use named expression
            if isinstance(outputs_cfg_val_init, str): 
                all_expected_output_strat_cols.append(outputs_cfg_val_init)
            elif isinstance(outputs_cfg_val_init, dict): 
                all_expected_output_strat_cols.extend(list(outputs_cfg_val_init.values()))
    
    for expected_col_name_init in all_expected_output_strat_cols:
        if expected_col_name_init not in df_results.columns:
            df_results[expected_col_name_init] = np.nan
    logger.debug(f"{main_log_prefix} Colonnes _strat de sortie initialisées à NaN: {all_expected_output_strat_cols}")

    for indicator_config in required_indicator_configs:
        indicator_name_orig = indicator_config.get('indicator_name')
        indicator_params = indicator_config.get('params', {})
        input_mapping = indicator_config.get('inputs', {}) 
        output_config = indicator_config.get('outputs')

        if not (
            isinstance(indicator_name_orig, str) and indicator_name_orig and
            isinstance(indicator_params, dict) and
            isinstance(input_mapping, dict) and
            output_config and isinstance(output_config, (str, dict)) # Sourcery: Merge isinstance calls
        ):
            logger.warning(f"{main_log_prefix} Configuration d'indicateur invalide ou incomplète : {indicator_config}. Indicateur ignoré.")
            continue

        indicator_name_lower = indicator_name_orig.lower()
        current_indicator_log_prefix = f"{main_log_prefix}[{indicator_name_orig}]"
        logger.info(f"{current_indicator_log_prefix} Calcul avec params: {indicator_params}, entrées mappées: {input_mapping}, config sorties: {output_config}")

        ta_input_kwargs: Dict[str, pd.Series] = {}
        valid_inputs_for_ta_call = True
        
        if not input_mapping and indicator_name_lower not in ['true_range']:
             logger.debug(f"{current_indicator_log_prefix} input_mapping est vide.")

        for ta_expected_arg_name, source_col_name_in_df_enriched in input_mapping.items():
            if source_col_name_in_df_enriched not in df_working_copy.columns:
                logger.error(f"{current_indicator_log_prefix} Colonne source '{source_col_name_in_df_enriched}' "
                             f"(pour l'entrée TA '{ta_expected_arg_name}') non trouvée dans df_source_enriched. "
                             f"Colonnes disponibles: {df_working_copy.columns.tolist()}. Indicateur ignoré.")
                valid_inputs_for_ta_call = False
                break 
            
            series_data_from_source = df_working_copy[source_col_name_in_df_enriched]
            if series_data_from_source.isnull().all():
                logger.warning(f"{current_indicator_log_prefix} Colonne source '{source_col_name_in_df_enriched}' "
                               f"(pour entrée TA '{ta_expected_arg_name}') est entièrement NaN.")
            
            try:
                ta_input_kwargs[ta_expected_arg_name.lower()] = series_data_from_source.astype(float)
            except ValueError as e_astype_indic:
                logger.error(f"{current_indicator_log_prefix} Impossible de convertir la colonne source '{source_col_name_in_df_enriched}' "
                             f"en float pour l'entrée TA '{ta_expected_arg_name}': {e_astype_indic}. Indicateur ignoré.")
                valid_inputs_for_ta_call = False
                break
        
        if not valid_inputs_for_ta_call:
            continue

        default_ta_inputs_map = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
        for ta_default_arg_name, base_col_name_df in default_ta_inputs_map.items():
            if ta_default_arg_name not in ta_input_kwargs and base_col_name_df in df_working_copy.columns:
                try:
                    ta_input_kwargs[ta_default_arg_name] = df_working_copy[base_col_name_df].astype(float)
                    logger.debug(f"{current_indicator_log_prefix} Utilisation de la colonne de base '{base_col_name_df}' pour l'entrée TA implicite '{ta_default_arg_name}'.")
                except ValueError:
                    logger.warning(f"{current_indicator_log_prefix} Échec conversion colonne de base '{base_col_name_df}' pour entrée TA implicite '{ta_default_arg_name}'.")

        primary_input_series_for_check: Optional[pd.Series] = ta_input_kwargs.get('close')
        if primary_input_series_for_check is None and ta_input_kwargs: 
            primary_input_series_for_check = next(iter(ta_input_kwargs.values()), None)

        if primary_input_series_for_check is None or not primary_input_series_for_check.notna().any():
            logger.warning(f"{current_indicator_log_prefix} L'entrée principale pour l'indicateur est vide ou entièrement NaN.")

        indicator_function: Optional[Any] = getattr(ta, indicator_name_lower, None)
        if indicator_function is None:
            for category_module_name in ['trend', 'momentum', 'overlap', 'volume', 'volatility', 'cycles', 'statistics', 'transform', 'performance', 'candles']:
                # Sourcery: Use named expression
                if (category_module := getattr(ta, category_module_name, None)) and hasattr(category_module, indicator_name_lower):
                    indicator_function = getattr(category_module, indicator_name_lower)
                    logger.debug(f"{current_indicator_log_prefix} Fonction indicateur '{indicator_name_lower}' trouvée dans ta.{category_module_name}.")
                    break
        
        if indicator_function is None:
            logger.error(f"{current_indicator_log_prefix} Fonction indicateur '{indicator_name_lower}' non trouvée dans pandas_ta.")
            continue

        try:
            indicator_result = indicator_function(**ta_input_kwargs, **indicator_params, append=False)

            if isinstance(indicator_result, pd.Series):
                if isinstance(output_config, str): 
                    df_results[output_config] = indicator_result.reindex(df_results.index)
                    logger.debug(f"{current_indicator_log_prefix} Indicateur (Series) '{indicator_name_orig}' ajouté comme colonne '{output_config}'. NaNs: {df_results[output_config].isnull().sum()}/{len(df_results)}")
                else:
                    logger.error(f"{current_indicator_log_prefix} L'indicateur '{indicator_name_orig}' a retourné une Series, "
                                 f"mais la configuration 'outputs' n'est pas une chaîne de caractères : {output_config}")
            
            elif isinstance(indicator_result, pd.DataFrame): 
                if isinstance(output_config, dict): 
                    for conceptual_key, desired_output_col_name_strat in output_config.items():
                        actual_col_from_ta_df = _map_conceptual_output_keys_to_ta_df_cols(
                            indicator_result.columns.tolist(), 
                            conceptual_key, 
                            indicator_name_orig,
                            indicator_params 
                        )
                        
                        if actual_col_from_ta_df and actual_col_from_ta_df in indicator_result.columns:
                            df_results[desired_output_col_name_strat] = indicator_result[actual_col_from_ta_df].reindex(df_results.index)
                            logger.debug(f"{current_indicator_log_prefix} Colonne TA '{actual_col_from_ta_df}' (de '{indicator_name_orig}') "
                                         f"mappée depuis clé '{conceptual_key}' et ajoutée comme '{desired_output_col_name_strat}'. NaNs: {df_results[desired_output_col_name_strat].isnull().sum()}/{len(df_results)}")
                        else:
                            logger.warning(f"{current_indicator_log_prefix} Colonne de sortie TA pour clé '{conceptual_key}' "
                                           f"non trouvée/mappable dans DataFrame de '{indicator_name_orig}'. "
                                           f"Colonnes TA: {indicator_result.columns.tolist()}. "
                                           f"'{desired_output_col_name_strat}' restera NaN.")
                else:
                    logger.error(f"{current_indicator_log_prefix} L'indicateur '{indicator_name_orig}' a retourné un DataFrame, "
                                 f"mais 'outputs' n'est pas un dict de mapping : {output_config}")
            
            elif indicator_result is None:
                 logger.warning(f"{current_indicator_log_prefix} Calcul de '{indicator_name_orig}' a retourné None.")
            else: 
                logger.warning(f"{current_indicator_log_prefix} Résultat inattendu pour '{indicator_name_orig}'. Type: {type(indicator_result)}.")

        except Exception as e_calc_indic_call:
            logger.error(f"{current_indicator_log_prefix} Erreur lors du calcul de l'indicateur '{indicator_name_orig}' avec params {indicator_params}: {e_calc_indic_call}", exc_info=True)
    
    for final_check_col_name in all_expected_output_strat_cols:
        if final_check_col_name not in df_results.columns:
             logger.warning(f"{main_log_prefix} Colonne _strat finale attendue '{final_check_col_name}' toujours manquante après tous les calculs. Ajout avec NaN.")
             df_results[final_check_col_name] = np.nan

    logger.info(f"{main_log_prefix} Calcul de tous les indicateurs terminé. Shape final df_results: {df_results.shape}")
    logger.debug(f"{main_log_prefix} Colonnes dans df_results: {df_results.columns.tolist()}")
    
    return df_results
