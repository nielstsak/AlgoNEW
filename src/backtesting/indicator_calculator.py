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

def calculate_indicators_for_trial(
    df_source_enriched: pd.DataFrame,
    required_indicator_configs: List[Dict[str, Any]],
    log_prefix_context: Optional[str] = ""
) -> pd.DataFrame:
    """
    Calcule dynamiquement une liste d'indicateurs techniques sur un DataFrame source enrichi.

    Args:
        df_source_enriched (pd.DataFrame): Le DataFrame source contenant les données OHLCV
            de base (1-minute) et potentiellement des K-lines agrégées pré-calculées
            (ex: 'Klines_5min_close', 'Klines_1h_high').
            L'index doit être un DatetimeIndex (UTC attendu).
        required_indicator_configs (List[Dict[str, Any]]): Une liste de dictionnaires,
            où chaque dictionnaire décrit un indicateur à calculer.
            Structure attendue pour chaque config d'indicateur :
            {
                'indicator_name': str,      # Nom de l'indicateur (ex: 'ema', 'rsi', 'bbands')
                'params': Dict[str, Any],   # Paramètres pour la fonction de l'indicateur
                                            # (ex: {'length': 20, 'std': 2})
                'inputs': Dict[str, str],   # Map des entrées attendues par pandas-ta
                                            # (ex: 'close', 'high', 'volume')
                                            # vers les noms de colonnes réels dans df_source_enriched
                                            # (ex: {'close': 'Klines_5min_close', 'volume': 'Klines_5min_volume'})
                                            # Si une entrée TA standard comme 'close' n'est pas mappée,
                                            # elle sera prise depuis les colonnes de base 1-min.
                'outputs': Union[str, Dict[str, str]] # Nom de la colonne de sortie (ex: "EMA_FAST_strat")
                                                      # OU un dictionnaire de mapping pour les indicateurs
                                                      # multi-sorties (ex: pour bbands,
                                                      # {'BBL_length_std': 'BB_LOWER_strat', ...}
                                                      # ou des clés conceptuelles comme {'lower': 'BB_LOWER_strat'})
            }
        log_prefix_context (Optional[str]): Préfixe de log optionnel pour le contexte (ex: trial ID).

    Returns:
        pd.DataFrame: Un DataFrame contenant les colonnes OHLCV de base (1-min)
                      du df_source_enriched original et les nouvelles colonnes
                      d'indicateurs (suffixées par `_strat` comme défini dans `outputs`).
                      L'index est préservé. Retourne un DataFrame vide avec les colonnes
                      OHLCV de base si une erreur critique survient.
    """
    main_log_prefix = f"[IndicatorCalc]{log_prefix_context if log_prefix_context else ''}"
    logger.info(f"{main_log_prefix} Démarrage du calcul des indicateurs pour le trial/contexte.")

    if df_source_enriched.empty:
        logger.warning(f"{main_log_prefix} df_source_enriched est vide. Retour d'un DataFrame vide.")
        return pd.DataFrame(columns=BASE_OHLCV_OUTPUT_COLUMNS)

    if not isinstance(df_source_enriched.index, pd.DatetimeIndex):
        logger.error(f"{main_log_prefix} L'index de df_source_enriched doit être un DatetimeIndex.")
        return pd.DataFrame(columns=BASE_OHLCV_OUTPUT_COLUMNS)
    if df_source_enriched.index.tz is None:
        logger.warning(f"{main_log_prefix} L'index de df_source_enriched n'a pas de timezone. "
                       "Supposition UTC. Il est fortement recommandé d'avoir un index UTC.")
        # df_source_enriched.index = df_source_enriched.index.tz_localize('UTC') # Optionnel, pourrait masquer des problèmes en amont

    # Préparer le DataFrame de résultat avec les colonnes OHLCV de base (1-min)
    missing_base_cols = [col for col in BASE_OHLCV_OUTPUT_COLUMNS if col not in df_source_enriched.columns]
    if missing_base_cols:
        logger.error(f"{main_log_prefix} Colonnes OHLCV de base manquantes dans df_source_enriched : {missing_base_cols}. "
                     "Impossible de continuer sans ces colonnes de base.")
        return pd.DataFrame(index=df_source_enriched.index, columns=BASE_OHLCV_OUTPUT_COLUMNS)

    df_results = df_source_enriched[BASE_OHLCV_OUTPUT_COLUMNS].copy()
    logger.debug(f"{main_log_prefix} DataFrame de résultats initialisé avec OHLCV de base. Shape: {df_results.shape}")

    for idx_cfg in required_indicator_configs:
        indicator_name_orig = idx_cfg.get('indicator_name')
        indicator_params = idx_cfg.get('params', {})
        input_mapping = idx_cfg.get('inputs', {})
        output_config = idx_cfg.get('outputs')

        if not indicator_name_orig or not isinstance(indicator_params, dict) or \
           not isinstance(input_mapping, dict) or not output_config:
            logger.warning(f"{main_log_prefix} Configuration d'indicateur invalide ou incomplète : {idx_cfg}. Indicateur ignoré.")
            continue

        indicator_name_lower = str(indicator_name_orig).lower()
        current_indicator_log_prefix = f"{main_log_prefix}[{indicator_name_orig}]"
        logger.info(f"{current_indicator_log_prefix} Calcul avec params: {indicator_params}, entrées: {input_mapping}, sorties: {output_config}")

        ta_input_series: Dict[str, pd.Series] = {}
        valid_inputs_for_ta = True
        for ta_expected_input_name, source_column_name_in_df in input_mapping.items():
            if source_column_name_in_df not in df_source_enriched.columns:
                logger.error(f"{current_indicator_log_prefix} Colonne source '{source_column_name_in_df}' (pour entrée TA '{ta_expected_input_name}') "
                             "non trouvée dans df_source_enriched. Indicateur ignoré.")
                valid_inputs_for_ta = False
                break
            
            series_data = df_source_enriched[source_column_name_in_df]
            if series_data.isnull().all():
                logger.warning(f"{current_indicator_log_prefix} Colonne source '{source_column_name_in_df}' (pour '{ta_expected_input_name}') est entièrement NaN.")
                # Laisser pandas-ta gérer cela, mais c'est un avertissement.
            
            try:
                ta_input_series[ta_expected_input_name.lower()] = series_data.astype(float) # pandas-ta attend souvent des float et des clés en minuscules
            except ValueError as e_astype:
                logger.error(f"{current_indicator_log_prefix} Impossible de convertir la colonne source '{source_column_name_in_df}' en float pour l'entrée TA '{ta_expected_input_name}': {e_astype}. Indicateur ignoré.")
                valid_inputs_for_ta = False
                break
        
        if not valid_inputs_for_ta:
            continue

        # Si certaines entrées TA standard (open, high, low, close, volume) ne sont pas dans input_mapping,
        # essayer de les prendre depuis les colonnes de base 1-min.
        default_ta_inputs = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
        for ta_default_name, base_col_name in default_ta_inputs.items():
            if ta_default_name not in ta_input_series and base_col_name in df_source_enriched.columns:
                try:
                    ta_input_series[ta_default_name] = df_source_enriched[base_col_name].astype(float)
                    logger.debug(f"{current_indicator_log_prefix} Utilisation de la colonne de base 1-min '{base_col_name}' pour l'entrée TA '{ta_default_name}'.")
                except ValueError: # Devrait être rare si BASE_OHLCV_OUTPUT_COLUMNS sont déjà numériques
                    logger.warning(f"{current_indicator_log_prefix} Échec de la conversion de la colonne de base '{base_col_name}' pour l'entrée TA '{ta_default_name}'.")


        # Vérifier si l'entrée principale (souvent 'close') a des données valides
        primary_input_key_for_check = 'close' if 'close' in ta_input_series else (list(ta_input_series.keys())[0] if ta_input_series else None)
        if not primary_input_key_for_check or not ta_input_series.get(primary_input_key_for_check, pd.Series(dtype=float)).notna().any(): # type: ignore
            logger.warning(f"{current_indicator_log_prefix} L'entrée principale ('{primary_input_key_for_check}') pour l'indicateur est vide ou entièrement NaN. "
                           "Le résultat de l'indicateur sera probablement NaN.")
            # Ne pas arrêter, laisser pandas-ta retourner NaN.
        
        try:
            indicator_function = getattr(ta, indicator_name_lower, None)
            if indicator_function is None:
                # Tentative de recherche dans les sous-modules si certains indicateurs y sont encore
                for category_module_name in ['trend', 'momentum', 'overlap', 'volume', 'volatility', 'cycles', 'statistics', 'transform', 'utils']:
                    category_module = getattr(ta, category_module_name, None)
                    if category_module and hasattr(category_module, indicator_name_lower):
                        indicator_function = getattr(category_module, indicator_name_lower)
                        logger.debug(f"{current_indicator_log_prefix} Fonction indicateur '{indicator_name_lower}' trouvée dans ta.{category_module_name}.")
                        break
            
            if indicator_function is None:
                logger.error(f"{current_indicator_log_prefix} Fonction indicateur '{indicator_name_lower}' non trouvée dans pandas_ta (ni top-level, ni sous-modules communs).")
                continue

            # `append=False` est crucial pour obtenir seulement les séries d'indicateurs
            indicator_result = indicator_function(**ta_input_series, **indicator_params, append=False) # type: ignore

            if isinstance(indicator_result, pd.Series):
                if isinstance(output_config, str):
                    df_results[output_config] = indicator_result.reindex(df_results.index)
                    logger.debug(f"{current_indicator_log_prefix} Indicateur (Series) '{indicator_name_orig}' ajouté comme colonne '{output_config}'.")
                else:
                    logger.error(f"{current_indicator_log_prefix} L'indicateur '{indicator_name_orig}' a retourné une Series, mais 'outputs' n'est pas une chaîne : {output_config}")
            
            elif isinstance(indicator_result, pd.DataFrame):
                if isinstance(output_config, dict):
                    for ta_result_col_name, desired_output_col_name_strat in output_config.items():
                        # ta_result_col_name est la clé du dict 'outputs', qui devrait correspondre
                        # à un nom de colonne dans indicator_result (ou une clé conceptuelle).
                        # pandas-ta retourne souvent des noms de colonnes formatés (ex: "BBL_20_2.0").
                        # La config 'outputs' doit mapper ces noms (ou des clés conceptuelles)
                        # aux noms finaux _strat.
                        
                        # Tentative de match direct
                        actual_col_from_ta_df = None
                        if ta_result_col_name in indicator_result.columns:
                            actual_col_from_ta_df = ta_result_col_name
                        else:
                            # Tentative de match partiel/conceptuel (ex: 'lower' pour 'BBL_...')
                            # Ceci est une heuristique et peut nécessiter d'être affiné.
                            for col_in_ta_df in indicator_result.columns:
                                if ta_result_col_name.lower() in col_in_ta_df.lower(): # Ex: 'bbl' dans 'BBL_20_2.0'
                                    actual_col_from_ta_df = col_in_ta_df
                                    logger.debug(f"{current_indicator_log_prefix} Clé de sortie conceptuelle '{ta_result_col_name}' "
                                                 f"mappée à la colonne TA réelle '{actual_col_from_ta_df}'.")
                                    break
                        
                        if actual_col_from_ta_df and actual_col_from_ta_df in indicator_result.columns:
                            df_results[desired_output_col_name_strat] = indicator_result[actual_col_from_ta_df].reindex(df_results.index)
                            logger.debug(f"{current_indicator_log_prefix} Colonne TA '{actual_col_from_ta_df}' (de '{indicator_name_orig}') "
                                         f"ajoutée comme '{desired_output_col_name_strat}'.")
                        else:
                            logger.warning(f"{current_indicator_log_prefix} Colonne de sortie attendue via clé '{ta_result_col_name}' "
                                           f"non trouvée dans le DataFrame retourné par '{indicator_name_orig}'. "
                                           f"Colonnes TA disponibles: {indicator_result.columns.tolist()}. "
                                           f"Colonne _strat '{desired_output_col_name_strat}' sera NaN.")
                            df_results[desired_output_col_name_strat] = np.nan # Assurer que la colonne existe
                else:
                    logger.error(f"{current_indicator_log_prefix} L'indicateur '{indicator_name_orig}' a retourné un DataFrame, "
                                 f"mais 'outputs' n'est pas un dictionnaire de mapping : {output_config}")
            
            elif indicator_result is None:
                 logger.warning(f"{current_indicator_log_prefix} Calcul de '{indicator_name_orig}' a retourné None. "
                                "Cela peut arriver si les données d'entrée sont insuffisantes ou invalides.")
                 if isinstance(output_config, str): df_results[output_config] = np.nan
                 elif isinstance(output_config, dict):
                     for out_name in output_config.values(): df_results[out_name] = np.nan
            else:
                logger.warning(f"{current_indicator_log_prefix} Résultat inattendu pour '{indicator_name_orig}'. Type: {type(indicator_result)}. "
                               "Attendu Series ou DataFrame.")

        except Exception as e_calc_indic:
            logger.error(f"{current_indicator_log_prefix} Erreur lors du calcul de '{indicator_name_orig}' avec params {indicator_params}: {e_calc_indic}", exc_info=True)
            if isinstance(output_config, str): df_results[output_config] = np.nan
            elif isinstance(output_config, dict):
                for out_name in output_config.values(): df_results[out_name] = np.nan
    
    # S'assurer que toutes les colonnes _strat attendues existent, même si remplies de NaN
    all_expected_output_strat_cols: List[str] = []
    for cfg in required_indicator_configs:
        outputs_cfg = cfg.get('outputs')
        if isinstance(outputs_cfg, str):
            all_expected_output_strat_cols.append(outputs_cfg)
        elif isinstance(outputs_cfg, dict):
            all_expected_output_strat_cols.extend(outputs_cfg.values())
    
    for expected_col in all_expected_output_strat_cols:
        if expected_col not in df_results.columns:
            logger.debug(f"{main_log_prefix} Colonne de sortie attendue '{expected_col}' non présente après calculs. Ajout avec NaN.")
            df_results[expected_col] = np.nan

    logger.info(f"{main_log_prefix} Calcul de tous les indicateurs terminé. Shape final df_results: {df_results.shape}")
    logger.debug(f"{main_log_prefix} Colonnes dans df_results: {df_results.columns.tolist()}")
    
    return df_results
