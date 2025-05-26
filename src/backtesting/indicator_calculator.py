# src/backtesting/indicator_calculator.py
"""
Ce module est responsable du calcul dynamique des séries d'indicateurs techniques
(ex: EMA, RSI, PSAR) sur un DataFrame donné, en utilisant les paramètres
spécifiques d'un "trial" d'optimisation. Il est principalement appelé par
ObjectiveFunctionEvaluator.
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
            et potentiellement des K-lines agrégées pré-calculées.
            L'index doit être un DatetimeIndex (UTC attendu).
        required_indicator_configs (List[Dict[str, Any]]): Une liste de dictionnaires,
            où chaque dictionnaire décrit un indicateur à calculer.
            Structure attendue pour chaque config d'indicateur :
            {
                'indicator_name': str,      # Nom de l'indicateur (ex: 'ema', 'rsi', 'bbands')
                'params': Dict[str, Any],   # Paramètres pour la fonction de l'indicateur
                'inputs': Dict[str, str],   # Map des entrées attendues par pandas-ta (ex: 'close', 'high')
                                            # vers les noms de colonnes réels dans df_source_enriched
                                            # (ex: {'close': 'Klines_5min_close'})
                'outputs': Union[str, Dict[str, str]] # Nom de la colonne de sortie (_strat) ou map
                                                      # pour les indicateurs multi-sorties.
            }
        log_prefix_context (Optional[str]): Préfixe de log optionnel pour le contexte (ex: trial ID).

    Returns:
        pd.DataFrame: Un DataFrame contenant les colonnes OHLCV de base (1-min)
                      du df_source_enriched original et les nouvelles colonnes
                      d'indicateurs (_strat) calculées. L'index est préservé.
    """
    main_log_prefix = f"[IndicatorCalc]{log_prefix_context if log_prefix_context else ''}"
    logger.info(f"{main_log_prefix} Démarrage du calcul des indicateurs pour le trial.")

    if df_source_enriched.empty:
        logger.warning(f"{main_log_prefix} df_source_enriched est vide. Retour d'un DataFrame vide.")
        return pd.DataFrame()

    if not isinstance(df_source_enriched.index, pd.DatetimeIndex):
        logger.error(f"{main_log_prefix} L'index de df_source_enriched doit être un DatetimeIndex.")
        # Tenter de convertir si une colonne 'timestamp' existe, sinon retourner vide
        if 'timestamp' in df_source_enriched.columns:
            try:
                df_source_enriched = df_source_enriched.set_index(pd.to_datetime(df_source_enriched['timestamp'], errors='coerce', utc=True))
                df_source_enriched = df_source_enriched.drop(columns=['timestamp'])
                if not isinstance(df_source_enriched.index, pd.DatetimeIndex) or df_source_enriched.index.hasnans:
                    raise ValueError("Conversion de la colonne timestamp en DatetimeIndex a échoué ou contient des NaNs.")
            except Exception as e_ts:
                logger.error(f"{main_log_prefix} Échec de la conversion de la colonne 'timestamp' en index : {e_ts}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()


    # Préparer le DataFrame de résultat avec les colonnes OHLCV de base (1-min)
    # Ces colonnes doivent exister dans df_source_enriched.
    missing_base_cols = [col for col in BASE_OHLCV_OUTPUT_COLUMNS if col not in df_source_enriched.columns]
    if missing_base_cols:
        logger.error(f"{main_log_prefix} Colonnes OHLCV de base manquantes dans df_source_enriched : {missing_base_cols}. Impossible de continuer.")
        return pd.DataFrame(index=df_source_enriched.index)

    df_results = df_source_enriched[BASE_OHLCV_OUTPUT_COLUMNS].copy()
    logger.debug(f"{main_log_prefix} DataFrame de résultats initialisé avec les colonnes OHLCV de base. Shape: {df_results.shape}")

    # Itérer sur chaque configuration d'indicateur requise
    for idx_cfg in required_indicator_configs:
        indicator_name_orig = idx_cfg.get('indicator_name')
        indicator_params = idx_cfg.get('params', {})
        input_mapping = idx_cfg.get('inputs', {}) # Ex: {'close': 'Klines_5min_close'}
        output_config = idx_cfg.get('outputs')     # Ex: "EMA_FAST_strat" ou {'BBL_...': 'BB_LOWER_strat', ...}

        if not indicator_name_orig or not isinstance(indicator_params, dict) or \
           not isinstance(input_mapping, dict) or not output_config:
            logger.warning(f"{main_log_prefix} Configuration d'indicateur invalide ou incomplète : {idx_cfg}. Indicateur ignoré.")
            continue

        indicator_name_lower = indicator_name_orig.lower()
        current_indicator_log_prefix = f"{main_log_prefix}[{indicator_name_orig}]"
        logger.info(f"{current_indicator_log_prefix} Calcul de l'indicateur avec params: {indicator_params}, entrées: {input_mapping}, sorties: {output_config}")

        # Préparer les séries d'entrée pour pandas-ta
        ta_input_series: Dict[str, pd.Series] = {}
        valid_inputs = True
        for ta_param_name, source_col_name in input_mapping.items():
            if source_col_name not in df_source_enriched.columns:
                logger.error(f"{current_indicator_log_prefix} Colonne source '{source_col_name}' pour l'entrée TA '{ta_param_name}' non trouvée dans df_source_enriched.")
                valid_inputs = False
                break
            
            series = df_source_enriched[source_col_name]
            if series.isnull().all():
                logger.warning(f"{current_indicator_log_prefix} Colonne source '{source_col_name}' (pour '{ta_param_name}') est entièrement NaN.")
                # Selon l'indicateur, cela peut être un problème ou non. pandas-ta gère souvent cela.
            
            # pandas-ta attend généralement des float pour les calculs
            try:
                ta_input_series[ta_param_name] = series.astype(float)
            except ValueError:
                logger.warning(f"{current_indicator_log_prefix} Impossible de convertir la colonne source '{source_col_name}' en float. Tentative avec pd.to_numeric.")
                ta_input_series[ta_param_name] = pd.to_numeric(series, errors='coerce')
                if ta_input_series[ta_param_name].isnull().all():
                    logger.error(f"{current_indicator_log_prefix} Conversion en numérique de '{source_col_name}' a résulté en une série de NaN.")
                    valid_inputs = False # Si une entrée critique devient entièrement NaN
                    break
        
        if not valid_inputs:
            logger.error(f"{current_indicator_log_prefix} Entrées invalides. Indicateur ignoré.")
            continue

        # Vérifier si l'entrée 'close' (ou la première entrée si 'close' n'est pas là) a des données non-NaN
        primary_input_key = 'close' if 'close' in ta_input_series else (list(ta_input_series.keys())[0] if ta_input_series else None)
        if not primary_input_key or not ta_input_series[primary_input_key].notna().any():
            logger.warning(f"{current_indicator_log_prefix} L'entrée principale ('{primary_input_key}') pour l'indicateur est vide ou entièrement NaN après préparation. L'indicateur pourrait retourner NaN.")
            # Ne pas arrêter ici, laisser pandas-ta gérer, mais loguer.

        # Calculer l'indicateur
        try:
            indicator_function = getattr(ta, indicator_name_lower, None)
            if indicator_function is None:
                # Essayer de chercher dans les sous-modules (ex: ta.trend.ema)
                # Ceci est moins courant avec les versions récentes de pandas-ta où la plupart sont top-level
                for category_module_name in ['trend', 'momentum', 'overlap', 'volume', 'volatility', 'cycles', 'statistics', 'transform', 'utils']:
                    category_module = getattr(ta, category_module_name, None)
                    if category_module and hasattr(category_module, indicator_name_lower):
                        indicator_function = getattr(category_module, indicator_name_lower)
                        break
            
            if indicator_function is None:
                logger.error(f"{current_indicator_log_prefix} Fonction indicateur '{indicator_name_lower}' non trouvée dans pandas_ta.")
                continue

            # L'option `append=False` est cruciale pour obtenir seulement les séries d'indicateurs
            indicator_result = indicator_function(**ta_input_series, **indicator_params, append=False) # type: ignore

            # Assigner le résultat au df_results
            if isinstance(indicator_result, pd.Series):
                if isinstance(output_config, str):
                    df_results[output_config] = indicator_result.reindex(df_results.index)
                    logger.debug(f"{current_indicator_log_prefix} Indicateur (Series) '{indicator_name_orig}' ajouté comme colonne '{output_config}'.")
                else:
                    logger.error(f"{current_indicator_log_prefix} L'indicateur '{indicator_name_orig}' a retourné une Series, mais 'outputs' n'est pas une chaîne : {output_config}")
            
            elif isinstance(indicator_result, pd.DataFrame):
                if isinstance(output_config, dict):
                    for ta_col_name, desired_output_name in output_config.items():
                        if ta_col_name in indicator_result.columns:
                            df_results[desired_output_name] = indicator_result[ta_col_name].reindex(df_results.index)
                            logger.debug(f"{current_indicator_log_prefix} Colonne d'indicateur '{ta_col_name}' (de '{indicator_name_orig}') ajoutée comme '{desired_output_name}'.")
                        else:
                            logger.warning(f"{current_indicator_log_prefix} Colonne de sortie attendue '{ta_col_name}' non trouvée dans le DataFrame retourné par '{indicator_name_orig}'. Colonnes disponibles: {indicator_result.columns.tolist()}")
                else:
                    logger.error(f"{current_indicator_log_prefix} L'indicateur '{indicator_name_orig}' a retourné un DataFrame, mais 'outputs' n'est pas un dictionnaire de mapping : {output_config}")
            
            elif indicator_result is None:
                 logger.warning(f"{current_indicator_log_prefix} Calcul de l'indicateur '{indicator_name_orig}' a retourné None. Cela peut arriver si les données d'entrée sont insuffisantes.")
                 # S'assurer que les colonnes de sortie sont créées avec NaN si elles étaient attendues
                 if isinstance(output_config, str):
                     df_results[output_config] = np.nan
                 elif isinstance(output_config, dict):
                     for desired_output_name in output_config.values():
                         df_results[desired_output_name] = np.nan

            else:
                logger.warning(f"{current_indicator_log_prefix} Résultat inattendu pour l'indicateur '{indicator_name_orig}'. Type: {type(indicator_result)}. Attendu Series ou DataFrame.")

        except Exception as e_calc:
            logger.error(f"{current_indicator_log_prefix} Erreur lors du calcul de l'indicateur '{indicator_name_orig}' avec params {indicator_params}: {e_calc}", exc_info=True)
            # S'assurer que les colonnes de sortie sont créées avec NaN en cas d'erreur
            if isinstance(output_config, str):
                df_results[output_config] = np.nan
            elif isinstance(output_config, dict):
                for desired_output_name in output_config.values():
                    df_results[desired_output_name] = np.nan
    
    # La méthode _calculate_indicators de BaseStrategy appliquera ffill si nécessaire.
    # Ici, nous retournons le DataFrame avec les indicateurs calculés, alignés sur l'index d'origine.
    logger.info(f"{main_log_prefix} Calcul de tous les indicateurs terminé. Shape final df_results: {df_results.shape}")
    logger.debug(f"{main_log_prefix} Colonnes dans df_results: {df_results.columns.tolist()}")
    
    return df_results

