import pandas as pd
import numpy as np
from collections import Counter
import logging
import datetime
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from analisador import ROULETTE_WHEEL_ORDER, GROUPS, get_color, calculate_gap
import analisador
import config
import threading
from xgboost import XGBClassifier

class MLStrategist:
    def __init__(self):
        self.models = {}
        self.is_trained = {}
        self.feature_names = {}
        self.performance_metrics = {}
        self.lock = threading.Lock()

    def _get_dozen_class(self, n):
        if 1 <= n <= 12: return 0
        if 13 <= n <= 24: return 1
        if 25 <= n <= 36: return 2
        return -1

    def _get_color_class(self, n):
        color = get_color(n)
        if color == 'red': return 0
        if color == 'black': return 1
        return -1

    def _get_number_class(self, n):
        return n

    def _create_feature_vector(self, history_slice: list, context: dict = None, heuristic_scores: dict = None) -> dict:
        features = {}
        total_spins = len(history_slice)
        last_num = history_slice[-1]
        features['lag_1'] = history_slice[-2] if len(history_slice) > 1 else -1
        features['lag_2'] = history_slice[-3] if len(history_slice) > 2 else -1
        features['gap_last_num'] = calculate_gap(history_slice, [last_num])
        counts = Counter(history_slice)
        for i in range(37):
            observed_freq = counts.get(i, 0) / total_spins
            features[f'dev_num_{i}'] = observed_freq - (1/37)
        for group_type, subgroups in GROUPS.items():
            for name, data in subgroups.items():
                clean_name = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
                features[f'gap_{group_type}_{clean_name}'] = calculate_gap(history_slice, data['numbers'])
        colors = [get_color(n) for n in history_slice]
        color_streak = 0
        for color in reversed(colors):
            if color == (colors[-1] if colors else ''):
                color_streak += 1
            else:
                break
        features['streak_color'] = color_streak
        if len(history_slice) > 1:
            try:
                wheel_positions = {num: i for i,num in enumerate(ROULETTE_WHEEL_ORDER)}
                last_idx = wheel_positions[last_num]
                prev_idx = wheel_positions[history_slice[-2]]
                jump = abs(last_idx - prev_idx)
                features['jump_distance'] = min(jump, len(ROULETTE_WHEEL_ORDER) - jump)
            except (ValueError, KeyError):
                features['jump_distance'] = -1
        else:
            features['jump_distance'] = -1
        if context:
            rhythm_map = {"hot": 1, "cold": -1, "neutral": 0}
            game_rhythm_context = context.get('game_rhythm', {})
            features['context_game_rhythm'] = rhythm_map.get(game_rhythm_context.get('context'), 0)
            session_analysis = context.get('session_analysis', {})
            if session_analysis and session_analysis.get('hottest_sector'):
                features['context_session_heat_deviation'] = session_analysis['hottest_sector'].get('deviation', 0)
            else:
                features['context_session_heat_deviation'] = 0
            twin_analysis = context.get('twin_analysis', {})
            features['context_is_twin'] = 1 if twin_analysis.get('is_current_a_twin') else 0
            alternating_patterns = context.get('alternating_patterns', {})
            if alternating_patterns:
                max_streak = max(p.get('streak', 0) for p in alternating_patterns.values())
                features['context_alternating_streak'] = max_streak
            else:
                features['context_alternating_streak'] = 0
        else:
            features['context_game_rhythm'] = 0
            features['context_session_heat_deviation'] = 0
            features['context_is_twin'] = 0
            features['context_alternating_streak'] = 0
        if heuristic_scores:
            features['context_heuristic_momentum_score'] = heuristic_scores.get('momentum_score', 0)
            features['context_heuristic_contrarian_score'] = heuristic_scores.get('contrarian_score', 0)
        else:
            features['context_heuristic_momentum_score'] = 0
            features['context_heuristic_contrarian_score'] = 0
        return features

    def tune_and_train(self, historico_completo: list, target_name: str):
        with self.lock:
            if len(historico_completo) < config.MIN_HISTORY_FOR_TRAINING:
                return

            logging.info(f"[{target_name}] Iniciando otimização e treinamento do modelo...")
            import json
            try:
                with open(config.IA_STRATEGISTS_FILE, 'r', encoding='utf-8') as f:
                    ia_strategists = json.load(f)
                default_weights = ia_strategists.get("O Equilibrista (Híbrido)", {}).get("weights", {})
            except Exception as e:
                logging.error(f"[{target_name}] Falha ao carregar estrategistas: {e}")
                default_weights = {}

            target_mappers = {'dozen': self._get_dozen_class, 'color': self._get_color_class, 'numbers': self._get_number_class}
            get_target_class = target_mappers.get(target_name)
            X, y_list = [], []
            start_index = max(50, config.MIN_HISTORY_FOR_TRAINING - 50)
            for i in range(start_index, len(historico_completo) - 1):
                history_slice = historico_completo[:i+1]
                target_number = historico_completo[i+1]
                analysis_at_i = analisador.analyze_historical_slice(history_slice, {}, active_strategy_weights=default_weights)
                context_at_i = analysis_at_i.get('intelligence', {})
                opp_scores = context_at_i.get('opportunity_scores', [])
                heuristic_scores_at_i = {
                    'momentum_score': next((s['score'] for s in opp_scores if s['profile'] == 'Momentum'), 0),
                    'contrarian_score': next((s['score'] for s in opp_scores if s['profile'] == 'Contrarian'), 0)
                }
                features = self._create_feature_vector(history_slice, context=context_at_i, heuristic_scores=heuristic_scores_at_i)
                target_class = get_target_class(target_number)
                if target_name in ['dozen', 'color'] and target_class == -1:
                    continue
                X.append(features)
                y_list.append(target_class)

            if not X or len(set(y_list)) < 2:
                logging.warning(f"[{target_name}] Dados de treinamento insuficientes.")
                return

            df_full = pd.DataFrame(X).fillna(-1)
            y_series = pd.Series(y_list)

            y_to_fit = y_series
            if target_name == 'numbers':
                y_to_fit = y_series.astype(pd.CategoricalDtype(categories=list(range(37))))

            model_params = {}
            if target_name == 'dozen':
                model_params['objective'] = 'multi:softprob'
                model_params['num_class'] = 3
            elif target_name == 'numbers':
                model_params['objective'] = 'multi:softprob'
                model_params['num_class'] = 37
                model_params['enable_categorical'] = True
            elif target_name == 'color':
                model_params['objective'] = 'binary:logistic'

            temp_model = XGBClassifier(random_state=42, n_estimators=100, **model_params)
            temp_model.fit(df_full, y_to_fit)

            importances = temp_model.feature_importances_
            num_features_to_select = min(config.N_TOP_FEATURES, len(df_full.columns))
            top_indices = np.argsort(importances)[-num_features_to_select:][::-1]
            top_feature_names = [df_full.columns[i] for i in top_indices]
            self.feature_names[target_name] = top_feature_names
            df_filtered = df_full[top_feature_names]

            best_model = None
            best_score = -1
            best_params = {} # <-- Initialize params dictionary

            if target_name == 'numbers':
                logging.info(f"[{target_name}] Usando treinamento simplificado (sem CV) para estabilidade.")
                # Define the fixed parameters used for the simplified model
                fixed_params = {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8}
                final_model = XGBClassifier(random_state=42, **fixed_params, **model_params)
                final_model.fit(df_filtered, y_to_fit)
                best_model = final_model
                best_score = 0
                # << THE FIX >> Add a placeholder for the parameters
                best_params = {"mode": "Simplified Training"}
            else:
                param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0]}
                tscv = TimeSeriesSplit(n_splits=3)
                xgb_model = XGBClassifier(random_state=42, **model_params)
                random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, n_iter=10, cv=tscv, n_jobs=1, random_state=42, scoring='accuracy')
                random_search.fit(df_filtered, y_to_fit)
                best_model = random_search.best_estimator_
                best_score = random_search.best_score_
                best_params = random_search.best_params_

            self.models[target_name] = best_model
            self.is_trained[target_name] = True
            self.performance_metrics[target_name] = {
                'best_cv_score': best_score,
                # This key is now guaranteed to exist for all models
                'best_params': best_params,
                'features_used': len(top_feature_names),
                'last_trained': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            }
            logging.info(f"[{target_name} | XGBoost] Treinamento concluído. Melhor Score CV: {best_score:.3f}")

    def predict(self, historico_atual: list, target_name: str, context: dict = None) -> dict:
        with self.lock:
            feature_names_for_model = self.feature_names.get(target_name)
            if not self.is_trained.get(target_name) or len(historico_atual) < 50 or not feature_names_for_model:
                return {"error": f"Modelo para '{target_name}' não treinado ou histórico insuficiente."}

            features_dict = self._create_feature_vector(historico_atual, context=context)
            features_df = pd.DataFrame([features_dict])[feature_names_for_model].fillna(-1)

            try:
                model = self.models[target_name]
                probabilities = model.predict_proba(features_df)[0]
                
                # FIX 2: Ensure keys are standard Python types for JSON serialization
                # For 'dozen' and 'color', the classes can be numpy integers.
                # We cast them to string to be safe JSON keys.
                prediction = {str(name): float(prob) for name, prob in zip(model.classes_, probabilities)}
                
                # For 'dozen' and 'color', map integer keys back to human-readable names
                if target_name == 'dozen':
                    dozen_map = {'0': "1ª Dúzia", '1': "2ª Dúzia", '2': "3ª Dúzia"}
                    prediction = {dozen_map.get(k, k): v for k, v in prediction.items()}
                elif target_name == 'color':
                    color_map = {'0': "Vermelho", '1': "Preto"}
                    prediction = {color_map.get(k, k): v for k, v in prediction.items()}

                best_bet = max(prediction, key=prediction.get)

                return {
                    "target_type": target_name.capitalize(),
                    "probabilities": prediction,
                    "best_bet": best_bet,
                    "confidence": prediction[best_bet]
                }
            except Exception as e:
                logging.error(f"Erro na predição de grupos: {e}", exc_info=True)
                return {"error": f"Falha ao gerar predição para '{target_name}'."}

    def predict_top_5_numbers(self, historico_atual: list, context: dict = None) -> dict:
        with self.lock:
            target_name = 'numbers'
            feature_names_for_model = self.feature_names.get(target_name)
            if not self.is_trained.get(target_name) or len(historico_atual) < 50 or not feature_names_for_model:
                return {"error": "Modelo para 'Números' não treinado ou histórico insuficiente."}

            features_dict = self._create_feature_vector(historico_atual, context=context)
            features_df = pd.DataFrame([features_dict])[feature_names_for_model].fillna(-1)

            try:
                model = self.models[target_name]
                probabilities = model.predict_proba(features_df)[0]
                
                predictions = []
                for pred_index, confidence in enumerate(probabilities):
                    if confidence >= config.CONFIDENCE_THRESHOLD:
                        # FIX 2: Ensure predicted number is a standard Python int
                        predicted_number = int(model.classes_[pred_index])
                        predictions.append({"number": predicted_number, "confidence": float(confidence)})
                
                predictions.sort(key=lambda x: x['confidence'], reverse=True)

                return {
                    "target_type": "Números de Alta Confiança",
                    "predictions": predictions
                }
            except Exception as e:
                logging.error(f"Erro na predição de Números: {e}", exc_info=True)
                return {"error": "Falha ao gerar predição de números."}