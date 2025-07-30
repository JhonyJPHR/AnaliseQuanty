# app.py
from flask import Flask, render_template, request, jsonify
import sys
from contextlib import contextmanager
import datetime
import threading
import logging
import json
import copy
import auto_optimizer
from collections import deque
from typing import Dict, List
from analysis_updater import AnalysisState
from threading import Thread # Usaremos a Thread do módulo threading
from multiprocessing import Lock # Mantenha o Lock do multiprocessing por enquanto
import multiprocessing 
from database import SessionLocal, Giro, init_db, AnaliseVeredito
import analise_de_erros
from bankroll_manager import BankrollManager
import dealer_analyzer
from ml_strategist import MLStrategist
import config
import analisador

app = Flask(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
bankroll_manager = BankrollManager()
current_session_id = None
session_lock = threading.Lock()
IA_STRATEGISTS_FILE = config.IA_STRATEGISTS_FILE
ml_model = MLStrategist()
TARGETS_TO_PREDICT = ['dozen', 'color', 'numbers']

analysis_state = AnalysisState()
analysis_cache = {}
cache_lock = threading.Lock()
prediction_outcomes = deque(maxlen=config.PREDICTION_OUTCOMES_MAX_LEN)
live_simulations = {}
simulation_lock = threading.Lock()

strategists_file_lock = threading.Lock()
optimization_log = deque(maxlen=config.OPTIMIZATION_LOG_MAX_LEN)
ia_strategists_cache = {}
ml_training_lock = Lock()
optimization_lock = Lock() # << MELHORIA >> Lock para evitar otimizações concorrentes


def load_ia_strategists_from_file():
    try:
        with open(IA_STRATEGISTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Erro CRÍTICO ao carregar o arquivo de estrategistas: {e}")
        return {}
    
@contextmanager
def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def load_ia_strategists():
    """
    << OTIMIZADO >> Retorna uma cópia profunda do cache de estrategistas em memória.
    Evita leituras repetidas do disco.
    """
    with strategists_file_lock:
        # Retorna uma cópia para que as modificações não afetem o cache original acidentalmente
        return copy.deepcopy(ia_strategists_cache)


def aplicar_ajustes_estrategia(ajustes: Dict[str, float], nome_estrategia: str):
    """
    << OTIMIZADO >> Aplica os ajustes de peso ao cache em memória e depois
    salva o estado atualizado no arquivo JSON.
    """
    global ia_strategists_cache # Informa que vamos modificar a variável global
    with strategists_file_lock:
        try:
            if nome_estrategia not in ia_strategists_cache:
                logging.error(f"Estratégia '{nome_estrategia}' não encontrada no cache para otimização.")
                return

            estrategia = ia_strategists_cache[nome_estrategia]
            logging.info(f"Aplicando ajustes à estratégia '{nome_estrategia}' em memória: {ajustes}")

            for chave, ajuste_percentual in ajustes.items():
                tipo_peso, nome_peso = chave.split('.')
                if tipo_peso in estrategia['weights'] and nome_peso in estrategia['weights'][tipo_peso]:
                    valor_antigo = estrategia['weights'][tipo_peso][nome_peso]
                    valor_novo = round(max(0, valor_antigo * (1 + ajuste_percentual)), 2)
                    # Modifica o cache diretamente
                    estrategia['weights'][tipo_peso][nome_peso] = valor_novo

                    log_entry = {
                        "timestamp": datetime.datetime.now().strftime('%d/%m %H:%M'),
                        "strategy": nome_estrategia,
                        "message": f"Peso '{chave}' ajustado de {valor_antigo} para {valor_novo} ({ajuste_percentual:+.0%})."
                    }
                    optimization_log.append(log_entry)
                    logging.warning(f"OTIMIZAÇÃO APLICADA: {log_entry['message']}")

            # Após modificar o cache, salva o estado completo no arquivo
            with open(IA_STRATEGISTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(ia_strategists_cache, f, indent=2, ensure_ascii=False)
            logging.info(f"Cache de estratégias atualizado salvo em '{IA_STRATEGISTS_FILE}'.")

        except Exception as e:
            logging.error(f"Falha crítica ao aplicar ajustes de estratégia: {e}", exc_info=True)


def executar_auto_otimizacao_async():
    """
    Thread que busca lições, analisa e aplica ajustes.
    """
    # << MELHORIA >> Adicionado lock para garantir execução única
    if not optimization_lock.acquire(block=False):
        logging.warning("Ciclo de otimização já está em andamento. Novo gatilho ignorado.")
        return
    
    try:
        logging.info("Iniciando ciclo de auto-otimização...")
        with get_db_session() as db:
            licoes_raw = db.query(AnaliseVeredito).order_by(
                AnaliseVeredito.timestamp.desc()).limit(50).all()  # Analisa um histórico maior de lições
            licoes = [l.__dict__ for l in licoes_raw]

        if not licoes:
            logging.info("Nenhuma lição no banco de dados para analisar.")
            return

        ajustes_propostos = auto_optimizer.analisar_licoes_e_propor_ajustes(
            licoes)

        if ajustes_propostos:
            with simulation_lock:
                if not live_simulations:
                    logging.warning(
                        "Otimização proposta, mas nenhuma simulação ativa para determinar a melhor estratégia.")
                    return
                relevant_strategists = {
                    name: data for name, data in live_simulations.items() if data['total_plays'] > 15}
                if not relevant_strategists:
                    logging.warning(
                        "Otimização proposta, mas nenhuma estratégia com dados de simulação suficientes.")
                    return
                best_strategist_name = max(
                    relevant_strategists, key=lambda name: relevant_strategists[name]['balance'])

            logging.info(
                f"Melhor estrategista segundo as simulações: '{best_strategist_name}'. Aplicando ajustes a ele.")
            aplicar_ajustes_estrategia(ajustes_propostos, best_strategist_name)
    finally:
        optimization_lock.release() # << MELHORIA >> Libera o lock


def initialize_live_simulations():
    """
    Carrega os estrategistas e inicializa o dicionário de simulações em tempo real.
    """
    global live_simulations
    strategists = load_ia_strategists()
    with simulation_lock:
        for name in strategists.keys():
            if name not in live_simulations:
                live_simulations[name] = {
                    'balance': 0, 'hits': 0, 'misses': 0,
                    'total_plays': 0, 'hit_rate': 0.0
                }
    logging.info(
        f"Simulações em tempo real inicializadas para {len(live_simulations)} estrategistas.")


def update_live_simulations(historico_anterior: list, numero_resultado: int):
    global live_simulations
    if not historico_anterior or len(historico_anterior) < 37:
        return

    strategists = load_ia_strategists()

    with simulation_lock:
        for name, data in strategists.items():
            strategy_weights = data.get('weights')
            if not strategy_weights or name not in live_simulations:
                continue

            analysis = analisador.analyze_historical_slice(
                historico_anterior, strategists, active_strategy_weights=strategy_weights)

            verdict = analysis.get('intelligence', {}).get(
                'analyst_bulletin', {}).get('final_verdict')

            if verdict and verdict.get('stake_units', 0) > 0:
                live_simulations[name]['total_plays'] += 1
                bet_numbers = verdict.get('numbers', [])
                stake = verdict.get('stake_units', 0)

                if numero_resultado in bet_numbers:
                    payout = (36 - len(bet_numbers)) * stake
                    live_simulations[name]['hits'] += 1
                    live_simulations[name]['balance'] += payout
                else:
                    loss = len(bet_numbers) * stake
                    live_simulations[name]['misses'] += 1
                    live_simulations[name]['balance'] -= loss

                total_plays = live_simulations[name]['total_plays']
                if total_plays > 0:
                    live_simulations[name]['hit_rate'] = (
                        live_simulations[name]['hits'] / total_plays) * 100


def select_dynamic_strategy() -> dict:
    with simulation_lock:
        if not live_simulations:
            return load_ia_strategists().get("O Equilibrista (Híbrido)", {}).get('weights')
        relevant_strategists = {
            name: data for name, data in live_simulations.items() if data['total_plays'] > 15}
        if not relevant_strategists:
            logging.info(
                "Meta-Estrategista: Aguardando mais dados. Usando 'O Equilibrista'.")
            return load_ia_strategists().get("O Equilibrista (Híbrido)", {}).get('weights')
        best_strategist_name = max(
            relevant_strategists, key=lambda name: relevant_strategists[name]['balance'])
        logging.info(
            f"Meta-Estrategista Dinâmico selecionou: '{best_strategist_name}'")
        return load_ia_strategists().get(best_strategist_name, {}).get('weights')


def train_ml_model_async():
    
    print(f"PROCESSO FILHO: Usando o executável Python em: {sys.executable}")
    print(f"PROCESSO FILHO: Caminhos de pacotes: {sys.path}")
    # Adquire a trava. Se já estiver em uso, não prossegue.
    if not ml_training_lock.acquire(block=False):
        logging.warning("Treinamento de ML já está em andamento. Novo gatilho ignorado.")
        return

    try:
        logging.info(
            "Disparando PROCESSO de treinamento de ML para todos os alvos...")
        with get_db_session() as db:
            giros = db.query(Giro).order_by(Giro.timestamp.asc()).all()
            numeros_historico = [giro.numero for giro in giros]

        if len(numeros_historico) >= config.MIN_HISTORY_FOR_TRAINING:
            for target in TARGETS_TO_PREDICT:
                ml_model.tune_and_train(numeros_historico, target_name=target)
        else:
            logging.info("Treinamento de ML adiado: histórico insuficiente.")
    
    finally:
        # Libera a trava ao final, independentemente de sucesso ou falha
        ml_training_lock.release()
        logging.info("Processo de treinamento de ML concluído e trava liberada.")



def update_analysis_cache():
    logging.info("Atualizando cache de análise com lógica incremental...")
    global analysis_cache
    try:
        # Pega o histórico mais recente direto do estado (que está em memória)
        numeros_historico_deque = analysis_state.historico
        numeros_historico = list(numeros_historico_deque)

        # Prepara o contexto para o ML e para a análise de sessão
        context = {}
        with session_lock:
            s_id = current_session_id
        if s_id:
            # A análise de sessão ainda precisa dos giros específicos da sessão
            with get_db_session() as db:
                session_giros_raw = db.query(Giro).filter(
                    Giro.session_id == s_id).order_by(Giro.timestamp.asc()).all()
                session_numeros = [g.numero for g in session_giros_raw]
                context['session_analysis'] = dealer_analyzer.analyze_session(
                    session_numeros)
        else:
            context['session_analysis'] = None

        # Roda as predições do ML (que operam sobre o histórico)
        group_predictions = []
        if len(numeros_historico) > 50:
            for target in ['dozen', 'color']:
                # << CORRIGIDO >> Chama o método 'predict' com o nome correto.
                prediction = ml_model.predict(
                    numeros_historico, target_name=target, context=context)
                if "error" not in prediction:
                    group_predictions.append(prediction)

        numbers_prediction = ml_model.predict_top_5_numbers(
            numeros_historico, context=context)
        best_group_prediction = max(
            group_predictions, key=lambda p: p['confidence']) if group_predictions else None

        # Seleciona a melhor estratégia com base nas simulações
        selected_weights = select_dynamic_strategy()

        session_analysis = context.get('session_analysis')

        if session_analysis and session_analysis.get('hottest_sector'):
            hottest_sector = session_analysis['hottest_sector']
            # Se o desvio do setor mais quente for muito alto (ex: > 50%)
            if hottest_sector['deviation'] > 0.5:
                logging.warning(f"VIÉS DE SESSÃO DETECTADO: Setor '{hottest_sector['name']}' está quente. Aumentando peso de Momentum.")
                # Cria uma cópia dos pesos para não alterar o original
                import copy
                modulated_weights = copy.deepcopy(selected_weights)
                # Aumenta temporariamente o peso de sinais de Momentum
                modulated_weights['momentum']['TERRAIN_MOMENTUM'] *= 1.5 # +50%
                modulated_weights['momentum']['STREAK'] *= 1.2 # +20%
                selected_weights = modulated_weights # Usa os pesos modulados para esta análise

        # Chama a análise final passando o estado já calculado
        final_analysis = analisador.analyze_statistics(
            state=analysis_state,
            ia_strategists=load_ia_strategists(),
            active_strategy_weights=selected_weights,
            ml_prediction=best_group_prediction,
            ml_numbers_prediction=numbers_prediction,
            session_analysis=context.get('session_analysis')
        )

        # Lógica para parar apostas se a banca estiver inativa
        if not bankroll_manager.is_active and bankroll_manager.status_message != "Sessão não iniciada.":
            if 'intelligence' in final_analysis and 'analyst_bulletin' in final_analysis['intelligence']:
                final_analysis['intelligence']['analyst_bulletin']['final_verdict'] = None
                final_analysis['intelligence']['analyst_bulletin']['summary'] = bankroll_manager.status_message

        # Garante que o objeto de performance sempre tenha a estrutura correta
        with cache_lock:
            # Define a estrutura padrão que o frontend espera
            default_performance = {'outcomes': [], 'total': 0, 'hits': 0}
            # Pega os dados de performance existentes, ou usa o padrão se não existir
            performance_data = analysis_cache.get(
                'ia_performance', default_performance)

            analysis_cache = final_analysis
            # Preserva a performance entre atualizações
            analysis_cache['ia_performance'] = performance_data

        logging.info("Cache de análise atualizado com sucesso.")
    except Exception as e:
        logging.error(
            f"Falha crítica ao atualizar o cache de análise: {e}", exc_info=True)


@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/dados')
def get_dados():
    with cache_lock:
        return jsonify(analysis_cache)


@app.route('/api/strategists')
def get_strategists():
    strategists = load_ia_strategists()
    strategist_info = {name: data.get(
        'description', 'Sem descrição.') for name, data in strategists.items()}
    return jsonify(strategist_info)


@app.route('/api/live_simulations')
def get_live_simulations():
    with simulation_lock:
        return jsonify(dict(live_simulations))


@app.route('/api/backtest', methods=['POST'])
def run_strategy_backtest():
    data = request.get_json()
    strategy_name = data.get('strategy_name')
    if not strategy_name:
        return jsonify({"error": "Nome da estratégia não fornecido."}), 400
    logging.info(f"Executando backtest para a estratégia: {strategy_name}")
    try:
        with get_db_session() as db:
            giros = db.query(Giro).order_by(Giro.timestamp.asc()).all()
            numeros_historico = [giro.numero for giro in giros]
        ia_strategists = load_ia_strategists()
        strategy_data = ia_strategists.get(strategy_name)
        if not strategy_data:
            return jsonify({"error": f"Estratégia '{strategy_name}' não encontrada."}), 404
        backtest_results = analisador.run_backtest(
            numeros_historico, strategy_data.get('weights'), ia_strategists)
        return jsonify(backtest_results)
    except Exception as e:
        logging.error(
            f"Erro durante o backtest para '{strategy_name}': {e}", exc_info=True)
        return jsonify({"error": "Ocorreu um erro interno durante o backtest."}), 500


@app.route('/api/giros', methods=['POST'])
def registrar_giro():
    try:
        data = request.get_json()
        if not data or 'numero' not in data or not isinstance(data['numero'], int) or not (0 <= data['numero'] <= 36):
            return jsonify({"erro": "Payload inválido."}), 400

        numero_resultado = data['numero']

        historico_antes_do_giro = list(analysis_state.historico)

        analysis_state.update_with_new_spin(numero_resultado)

        with cache_lock:
            analise_anterior = json.loads(json.dumps(analysis_cache))
        # Pega o veredito heurístico e suas razões
        veredicto_anterior = analise_anterior.get('intelligence', {}).get(
            'analyst_bulletin', {}).get('final_verdict')
        razoes_da_aposta = veredicto_anterior.get('raw_reasons', []) if veredicto_anterior and veredicto_anterior.get(
            'stake_units', 0) > 0 else None

        # Pega a predição do modelo de ML (para análise de erro do ML)
        predicao_numeros_anterior = analise_anterior.get(
            'intelligence', {}).get('ml_numbers_prediction')

        # 1. Analisa o veredito passado (tanto ML quanto heurístico)
        if predicao_numeros_anterior and not predicao_numeros_anterior.get("error"):
            resultado_analise_erro = analise_de_erros.analisar_veredicto_passado(
                predicao_anterior=predicao_numeros_anterior,
                analise_anterior=analise_anterior,
                numero_real=numero_resultado,
                prediction_reasons=razoes_da_aposta
            )
            if resultado_analise_erro:
                with get_db_session() as db:
                    nova_analise = AnaliseVeredito(**resultado_analise_erro)
                    db.add(nova_analise)
                    db.commit()

        # 2. Atualiza a performance oficial e a BANCA PRINCIPAL
        global prediction_outcomes
        if veredicto_anterior and veredicto_anterior.get('stake_units', 0) > 0:
            bet_numbers = veredicto_anterior.get('numbers', [])
            is_hit = numero_resultado in bet_numbers
            stake = veredicto_anterior.get('stake_units', 0)

            profit_or_loss = ((36 - len(bet_numbers))
                              * stake) if is_hit else (-len(bet_numbers) * stake)

            if bankroll_manager.is_active:
                bankroll_manager.update_balance(profit_or_loss)

            prediction_outcomes.append({
                'profile': veredicto_anterior.get('profile', 'N/A'),
                'hit': is_hit, 'prediction_name': veredicto_anterior.get('name', 'N/A'),
                'outcome_number': numero_resultado
            })
            with cache_lock:
                if 'ia_performance' not in analysis_cache:
                    analysis_cache['ia_performance'] = {}
                analysis_cache['ia_performance'] = {
                    'outcomes': list(prediction_outcomes),
                    'total': len(prediction_outcomes),
                    'hits': sum(1 for o in prediction_outcomes if o['hit']),
                }

        # 3. Atualiza as simulações
        if historico_antes_do_giro:
            update_live_simulations(historico_antes_do_giro, numero_resultado)

        # 4. Salva o novo giro
        with get_db_session() as db:
            with session_lock:
                s_id = current_session_id
            novo_giro = Giro(
                numero=numero_resultado,
                cor=analisador.get_color(numero_resultado),
                timestamp=datetime.datetime.now(datetime.UTC),
                session_id=s_id
            )
            db.add(novo_giro)
            db.commit()
            total_giros = db.query(Giro).count()

        # 5. Atualiza o cache para a próxima rodada
        update_analysis_cache()

        # 6. Dispara gatilhos de treinamento e otimização
        if total_giros > 0 and total_giros % config.SPINS_FOR_ML_RETRAIN == 0:
            logging.info(
                f"Gatilho de retreinamento do ML atingido ({total_giros} giros).")
            p = Thread(target=train_ml_model_async) 
            p.start()
        if total_giros > 0 and total_giros % config.SPINS_FOR_AUTO_OPTIMIZE == 0:
            logging.info(
                f"Gatilho de auto-otimização atingido ({total_giros} giros).")
            threading.Thread(target=executar_auto_otimizacao_async).start()

        return jsonify({"sucesso": f"Número {numero_resultado} registrado."}), 201
    except Exception as e:
        logging.error(f"Erro fatal ao registrar giro: {e}", exc_info=True)
        return jsonify({"erro": "Ocorreu um erro interno no servidor ao registrar o giro."}), 500


@app.route('/api/optimization_log')
def get_optimization_log():
    return jsonify(list(optimization_log))


@app.route('/api/giros/ultimo', methods=['DELETE'])
def deletar_ultimo_giro():
    """
    << OTIMIZADO >> Deleta o último giro do banco de dados e reverte o estado
    em memória sem precisar recarregar e re-simular todo o histórico.
    """
    try:
        with get_db_session() as db:
            ultimo_giro = db.query(Giro).order_by(
                Giro.timestamp.desc()).first()
            if ultimo_giro:
                logging.info(f"Removendo o giro ID {ultimo_giro.id} (número {ultimo_giro.numero}) do banco de dados.")
                # 1. Remove do banco de dados
                db.delete(ultimo_giro)
                db.commit()

                # 2. Reverte o estado principal em memória
                analysis_state.revert_last_spin()

                # 3. Limpa os resultados de predições recentes
                if prediction_outcomes:
                    prediction_outcomes.pop()

                # 4. Recalcula as simulações ao vivo (ainda a forma mais segura)
                #    usando o histórico já em memória (muito mais rápido)
                logging.info("Recalculando simulações ao vivo a partir do estado revertido...")
                initialize_live_simulations()
                hist_nums_revertido = list(analysis_state.historico)
                if hist_nums_revertido:
                    for i in range(min(37, len(hist_nums_revertido)), len(hist_nums_revertido)):
                        update_live_simulations(hist_nums_revertido[:i], hist_nums_revertido[i])
                
                # 5. Atualiza o cache de análise para refletir o estado revertido
                update_analysis_cache()
                
                return jsonify({"sucesso": "Último giro removido e estado recalculado eficientemente."}), 200
            else:
                return jsonify({"info": "Nenhum giro para remover."}), 404
    except Exception as e:
        logging.error(f"Erro ao deletar último giro: {e}", exc_info=True)
        return jsonify({"erro": "Não foi possível deletar o último giro."}), 500


@app.route('/api/analises_de_erros')
def get_analises_de_erros():
    try:
        with get_db_session() as db:
            analises = db.query(AnaliseVeredito).order_by(
                AnaliseVeredito.timestamp.desc()).limit(20).all()
            resultado = [{
                "timestamp": analise.timestamp.strftime('%d/%m/%Y %H:%M:%S'),
                "predicted_numbers": analise.predicted_numbers,
                "actual_number": analise.actual_number,
                "was_hit": analise.was_hit,
                "winning_number_profile": analise.winning_number_profile,
                "ai_conclusion": analise.ai_conclusion,
                "prediction_reasons": analise.prediction_reasons
            } for analise in analises]
            return jsonify(resultado)
    except Exception as e:
        logging.error(
            f"Erro ao buscar análises de erros: {e}", exc_info=True)
        return jsonify({"erro": "Não foi possível buscar os dados de análise."}), 500


@app.route('/api/ml_performance')
def get_ml_performance():
    return jsonify(ml_model.performance_metrics)


@app.route('/api/bankroll/status', methods=['GET'])
def get_bankroll_status():
    return jsonify(bankroll_manager.get_status())


@app.route('/api/bankroll/start', methods=['POST'])
def start_bankroll_session():
    data = request.get_json()
    try:
        initial = float(data.get('initial_bankroll', 100))
        stop_loss = float(data.get('stop_loss', 0.2))
        stop_win = float(data.get('stop_win', 0.4))
        bankroll_manager.start_session(initial, stop_loss, stop_win)
        return jsonify({"sucesso": "Sessão de banca iniciada."}), 200
    except (ValueError, TypeError) as e:
        return jsonify({"erro": f"Dados inválidos: {e}"}), 400


@app.route('/api/bankroll/stop', methods=['POST'])
def stop_bankroll_session():
    bankroll_manager.stop_session()
    return jsonify({"sucesso": "Sessão de banca parada."}), 200


@app.route('/api/session/status', methods=['GET'])
def get_session_status():
    with session_lock:
        return jsonify({"current_session_id": current_session_id})


@app.route('/api/session/start', methods=['POST'])
def start_session():
    global current_session_id
    data = request.get_json()
    session_name = data.get('session_name')
    if not session_name or len(session_name.strip()) == 0:
        return jsonify({"erro": "Nome da sessão não pode ser vazio."}), 400

    with session_lock:
        current_session_id = session_name.strip()

    logging.info(f"Sessão iniciada: '{current_session_id}'")
    update_analysis_cache()
    return jsonify({"sucesso": f"Sessão '{current_session_id}' iniciada."})


@app.route('/api/session/stop', methods=['POST'])
def stop_session():
    global current_session_id
    with session_lock:
        logging.info(f"Sessão parada: '{current_session_id}'")
        current_session_id = None
    update_analysis_cache()
    return jsonify({"sucesso": "Sessão parada."})


if __name__ == '__main__':
    print(f"PROCESSO PAI: Usando o executável Python em: {sys.executable}")
    print(f"PROCESSO PAI: Caminhos de pacotes: {sys.path}")
        
    if sys.platform == 'win32':
            multiprocessing.set_executable(sys.executable)
    # << MELHORIA >> Fluxo de inicialização consolidado e otimizado
    # 1. Inicializa o DB e carrega os estrategistas para o cache de memória
    init_db()
    logging.info(f"Carregando estrategistas do arquivo '{IA_STRATEGISTS_FILE}' para a memória...")
    ia_strategists_cache = load_ia_strategists_from_file()

    # 2. Carrega todo o histórico de giros do banco de dados UMA ÚNICA VEZ
    logging.info("Carregando histórico completo do banco de dados...")
    with get_db_session() as db:
        giros_hist = db.query(Giro).order_by(Giro.timestamp.asc()).all()
        hist_nums = [g.numero for g in giros_hist]
    logging.info(f"Histórico de {len(hist_nums)} giros carregado.")

    # 3. Constrói o estado de análise incremental a partir do histórico carregado
    logging.info("Construindo estado de análise inicial a partir do histórico...")
    analysis_state.initialize_from_history(hist_nums)

    # 4. Inicializa e preenche as simulações ao vivo usando o histórico já em memória
    logging.info("Inicializando e preenchendo simulações em tempo real com dados históricos...")
    initialize_live_simulations()
    if len(hist_nums) >= 37:
        for i in range(37, len(hist_nums)):
            # Passa a fatia do histórico ANTES do resultado para simular uma aposta
            update_live_simulations(hist_nums[:i], hist_nums[i])
    logging.info("Simulações históricas concluídas.")

    # 5. Carrega os resultados de apostas passadas para restaurar a métrica de performance da IA
    logging.info("Carregando histórico de performance da IA do banco de dados...")
    with get_db_session() as db:
        analises_passadas = db.query(AnaliseVeredito).order_by(AnaliseVeredito.timestamp.asc()).all()
        
        for analise in analises_passadas:
            # Considera apenas análises que representaram uma aposta real (tinham razões)
            reasons = json.loads(analise.prediction_reasons or '[]')
            if reasons:
                prediction_outcomes.append({
                    'profile': 'N/A', # Simplificação, pois não salvamos isso no DB
                    'hit': analise.was_hit,
                    'prediction_name': 'N/A', # Simplificação
                    'outcome_number': analise.actual_number
                })
    
    # Atualiza o cache com os dados de performance carregados
    if prediction_outcomes:
        with cache_lock:
            # Garante que a estrutura do cache exista
            if 'ia_performance' not in analysis_cache:
                analysis_cache['ia_performance'] = {}
            analysis_cache['ia_performance'] = {
                'outcomes': list(prediction_outcomes),
                'total': len(prediction_outcomes),
                'hits': sum(1 for o in prediction_outcomes if o['hit']),
            }
    logging.info(f"{len(prediction_outcomes)} resultados de performance carregados.")

    # 6. Dispara o treinamento inicial do ML em um processo separado se houver dados suficientes
    if len(hist_nums) >= config.MIN_HISTORY_FOR_TRAINING:
        logging.info("Disparando processo de treinamento inicial do ML em background.")
        p = Thread(target=train_ml_model_async)
        p.start()
    
    # 7. Gera o primeiro cache de análise para o dashboard
    update_analysis_cache()
    
    # 8. Inicia a aplicação Flask
    logging.info("Inicialização completa. Iniciando o servidor Flask.")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)