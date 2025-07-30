# config.py
"""
Arquivo de configuração central para todos os parâmetros ajustáveis da aplicação.
"""

# --- Configurações Gerais da Aplicação ---
IA_STRATEGISTS_FILE = 'ia_strategists.json'
DATABASE_URL = 'sqlite:///roleta_data.db'
HISTORICO_MAX_LEN = 1000  # Tamanho máximo do histórico em memória para análise.
PREDICTION_OUTCOMES_MAX_LEN = 500 # Máximo de resultados de apostas para manter em memória.
OPTIMIZATION_LOG_MAX_LEN = 100 # Máximo de logs de otimização para manter em memória.

# --- Configurações do Auto-Otimizador ---
MIN_OBSERVACOES_PARA_AJUSTE = 2  # Um sinal precisa aparecer em pelo menos 5 apostas para ser ajustado.
TAXA_SUCESSO_PARA_AUMENTO = 0.60  # Se a taxa de sucesso for > 60%, aumenta o peso.
TAXA_FALHA_PARA_REDUCAO = 0.65   # Se a taxa de falha for > 65% (sucesso < 35%), diminui o peso.
PERCENTUAL_AJUSTE = 0.05          # Aumenta ou diminui o peso em 5%.

# --- Configurações do Machine Learning (ML) ---
MIN_HISTORY_FOR_TRAINING = 150 # Histórico mínimo de giros para iniciar o treinamento do ML.
N_TOP_FEATURES = 75            # Número de features mais importantes a serem selecionadas para o modelo final.
CONFIDENCE_THRESHOLD = 0.04    # Limiar de confiança para uma predição de número ser considerada válida (4%).

# 0.5 significa que um "quase erro" conta como metade de um erro completo.
PENALTY_REDUCTION_FACTOR_FOR_NEAR_MISS = 0.5 
# --- Configurações de Gatilhos (Triggers) ---
SPINS_FOR_ML_RETRAIN = 50       # Retreinar o modelo de ML a cada 50 giros.
SPINS_FOR_AUTO_OPTIMIZE = 25    # Rodar a auto-otimização a cada 25 giros.

# --- Configurações de Análise de Sessão ---
MIN_SPINS_FOR_SESSION_ANALYSIS = 15 # Giros mínimos para iniciar a análise de viés do dealer/mesa.