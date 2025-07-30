# analise_de_erros.py
import json
import datetime 
from typing import Dict, Any, List

# Importamos as ferramentas de análise que já construímos
import analisador

def _gerar_perfil_numero(numero: int, analise_contexto: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gera um perfil de sinais para um número específico com base em uma análise já realizada.
    """
    if not analise_contexto or 'numbers_stats' not in analise_contexto:
        return {"erro": "Contexto de análise insuficiente."}

    perfil = {}
    stats_num = analise_contexto['numbers_stats'].get(numero, {})
    
    # Sinais básicos
    perfil['gap'] = stats_num.get('gap', 'N/A')
    dev = stats_num.get('deviation', 0)
    perfil['deviation'] = f"{dev:.2%}"
    if dev > 0.5:
        perfil['calor'] = "Muito Quente"
    elif dev < -0.5:
        perfil['calor'] = "Muito Frio"

    # Sinais de contexto (ex: era um seguidor?)
    last_number_no_contexto = analise_contexto.get('last_number')
    if last_number_no_contexto is not None:
        followers = analise_contexto.get('intelligence', {}).get('follower_analysis', {}).get('top_followers', [])
        if any(f['num'] == numero for f in followers):
            perfil['seguidor'] = f"Sim, do {last_number_no_contexto}"

    # Sinais de posição na roda
    wheel_jumps = analise_contexto.get('intelligence', {}).get('wheel_jump_analysis', {})
    if wheel_jumps.get('most_common_jumps'):
        wheel_pos = {num: i for i, num in enumerate(analisador.ROULETTE_WHEEL_ORDER)}
        if last_number_no_contexto in wheel_pos:
            last_pos = wheel_pos[last_number_no_contexto]
            current_pos = wheel_pos.get(numero)
            if current_pos is not None:
                diff = abs(current_pos - last_pos)
                jump = min(diff, 37 - diff)
                if any(j['jump'] == jump for j in wheel_jumps['most_common_jumps']):
                    perfil['salto_comum'] = f"Sim, salto de {jump}"

    return perfil


def analisar_veredicto_passado(
    predicao_anterior: Dict[str, Any],
    analise_anterior: Dict[str, Any],
    numero_real: int,
    prediction_reasons: List[str] = None
) -> Dict[str, Any]:
    """
    Analisa uma predição passada, compara com o resultado real e gera uma conclusão.
    Retorna um dicionário pronto para ser salvo no banco de dados.
    """
    # << MELHORIA >> Inicializa variáveis para garantir que sempre tenham um valor
    outcome_quality = 'NO_BET'
    was_heuristic_hit = False

    predicted_ml_numbers = [p['number'] for p in predicao_anterior.get('predictions', [])]
    was_ml_hit = numero_real in predicted_ml_numbers

    # << MELHORIA >> Lógica de análise de erro reestruturada para maior clareza
    # Analisa o resultado apenas se uma aposta heurística foi de fato recomendada e registrada
    verdict = analise_anterior.get('intelligence', {}).get('analyst_bulletin', {}).get('final_verdict')
    if verdict and verdict.get('stake_units', 0) > 0 and prediction_reasons:
        was_heuristic_hit = numero_real in verdict.get('numbers', [])
        
        if was_heuristic_hit:
            outcome_quality = "HIT"
        else:
            outcome_quality = "MISS" # Define 'MISS' como padrão se a aposta foi feita e errou.
            
            # Verifica se foi um "quase acerto" na roda
            vizinhos_da_aposta = []
            for n in verdict.get('numbers', []):
                try:
                    idx = analisador.ROULETTE_WHEEL_ORDER.index(n)
                    # Adiciona os dois vizinhos de cada número da aposta
                    vizinhos_da_aposta.append(analisador.ROULETTE_WHEEL_ORDER[(idx - 1) % 37])
                    vizinhos_da_aposta.append(analisador.ROULETTE_WHEEL_ORDER[(idx + 1) % 37])
                except ValueError:
                    continue

            # Se o número real for um dos vizinhos, classifica como NEAR_MISS
            if numero_real in set(vizinhos_da_aposta):
                outcome_quality = "NEAR_MISS_WHEEL"

    # Gera o perfil do número que realmente caiu para entender o cenário
    perfil_ganhador = _gerar_perfil_numero(numero_real, analise_anterior)

    # << MELHORIA >> Conclusão da IA agora comenta sobre ambos os sistemas (ML e Heurístico)
    conclusao_parts = []
    if was_ml_hit:
        conclusao_parts.append(f"✅ ACERTO DO ML! O número {numero_real} estava entre os previstos.")
    else:
        conclusao_parts.append(f"❌ ERRO DO ML. Previsão foi {predicted_ml_numbers}, resultado foi {numero_real}.")

    if prediction_reasons: # Se houve uma aposta heurística
        if was_heuristic_hit:
            conclusao_parts.append(f"🎯 ACERTO HEURÍSTICO na aposta em '{verdict.get('name')}'.")
        else:
            conclusao_parts.append(f"❌ ERRO HEURÍSTICO na aposta em '{verdict.get('name')}'.")
    
    # Adiciona contexto sobre o número ganhador se o ML errou
    if not was_ml_hit:
        profile_str = json.dumps(perfil_ganhador, ensure_ascii=False, sort_keys=True)
        conclusao_parts.append(f"Lição: O perfil do número ganhador que foi subestimado era: {profile_str}.")

    conclusao = " ".join(conclusao_parts)
    
    return {
        "predicted_numbers": json.dumps(predicted_ml_numbers),
        "actual_number": numero_real,
        "was_hit": was_heuristic_hit, # Representa o acerto da aposta que foi de fato realizada
        "outcome_quality": outcome_quality,
        "winning_number_profile": json.dumps(perfil_ganhador, ensure_ascii=False),
        "ai_conclusion": conclusao,
        "prediction_reasons": json.dumps(prediction_reasons or []),
        "timestamp": datetime.datetime.now()
    }