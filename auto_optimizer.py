# auto_optimizer.py
import logging
import json
from typing import List, Dict
from collections import Counter
import config 

# Mapeia os códigos de "raw_reasons" para os pesos correspondentes no arquivo ia_strategists.json
MAPEAMENTO_SINAL_PESO = {
    # Sinais Contrarian
    'Gap(x)': ("contrarian", "GAP_RATIO"),
    'Frio(%)': ("contrarian", "DEVIATION"),
    'Tensão': ("contrarian", "REGRESSION_TENSION"),
    'Nº Frio': ("contrarian", "COLD_NUMBER_RANK"),
    'Pressão(x)': ("contrarian", "PRESSURE"),
    # Sinais Momentum
    'Nº Quente': ("momentum", "HOT_NUMBER_RANK"),
    'Streak(x)': ("momentum", "STREAK"),
    'Seguidor': ("momentum", "FOLLOWER"),
    'Terreno Quente': ("momentum", "TERRAIN_MOMENTUM"),
    'Alternância(z)': ("momentum", "ALTERNATING_PATTERN_STREAK"),
    'Salto Roda': ("momentum", "COMMON_WHEEL_JUMP"),
    'Pós-Gêmeo': ("momentum", "POST_TWIN_FOLLOWER"),
    'Setor Quente(S)': ("momentum", "SESSION_SECTOR_HEAT"),
    'Final(S)': ("momentum", "FINAL_STREAK"),
    'Espelho(S)': ("momentum", "MIRROR_FOLLOWER"),
    # Sinais de ML (não ajustáveis por heurística, mas listados para completude)
    'ML-Grupo': ("fusion", "ML_CONVICTION_BONUS"),
    'ML-Nº': ("fusion", "ML_NUMBER_CONVERGENCE"),
}

def analisar_licoes_e_propor_ajustes(licoes: List[Dict]) -> Dict[str, float]:
    """
    Analisa uma lista de vereditos (lições) do banco de dados, calcula a taxa de
    sucesso de cada sinal heurístico e propõe ajustes de peso para cima ou para baixo.
    Leva em consideração a qualidade do erro (ex: "quase acertos").
    """
    import config
    if not licoes:
        return {}

    logging.info(f"Analisando {len(licoes)} vereditos para auto-otimização com análise de 'quase acertos'...")

    acertos_por_sinal = Counter()
    erros_por_sinal = Counter()

    # 1. Contabiliza acertos e erros para cada sinal que participou de uma aposta
    for licao in licoes:
        try:
            # Pula vereditos que não resultaram em uma aposta real
            if licao.get('prediction_reasons') is None:
                continue
            
            reasons_str = licao.get('prediction_reasons', '[]')
            # Garante que a string não é vazia antes de tentar decodificar
            if not reasons_str or reasons_str == 'null':
                continue
                
            razoes = json.loads(reasons_str)
            if not razoes:
                continue
            
            peso_licao = 1 / (len(razoes) + 1)
            foi_acerto = licao.get('was_hit', False)

            for razao in razoes:
                razao_limpa = ''.join(filter(lambda c: not c.isdigit() and c not in '()', razao))

                if foi_acerto:
                    acertos_por_sinal[razao_limpa] += peso_licao
                else:
                    # Se não foi um acerto, verifica a qualidade do erro
                    outcome_quality = licao.get('outcome_quality', 'MISS')

                    if outcome_quality == 'NEAR_MISS_WHEEL':
                        # Aplica a penalidade reduzida para "quase acertos"
                        penalidade = peso_licao * config.PENALTY_REDUCTION_FACTOR_FOR_NEAR_MISS
                        erros_por_sinal[razao_limpa] += penalidade
                    else:
                        # Aplica a penalidade completa para erros normais
                        erros_por_sinal[razao_limpa] += peso_licao

        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(f"Não foi possível decodificar 'prediction_reasons' para o veredito ID {licao.get('id')}: {e}")
            continue

    ajustes_propostos = {}
    
    # 2. Itera sobre todos os sinais conhecidos e avalia sua performance
    todos_os_sinais = set(acertos_por_sinal.keys()) | set(erros_por_sinal.keys())
    
    for sinal in todos_os_sinais:
        if sinal not in MAPEAMENTO_SINAL_PESO:
            continue # Ignora sinais não mapeados como 'ML-Grupo'

        acertos = acertos_por_sinal[sinal]
        erros = erros_por_sinal[sinal]
        total_observacoes = acertos + erros

        if total_observacoes < config.MIN_OBSERVACOES_PARA_AJUSTE:
            continue # Ignora sinais com poucos dados para evitar ajustes por flutuações

        taxa_sucesso = acertos / total_observacoes
        
        tipo_peso, nome_peso = MAPEAMENTO_SINAL_PESO[sinal]
        chave_ajuste = f"{tipo_peso}.{nome_peso}"

        # 3. Propõe o ajuste com base na performance do sinal
        if taxa_sucesso > config.TAXA_SUCESSO_PARA_AUMENTO:
            magnitude_ajuste = config.PERCENTUAL_AJUSTE * (1 + (taxa_sucesso - config.TAXA_SUCESSO_PARA_AUMENTO) * 2)
            ajustes_propostos[chave_ajuste] = min(magnitude_ajuste, config.PERCENTUAL_AJUSTE * 3) # Limita o ajuste máximo
            logging.warning(
                f"PROPOSTA DE AUMENTO: Sinal '{sinal}' tem {taxa_sucesso:.0%} de sucesso ({acertos}/{total_observacoes}). Propondo +{config.PERCENTUAL_AJUSTE:.0%} para o peso '{chave_ajuste}'."
            )
        elif taxa_sucesso < (1 - config.TAXA_FALHA_PARA_REDUCAO):
            taxa_falha = 1 - taxa_sucesso
            magnitude_ajuste = -config.PERCENTUAL_AJUSTE * (1 + (taxa_falha - config.TAXA_FALHA_PARA_REDUCAO) * 2)
            ajustes_propostos[chave_ajuste] = max(magnitude_ajuste, -config.PERCENTUAL_AJUSTE * 3) # Limita a redução máxima
            logging.warning(
                f"PROPOSTA DE REDUÇÃO: Sinal '{sinal}' tem apenas {taxa_sucesso:.0%} de sucesso ({acertos}/{total_observacoes}). Propondo -{config.PERCENTUAL_AJUSTE:.0%} para o peso '{chave_ajuste}'."
            )

    if not ajustes_propostos:
        logging.info("Nenhum ajuste de peso atingiu os limiares de otimização nesta análise.")

    return ajustes_propostos