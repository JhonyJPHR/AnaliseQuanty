import logging
from collections import Counter
from typing import List, Dict, Any

# Importa lógicas e constantes que já temos no analisador principal
from analisador import _analyze_wheel_jumps, _analyze_followers, NEIGHBOR_BETS, ROULETTE_WHEEL_ORDER
import config

def analyze_session(session_spins: List[int]) -> Dict[str, Any]:
    """
    Analisa uma lista de giros pertencentes a uma única sessão/dealer.
    Foca em padrões que podem indicar viés físico ou de lançamento.
    """
    total_spins = len(session_spins)
    if total_spins < config.MIN_SPINS_FOR_SESSION_ANALYSIS:
        return {"status": f"Aguardando mais dados da sessão ({total_spins}/{config.MIN_SPINS_FOR_SESSION_ANALYSIS} giros)."}

    # Reutiliza análises que já criamos, mas aplicadas apenas aos dados da sessão
    wheel_jump_analysis = _analyze_wheel_jumps(session_spins)
    follower_analysis = _analyze_followers(session_spins)

    # Análise de "calor" dos setores da roda (Voisins, Tiers, Orphelins) na sessão
    sector_heat = {}
    for name, data in NEIGHBOR_BETS.items():
        expected_hits = total_spins * data['prob']
        actual_hits = sum(1 for num in session_spins if num in data['numbers'])
        
        # Calcula o desvio em relação ao esperado para esta sessão
        deviation = (actual_hits - expected_hits) / expected_hits if expected_hits > 0 else 0
        
        sector_heat[name] = {
            "hits": actual_hits,
            "expected": f"{expected_hits:.1f}",
            "deviation": deviation # Desvio positivo significa que o setor está "quente" na sessão
        }
    
    # Encontra o setor mais quente
    hottest_sector = None
    if sector_heat:
        hottest_sector = max(sector_heat, key=lambda name: sector_heat[name]['deviation'])

    return {
        "status": "Análise de sessão ativa.",
        "total_session_spins": total_spins,
        "wheel_jumps": wheel_jump_analysis,
        "followers": follower_analysis,
        "sector_heat": sector_heat,
        "hottest_sector": {
            "name": hottest_sector,
            "deviation": sector_heat[hottest_sector]['deviation']
        } if hottest_sector else None
    }