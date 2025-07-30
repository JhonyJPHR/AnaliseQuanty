# analisador.py (v2.1 - com correção de sintaxe)
from collections import Counter, defaultdict
from typing import List, Dict, Any
import math
import numpy as np  # << CORRIGIDO >>

# --- DEFINIÇÕES FUNDAMENTAIS ---
ROULETTE_WHEEL_ORDER = [
    0,
    32,
    15,
    19,
    4,
    21,
    2,
    25,
    17,
    34,
    6,
    27,
    13,
    36,
    11,
    30,
    8,
    23,
    10,
    5,
    24,
    16,
    33,
    1,
    20,
    14,
    31,
    9,
    22,
    18,
    29,
    7,
    28,
    12,
    35,
    3,
    26,
]
ROULETTE_COLORS = {
    0: "green",
    1: "red",
    2: "black",
    3: "red",
    4: "black",
    5: "red",
    6: "black",
    7: "red",
    8: "black",
    9: "red",
    10: "black",
    11: "black",
    12: "red",
    13: "black",
    14: "red",
    15: "black",
    16: "red",
    17: "black",
    18: "red",
    19: "red",
    20: "black",
    21: "red",
    22: "black",
    23: "red",
    24: "black",
    25: "red",
    26: "black",
    27: "red",
    28: "black",
    29: "black",
    30: "red",
    31: "black",
    32: "red",
    33: "black",
    34: "red",
    35: "black",
    36: "red",
}
ALL_NUMBERS = list(ROULETTE_COLORS.keys())
GROUPS = {
    "Dúzias": {
        "1ª Dúzia": {"numbers": list(range(1, 13)), "prob": 12 / 37},
        "2ª Dúzia": {"numbers": list(range(13, 25)), "prob": 12 / 37},
        "3ª Dúzia": {"numbers": list(range(25, 37)), "prob": 12 / 37},
    },
    "Colunas": {
        "1ª Coluna": {
            "numbers": [n for n in range(1, 37) if n % 3 == 1],
            "prob": 12 / 37,
        },
        "2ª Coluna": {
            "numbers": [n for n in range(1, 37) if n % 3 == 2],
            "prob": 12 / 37,
        },
        "3ª Coluna": {
            "numbers": [n for n in range(1, 37) if n % 3 == 0],
            "prob": 12 / 37,
        },
    },
    "Metades": {
        "Baixo (1-18)": {"numbers": list(range(1, 19)), "prob": 18 / 37},
        "Alto (19-36)": {"numbers": list(range(19, 37)), "prob": 18 / 37},
        "Vermelho": {
            "numbers": [n for n, c in ROULETTE_COLORS.items() if c == "red"],
            "prob": 18 / 37,
        },
        "Preto": {
            "numbers": [n for n, c in ROULETTE_COLORS.items() if c == "black"],
            "prob": 18 / 37,
        },
        "Par": {"numbers": [n for n in range(1, 37) if n % 2 == 0], "prob": 18 / 37},
        "Ímpar": {"numbers": [n for n in range(1, 37) if n % 2 != 0], "prob": 18 / 37},
    },
}
NEIGHBOR_BETS = {
    "Vizinhos do Zero (Voisins)": {
        "numbers": [22, 18, 29, 7, 28, 12, 35, 3, 26, 0, 32, 15, 19, 4, 21, 2, 25],
        "prob": 17 / 37,
    },
    "Terço do Cilindro (Tiers)": {
        "numbers": [27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33],
        "prob": 12 / 37,
    },
    "Órfãos (Orphelins)": {"numbers": [1, 20, 14, 31, 9, 17, 34, 6], "prob": 8 / 37},
}
GROUPS.update({"Apostas na Roda": NEIGHBOR_BETS})
MIRROR_PAIRS = {"12-21": [12, 21], "13-31": [13, 31], "23-32": [23, 32]}


# --- FUNÇÕES AUXILIARES ---
def get_color(number: int) -> str:
    return ROULETTE_COLORS.get(number, "desconhecido")


def get_properties(n: int) -> Dict:
    """Retorna um dicionário de propriedades para um número."""
    props = {}
    if n == 0:
        return {
            "color": "green",
            "parity": "none",
            "half": "none",
            "dozen": "none",
            "column": "none",
        }
    props["color"] = get_color(n)
    props["parity"] = "Par" if n % 2 == 0 else "Ímpar"
    props["half"] = "Baixo (1-18)" if 1 <= n <= 18 else "Alto (19-36)"
    if 1 <= n <= 12:
        props["dozen"] = "1ª Dúzia"
    elif 13 <= n <= 24:
        props["dozen"] = "2ª Dúzia"
    else:
        props["dozen"] = "3ª Dúzia"
    col = n % 3
    if col == 1:
        props["column"] = "1ª Coluna"
    elif col == 2:
        props["column"] = "2ª Coluna"
    else:
        props["column"] = "3ª Coluna"
    return props


def calculate_gap(historico: List[int], number_set: List[int]) -> int:
    if not historico:
        return 0
    try:
        return next(i for i, num in enumerate(reversed(historico)) if num in number_set)
    except StopIteration:
        return len(historico)

def _calculate_bet_volatility(numbers: List[int]) -> Dict[str, Any]:
    """
    Calcula um índice de volatilidade e uma descrição para um conjunto de números.
    Retorna um índice onde valores mais altos indicam maior risco/volatilidade.
    """
    count = len(numbers)
    if count >= 18:
        return {"index": 1.0, "description": "Baixo Risco"} # e.g., Cores, Par/Ímpar
    elif count >= 13:
        return {"index": 1.2, "description": "Baixo-Médio Risco"} # e.g., Vizinhos do Zero
    elif count >= 9:
        return {"index": 1.5, "description": "Médio Risco"} # e.g., Dúzias, Colunas
    elif count >= 7:
        return {"index": 2.0, "description": "Médio-Alto Risco"} # e.g., Órfãos
    elif count >= 5:
        return {"index": 2.5, "description": "Alto Risco"} # e.g., Linha
    elif count >= 2:
        return {"index": 4.0, "description": "Muito Alto Risco"} # e.g., Split, Canto
    elif count == 1:
        return {"index": 8.0, "description": "Risco Extremo"} # e.g., Pleno
    else:
        return {"index": 1.0, "description": "N/A"}
# --- MÓDULOS DE ANÁLISE ---


def _analyze_finals(historico: List[int]) -> Dict:
    if not historico:
        return {}
    last_number = historico[-1]
    if last_number == 0:
        return {"status": "Zero não possui final."}
    last_final = last_number % 10
    streak = 0
    for num in reversed(historico):
        if num > 0 and (num % 10) == last_final:
            streak += 1
        else:
            break
    numbers_in_final = [n for n in range(1, 37) if (n % 10) == last_final]
    return {
        "last_final": last_final,
        "current_streak": streak,
        "numbers": numbers_in_final,
        "gap": calculate_gap(historico, numbers_in_final),
    }


def _analyze_twins(historico: List[int]) -> Dict:
    if len(historico) < 2:
        return {}
    last_twin_gap = -1
    for i in range(len(historico) - 2, -1, -1):
        if historico[i] == historico[i + 1]:
            last_twin_gap = (len(historico) - 1) - (i + 1)
            break
    is_twin_now = historico[-1] == historico[-2]
    followers = []
    if len(historico) > 2:
        for i in range(len(historico) - 2):
            if historico[i] == historico[i + 1]:
                followers.append(historico[i + 2])
    post_twin_behavior = {}
    if followers:
        most_common_followers = Counter(followers).most_common(5)
        post_twin_behavior["top_followers"] = [
            {"num": num, "count": count} for num, count in most_common_followers
        ]
    else:
        post_twin_behavior["status"] = "Sem dados suficientes."
    return {
        "is_current_a_twin": is_twin_now,
        "last_twin_number": historico[-2] if is_twin_now else None,
        "gap_since_last_twin": (
            0
            if is_twin_now
            else (last_twin_gap if last_twin_gap != -1 else len(historico))
        ),
        "post_twin_analysis": post_twin_behavior,
    }


def _analyze_alternating_patterns(historico: List[int]) -> Dict[str, Any]:
    if len(historico) < 2:
        return {}
    patterns = {}
    properties_to_check = ["color", "parity", "half"]
    for prop in properties_to_check:
        streak = 0
        for i in range(len(historico) - 1, 0, -1):
            current_props = get_properties(historico[i])
            prev_props = get_properties(historico[i - 1])
            if historico[i] == 0 or historico[i - 1] == 0:
                if streak > 1:
                    break
                else:
                    continue
            if current_props.get(prop) != prev_props.get(prop):
                streak += 1
            else:
                break
        if streak >= 1:
            patterns[prop] = {
                "streak": streak + 1,
                "description": f"{get_properties(historico[-1]).get(prop, 'N/A')} / {get_properties(historico[-2]).get(prop, 'N/A')}",
            }
    return patterns


def _analyze_wheel_jumps(historico: List[int]) -> Dict[str, Any]:
    if len(historico) < 20:
        return {"status": "Aguardando mais dados..."}
    jumps = []
    wheel_pos = {num: i for i, num in enumerate(ROULETTE_WHEEL_ORDER)}
    for i in range(1, len(historico)):
        prev_num, current_num = historico[i - 1], historico[i]
        if prev_num not in wheel_pos or current_num not in wheel_pos:
            continue
        pos_prev = wheel_pos[prev_num]
        pos_current = wheel_pos[current_num]
        diff = abs(pos_current - pos_prev)
        jump = min(diff, 37 - diff)
        jumps.append(jump)
    if not jumps:
        return {}
    jump_counts = Counter(jumps)
    most_common = jump_counts.most_common(5)
    avg_jump = np.mean(jumps)
    std_dev_jump = np.std(jumps)
    return {
        "most_common_jumps": [{"jump": j, "count": c} for j, c in most_common],
        "average_jump": f"{avg_jump:.2f}",
        "std_dev": f"{std_dev_jump:.2f}",
        "last_jump": jumps[-1] if jumps else None,
    }


def _analyze_terrain(historico: List[int], numbers_stats: Dict) -> Dict:
    if not historico:
        return {}
    last_number = historico[-1]
    terrain = {}
    for name, data in NEIGHBOR_BETS.items():
        if last_number in data["numbers"]:
            terrain["name"] = name
            terrain["numbers"] = data["numbers"]
            break
    if not terrain:
        return {}
    terrain["gap"] = calculate_gap(historico, terrain["numbers"])
    terrain["total_hits"] = sum(numbers_stats[n]["hits"] for n in terrain["numbers"])
    terrain["last_number"] = last_number
    return terrain


def _generate_analyst_bulletin(
    game_rhythm: Dict,
    opportunity_scores: List[Dict],
    weights: Dict,
    bankroll_status: Dict = None,
    ia_performance: Dict = None

) -> Dict:
    if not opportunity_scores:
        return {
            "summary": "O jogo está equilibrado. A IA não identificou nenhuma anomalia estatística significativa para uma aposta de alta convicção no momento.",
            "confidence": "Baixa",
            "final_verdict": None,
            "conflict_warning": None,
        }

    top_opportunity = opportunity_scores[0]
    score = top_opportunity["normalized_score"]

    # --- LÓGICA DE STAKE BASE ---
    base_stake = 0.0
    stake_description = "Aposta de Observação"
    if score >= 9.5:
        base_stake = 3.0
        stake_description = "Confiança Máxima"
    elif score >= 8.0:
        base_stake = 2.0
        stake_description = "Aposta Forte"
    elif score >= 6.0:
        base_stake = 1.0
        stake_description = "Aposta Padrão"
    elif score >= 4.0:
        base_stake = 0.5
        stake_description = "Aposta de Baixa Confiança"
    elif score >= 3.5:
        base_stake = 0.25
        stake_description = "Aposta Exploratória"

    # --- AJUSTE DE PERFORMANCE E BANCA ---
    performance_factor = 1.0
    if ia_performance and ia_performance.get('total', 0) > 10:
        recent_outcomes = ia_performance['outcomes'][-10:]
        if recent_outcomes: # Garante que a lista não está vazia
            recent_hit_rate = sum(1 for o in recent_outcomes if o['hit']) / len(recent_outcomes)
            if recent_hit_rate > 0.7:
                performance_factor = 1.1
            elif recent_hit_rate < 0.3:
                performance_factor = 0.8
    
    final_stake = base_stake * performance_factor
    
    if bankroll_status and bankroll_status.get('is_active', False):
        stake_as_fraction_of_bankroll = bankroll_status['current_bankroll'] * 0.01
        if stake_as_fraction_of_bankroll > 0:
            final_stake = min(final_stake, stake_as_fraction_of_bankroll)

    # --- INICIALIZAÇÃO INCONDICIONAL DAS VARIÁVEIS DE VOLATILIDADE ---
    volatility_info = _calculate_bet_volatility(top_opportunity.get('numbers', []))
    volatility_index = volatility_info.get('index', 1.0)
    volatility_description = volatility_info.get('description', 'N/A')
    volatility_sensitivity = weights.get("fusion", {}).get("VOLATILITY_SENSITIVITY", 0.75)

    # --- AJUSTE FINAL PELA VOLATILIDADE ---
    # Ajusta o valor JÁ CALCULADO com base na volatilidade
    adjusted_stake_by_volatility = final_stake / (1 + (volatility_index - 1) * volatility_sensitivity)
    
    # Garante que o stake não seja ridiculamente pequeno e arredonda
    stake_units = max(round(adjusted_stake_by_volatility, 2), 0.0) if adjusted_stake_by_volatility > 0 else 0.0

    # --- CONSTRUÇÃO DO VEREDITO FINAL ---
    top_opportunity["stake_units"] = stake_units
    top_opportunity["stake_description"] = stake_description
    top_opportunity["volatility_description"] = volatility_description

    confidence = "Baixa"
    if score > 8.0:
        confidence = "Muito Alta"
    elif score > 6.0:
        confidence = "Alta"
    elif score > 4.0:
        confidence = "Moderada"

    summary = f"O jogo apresenta um ritmo **{game_rhythm.get('description', 'indefinido')}**. "
    if top_opportunity["profile"] == "Contrarian":
        summary += f"Os sinais de **Correção (Contrarian)** estão mais fortes."
    else:
        summary += f"Os sinais de **Tendência (Momentum)** estão mais fortes."

    if stake_units > 0:
        summary += f" A IA recomenda uma **{stake_description}** com risco **{volatility_description}**, ajustando a aposta para **{stake_units} unidades**."
    else:
        summary += " No momento, a IA recomenda apenas **observação**."

    conflict_warning = None
    if game_rhythm.get("context") == "hot" and top_opportunity["profile"] == "Contrarian":
        conflict_warning = "ALERTA DE TENSÃO: O ritmo do jogo é de 'Momentum', mas a oportunidade mais forte é 'Contrarian'. Apostar contra a tendência é um movimento de alto risco."
    elif game_rhythm.get("context") == "cold" and top_opportunity["profile"] == "Momentum":
        conflict_warning = "ALERTA DE CAUTELA: O ritmo do jogo é de 'Correção', mas a oportunidade mais forte é de 'Momentum'. A tendência pode estar perto do fim."

    return {
        "summary": summary,
        "confidence": confidence,
        "final_verdict": top_opportunity,
        "conflict_warning": conflict_warning,
    }


def _analyze_game_rhythm(historico: List[int], window_size: int = 37) -> Dict:
    if len(historico) < window_size:
        return {"context": "neutral", "description": "Aguardando mais dados..."}
    
    # << CORRIGIDO >> Converte a deque para uma lista antes de fatiar
    recent_history = list(historico)[-window_size:]
    
    counts = Counter(recent_history)
    repeated_hits = sum(1 for count in counts.values() if count > 1)
    total_unique_numbers = len(counts)
    repetition_index = (
        (repeated_hits / total_unique_numbers) * 100 if total_unique_numbers > 0 else 0
    )
    coverage_index = (total_unique_numbers / 37) * 100
    context, description = "neutral", "Ritmo Neutro / Equilibrado"
    if repetition_index > 40:
        context, description = (
            "hot",
            f"Ritmo Quente / Repetitivo 🔥 ({repetition_index:.0f}% de repetição)",
        )
    elif coverage_index > 85:
        context, description = (
            "cold",
            f"Ritmo Frio / Distribuído ❄️ ({coverage_index:.0f}% de cobertura)",
        )
    return {"context": context, "description": description}

def _calculate_conviction_bonus(signal_count):
    if signal_count <= 1:
        return 1.0
    # Bônus cresce com os sinais, mas com retornos decrescentes
    return 1.0 + math.log1p(signal_count - 1) * 0.5 

def _calculate_opportunity_scores(
    group_stats: Dict,
    numbers_stats: Dict,
    hot_numbers: List[Dict],
    cold_numbers: List[Dict],
    streak_analysis: Dict,
    follower_analysis: Dict,
    terrain_analysis: Dict,
    pressure_analysis: Dict,
    twin_analysis: Dict,
    alternating_patterns: Dict,
    wheel_jump_analysis: Dict,
    final_analysis: Dict,          # << NOVO >> Adicionado argumento
    mirror_analysis: Dict,        # << NOVO >> Adicionado argumento
    historico: List[int],
    total_spins: int,
    weights: Dict,
    ml_prediction: Dict = None,
    ml_numbers_prediction: Dict = None,
    session_analysis: Dict = None,
) -> List[Dict]:

    FUSION_WEIGHTS = weights.get("fusion", {})
    ML_GROUP_BONUS = FUSION_WEIGHTS.get("ML_CONVICTION_BONUS", 0)
    ML_NUMBER_BONUS = FUSION_WEIGHTS.get("ML_NUMBER_CONVERGENCE", 0)

    if not weights:
        return []

    CONTRARIAN_WEIGHTS = weights.get("contrarian", {})
    MOMENTUM_WEIGHTS = weights.get("momentum", {})
    CONVICTION_BONUS = {1: 1.0, 2: 1.2, 3: 1.5, 4: 1.8, 5: 2.2}
    opportunities = []

    all_groups = {}
    for group_type, subgroups in group_stats.items():
        if group_type == "Apostas na Roda":
            continue
        for name, stats in subgroups.items():
            all_groups[name] = {"stats": stats, "type": group_type}

    if terrain_analysis.get("name"):
        name = terrain_analysis["name"]
        all_groups[name] = {
            "stats": {
                "gap": terrain_analysis["gap"],
                "deviation": (
                    sum(
                        numbers_stats[n]["deviation"]
                        for n in terrain_analysis["numbers"]
                    )
                    / len(terrain_analysis["numbers"])
                    if terrain_analysis["numbers"]
                    else 0
                ),
                "prob": len(terrain_analysis["numbers"]) / 37,
                "numbers": terrain_analysis["numbers"],
            },
            "type": "Terreno",
        }

    last_number = historico[-1] # Pega o último número para as novas análises

    for name, data in all_groups.items():
        stats = data["stats"]
        score_c, signals_c, reasons_c = 0, 0, []
        score_m, signals_m, reasons_m = 0, 0, []
        group_numbers = set(stats.get("numbers", []))
        if not group_numbers:
            continue

        # --- Fatores de Fusão com ML ---
        if ml_prediction and not ml_prediction.get("error") and ML_GROUP_BONUS > 0:
            if name == ml_prediction.get("best_bet"):
                ml_conviction_score = (
                    ml_prediction.get("confidence", 0) * ML_GROUP_BONUS
                )
                score_c += ml_conviction_score
                score_m += ml_conviction_score
                reasons_c.append("ML-Grupo")
                reasons_m.append("ML-Grupo")

        # << CORRIGIDO >> Bloco inteiro com a indentação correta
        if (
            ml_numbers_prediction
            and not ml_numbers_prediction.get("error")
            and ML_NUMBER_BONUS > 0
        ):
            predicted_ml_numbers = {
                p["number"] for p in ml_numbers_prediction.get("predictions", [])
            }
            convergence = predicted_ml_numbers.intersection(group_numbers)

            if convergence:
                convergence_confidence = sum(
                    p["confidence"]
                    for p in ml_numbers_prediction["predictions"]
                    if p["number"] in convergence
                )
                convergence_score = convergence_confidence * ML_NUMBER_BONUS

                score_c += convergence_score
                score_m += convergence_score
                reasons_c.append(f"ML-Nº({len(convergence)})")
                reasons_m.append(f"ML-Nº({len(convergence)})")

        prob = stats.get("prob", 0)

        if prob == 0:
            continue
        group_numbers = set(stats.get("numbers", []))
        if not group_numbers:
            continue
        expected_gap = (1 / prob) - 1 if prob > 0 else float("inf")

        # Fatores Contrarian (sem alterações)
        if stats["gap"] > expected_gap:
            gap_ratio = stats["gap"] / expected_gap if expected_gap > 0 else 1
            gap_score = (gap_ratio - 1) * math.log(1 / prob) if prob > 0 else 0
            score_c += min(gap_score, 3.0) * CONTRARIAN_WEIGHTS.get("GAP_RATIO", 0)
            signals_c += 1
            reasons_c.append("Gap(x)")
        deviation = stats.get("deviation", 0)
        normalized_deviation = deviation / prob if prob else 0
        if normalized_deviation < -0.2:
            deviation_score = abs(normalized_deviation) / 10.0
            score_c += min(deviation_score, 1.5) * CONTRARIAN_WEIGHTS.get(
                "DEVIATION", 0
            )
            signals_c += 1
            reasons_c.append("Frio(%)")
            regression_tension = (
                (normalized_deviation**2) * math.log10(total_spins + 1) / 100
            )
            score_c += min(regression_tension, 2.0) * CONTRARIAN_WEIGHTS.get(
                "REGRESSION_TENSION", 0
            )
            signals_c += 1
            reasons_c.append("Tensão")
        cold_num_ranks = {c["num"]: 5 - i for i, c in enumerate(cold_numbers)}
        intersect_cold = [
            cold_num_ranks[n] for n in group_numbers if n in cold_num_ranks
        ]
        if intersect_cold:
            score_c += (max(intersect_cold) / 5) * CONTRARIAN_WEIGHTS.get(
                "COLD_NUMBER_RANK", 0
            )
            signals_c += 1
            reasons_c.append("Nº Frio")
        if name in pressure_analysis:
            score_c += min(
                pressure_analysis[name].get("streak", 0) / 10, 1.0
            ) * CONTRARIAN_WEIGHTS.get("PRESSURE", 0)
            signals_c += 1
            reasons_c.append("Pressão(x)")

        # Fatores Momentum (com novas lógicas)
        hot_num_ranks = {h["num"]: 5 - i for i, h in enumerate(hot_numbers)}
        intersect_hot = [hot_num_ranks[n] for n in group_numbers if n in hot_num_ranks]
        if intersect_hot:
            score_m += (max(intersect_hot) / 5) * MOMENTUM_WEIGHTS.get(
                "HOT_NUMBER_RANK", 0
            )
            signals_m += 1
            reasons_m.append("Nº Quente")
        if any(
            s.get("name") == name and s.get("count", 0) > 1
            for s in streak_analysis.values()
        ):
            streak_data = next(
                (s for s in streak_analysis.values() if s.get("name") == name), {}
            )
            score_m += min(streak_data.get("count", 0) / 4, 1.5) * MOMENTUM_WEIGHTS.get(
                "STREAK", 0
            )
            signals_m += 1
            reasons_m.append("Streak(x)")
        follower_num_set = {
            f["num"] for f in follower_analysis.get("top_followers", [])
        }
        if not follower_num_set.isdisjoint(group_numbers):
            score_m += MOMENTUM_WEIGHTS.get("FOLLOWER", 0)
            signals_m += 1
            reasons_m.append("Seguidor")
        if data["type"] == "Terreno":
            score_m += MOMENTUM_WEIGHTS.get("TERRAIN_MOMENTUM", 0)
            signals_m += 1
            reasons_m.append("Terreno Quente")

        # Lógica para usar os novos sinais heurísticos
        prop_map = {
            "Vermelho": "color",
            "Preto": "color",
            "Par": "parity",
            "Ímpar": "parity",
            "Baixo (1-18)": "half",
            "Alto (19-36)": "half",
        }
        if name in prop_map:
            prop_type = prop_map[name]
            if prop_type in alternating_patterns:
                pattern_data = alternating_patterns[prop_type]
                if pattern_data["streak"] > 2:
                    score_m += (pattern_data["streak"] / 5.0) * MOMENTUM_WEIGHTS.get(
                        "ALTERNATING_PATTERN_STREAK", 0
                    )
                    signals_m += 1
                    reasons_m.append("Alternância(z)")
        if wheel_jump_analysis.get("most_common_jumps"):
            last_num = historico[-1]
            wheel_pos = {num: i for i, num in enumerate(ROULETTE_WHEEL_ORDER)}
            if last_num in wheel_pos:
                last_pos = wheel_pos[last_num]
                for jump_data in wheel_jump_analysis["most_common_jumps"][:2]:
                    jump_dist = jump_data["jump"]
                    target_pos_1 = (last_pos + jump_dist) % 37
                    target_pos_2 = (last_pos - jump_dist + 37) % 37
                    target_num_1 = ROULETTE_WHEEL_ORDER[target_pos_1]
                    target_num_2 = ROULETTE_WHEEL_ORDER[target_pos_2]
                    if (
                        target_num_1 in stats["numbers"]
                        or target_num_2 in stats["numbers"]
                    ):
                        score_m += (
                            1.0 - (jump_data["jump"] / 18.0)
                        ) * MOMENTUM_WEIGHTS.get("COMMON_WHEEL_JUMP", 0)
                        signals_m += 1
                        reasons_m.append("Salto Roda")
                        break
        if twin_analysis.get("is_current_a_twin"):
            post_twin_data = twin_analysis.get("post_twin_analysis", {})
            if post_twin_data.get("top_followers"):
                follower_nums = {f["num"] for f in post_twin_data["top_followers"]}
                if not follower_nums.isdisjoint(stats["numbers"]):
                    score_m += MOMENTUM_WEIGHTS.get("POST_TWIN_FOLLOWER", 0)
                    signals_m += 1
                    reasons_m.append("Pós-Gêmeo")
                    
        if session_analysis and session_analysis.get('hottest_sector'):
            hottest = session_analysis['hottest_sector']
            # Se a aposta atual for no setor mais quente da sessão e o desvio for significativo
            if name == hottest.get('name') and hottest.get('deviation', 0) > 0.25:
                # O score é proporcional ao desvio (quão mais quente que o esperado)
                session_score = hottest['deviation'] * MOMENTUM_WEIGHTS.get("SESSION_SECTOR_HEAT", 0)
                score_m += min(session_score, 3.0) # Limita o bônus máximo
                signals_m += 1
                reasons_m.append("Setor Quente(S)")
        # Lógica para Finais (Momentum)
        if final_analysis and final_analysis.get("current_streak", 0) > 1:
            final_numbers = set(final_analysis.get('numbers', []))
            # Se o grupo da aposta atual contém números do final que está em sequência
            if not final_numbers.isdisjoint(group_numbers):
                score_m += (final_analysis["current_streak"] / 4.0) * MOMENTUM_WEIGHTS.get("FINAL_STREAK", 0)
                signals_m += 1
                reasons_m.append("Final(S)")

        # Lógica para Espelhos (Momentum)
        if mirror_analysis:
            for pair_name, pair_numbers in MIRROR_PAIRS.items():
                if last_number in pair_numbers:
                    # Encontra o parceiro do espelho
                    mirror_partner = pair_numbers[0] if last_number == pair_numbers[1] else pair_numbers[1]
                    # Se o parceiro do espelho está no grupo da aposta atual
                    if mirror_partner in group_numbers:
                        score_m += MOMENTUM_WEIGHTS.get("MIRROR_FOLLOWER", 0)
                        signals_m += 1
                        reasons_m.append("Espelho(S)")
                        break # Para de checar outros pares de espelho
        

        if score_c > 0:
            score_c *= _calculate_conviction_bonus(signals_c)
        if score_m > 0:
            score_m *= _calculate_conviction_bonus(signals_m)
        

        final_score, profile, raw_reasons = 0, None, []
        if score_c > 0.5 and score_c >= score_m:
            final_score, profile, raw_reasons = score_c, "Contrarian", reasons_c
        elif score_m > 0.5 and score_m > score_c:
            final_score, profile, raw_reasons = score_m, "Momentum", reasons_m

        if profile:
            # Calcula a volatilidade aqui
            volatility_info = _calculate_bet_volatility(list(group_numbers))
            volatility_index = volatility_info.get('index', 1.0)
            
            # Cria um score ajustado ao risco
            risk_adjusted_score = final_score / (volatility_index ** 0.75) # O expoente 0.75 suaviza o impacto

            opportunities.append({
                "name": name,
                "score": final_score,
                "risk_adjusted_score": risk_adjusted_score, # Novo score para ordenação
                "profile": profile,
                "raw_reasons": list(set(raw_reasons)),
                "numbers": list(group_numbers),
                "volatility_description": volatility_info.get('description'),
            })

    if not opportunities:
        return []
    max_score = max(op["score"] for op in opportunities) if opportunities else 1.0
    for op in opportunities:
        op["normalized_score"] = (
            min((op["score"] / max_score) * 10, 10) if max_score > 0 else 0
        )
    return sorted(opportunities, key=lambda x: x["risk_adjusted_score"], reverse=True)


def run_backtest(
    full_historico: List[int], strategy_weights: Dict, ia_strategists: Dict
) -> Dict:
    if len(full_historico) < 50:
        return {"error": "Histórico insuficiente para backtest (mínimo 50 giros)."}

    balance = 0
    hits, misses = 0, 0
    max_drawdown = 0
    current_win_streak, max_win_streak = 0, 0
    current_loss_streak, max_loss_streak = 0, 0
    play_history = []

    for i in range(37, len(full_historico)):
        past_history = full_historico[:i]
        next_number = full_historico[i]

        # A análise agora já calcula o stake junto com o veredito
        analysis = analyze_historical_slice(
            past_history, ia_strategists, active_strategy_weights=strategy_weights
        )
        verdict = (
            analysis.get("intelligence", {})
            .get("analyst_bulletin", {})
            .get("final_verdict")
        )

        # << NOVO >> A condição de aposta agora verifica se o stake é maior que zero
        if verdict and verdict.get("stake_units", 0) > 0:
            bet_numbers = verdict["numbers"]
            bet_name = verdict["name"]
            stake = verdict["stake_units"]  # Usa o stake dinâmico

            # Custo e Payout são multiplicados pelo stake
            cost = len(bet_numbers) * stake
            balance -= cost

            total_plays = (
                hits + misses + 1
            )  # Atualiza aqui para evitar divisão por zero

            if next_number in bet_numbers:
                payout = 36 * stake  # Payout também é proporcional ao stake
                balance += payout
                hits += 1
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
                outcome = "HIT"
            else:
                misses += 1
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
                outcome = "MISS"

            # Armazena o histórico com o valor da aposta
            play_history.append(
                {
                    "spin": i,
                    "bet_on": f"{bet_name} ({stake} un.)",
                    "outcome": outcome,
                    "result_number": next_number,
                    "balance": balance,
                }
            )

        max_drawdown = min(max_drawdown, balance)

    total_plays = hits + misses
    hit_rate = (hits / total_plays * 100) if total_plays > 0 else 0

    # << MELHORIA >> Prepara dados para o gráfico do frontend
    balance_over_time = [p['balance'] for p in play_history]

    # Formata o balanço para exibição na lista de histórico
    for p in play_history:
        p['balance'] = f"{p['balance']:.1f}"

    return {
        "final_balance": f"{balance:.1f} unidades",
        "total_plays": total_plays,
        "hits": hits,
        "misses": misses,
        "hit_rate": f"{hit_rate:.2f}%",
        "max_drawdown": f"{max_drawdown:.1f} unidades",
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "play_history": play_history[-20:], # Retorna só os últimos 20 para a lista detalhada
        "balance_over_time": balance_over_time, # Retorna o histórico completo de balanço para o gráfico
    }


def _analyze_followers(historico: List[int]) -> Dict:
    if len(historico) < 2:
        return {}
    last_number = historico[-1]
    followers = [
        historico[i + 1]
        for i in range(len(historico) - 1)
        if historico[i] == last_number
    ]
    if not followers:
        return {"last_number": last_number, "top_followers": []}
    top_followers = Counter(followers).most_common(5)
    return {
        "last_number": last_number,
        "top_followers": [{"num": num, "count": count} for num, count in top_followers],
    }


def _analyze_dynamic_neighbors(historico: List[int], numbers_stats: Dict) -> Dict:
    if len(historico) < 1:
        return {}
    last_number = historico[-1]
    try:
        idx = ROULETTE_WHEEL_ORDER.index(last_number)
    except ValueError:
        return {}
    neighbors = [ROULETTE_WHEEL_ORDER[(idx - i) % 37] for i in [2, 1]] + [
        ROULETTE_WHEEL_ORDER[(idx + i) % 37] for i in [1, 2]
    ]
    return {
        "last_number": last_number,
        "neighbors": neighbors,
        "total_hits": sum(numbers_stats[n]["hits"] for n in neighbors),
        "gap": calculate_gap(historico, neighbors),
    }


def _analyze_oppositional_pressure(historico: List[int]) -> Dict:
    """
    Identifica grupos que estão sob 'pressão' porque seu oposto está em uma longa sequência.
    Ex: Se 'Vermelho' saiu 5x seguidas, 'Preto' está sob alta pressão.
    """
    if not historico or len(historico) < 3:
        return {}

    pressure_alerts = {}
    
    # Mapeia grupos e seus opostos
    opposites_map = {
        "Vermelho": "Preto", "Preto": "Vermelho",
        "Par": "Ímpar", "Ímpar": "Par",
        "Baixo (1-18)": "Alto (19-36)", "Alto (19-36)": "Baixo (1-18)",
        "1ª Dúzia": ["2ª Dúzia", "3ª Dúzia"],
        "2ª Dúzia": ["1ª Dúzia", "3ª Dúzia"],
        "3ª Dúzia": ["1ª Dúzia", "2ª Dúzia"],
        "1ª Coluna": ["2ª Coluna", "3ª Coluna"],
        "2ª Coluna": ["1ª Coluna", "3ª Coluna"],
        "3ª Coluna": ["1ª Coluna", "2ª Coluna"],
    }

    # Analisa streaks para criar pressão
    streaks = _analyze_current_streaks(historico)
    
    for streak_type, streak_info in streaks.items():
        streaking_group = streak_info['name']
        streak_count = streak_info['count']

        if streak_count > 2 and streaking_group in opposites_map:
            # Identifica o(s) grupo(s) oposto(s)
            opposite_groups = opposites_map[streaking_group]
            if not isinstance(opposite_groups, list):
                opposite_groups = [opposite_groups]
            
            for opposite_group in opposite_groups:
                 pressure_alerts[opposite_group] = {"streak": streak_count}

    return pressure_alerts


def _analyze_post_zero_behavior(historico: List[int]) -> Dict:
    if 0 not in historico or len(historico) < 2:
        return {}
    return {}


def _analyze_current_streaks(historico: List[int]) -> Dict:
    if not historico:
        return {}
    streaks = {}

    def get_prop_val(n, prop_type):
        if prop_type == "color":
            return get_color(n)
        props = get_properties(n)
        return props.get(prop_type, "Zero")

    for prop_type in ["color", "dozen"]:
        last_prop = get_prop_val(historico[-1], prop_type)
        if last_prop != "Zero":
            count = 0
            for num in reversed(historico):
                if get_prop_val(num, prop_type) == last_prop:
                    count += 1
                else:
                    break
            if count > 1:
                streaks[prop_type.capitalize()] = {"name": last_prop, "count": count}
    return streaks


def _analyze_individual_numbers(
    historico: List[int], total_spins: int, frequencies: Counter
) -> Dict:
    numbers_stats = {}
    theoretical_freq = 1 / 37
    for num in ALL_NUMBERS:
        hits = frequencies.get(num, 0)
        observed_freq = hits / total_spins if total_spins > 0 else 0
        deviation = (
            (observed_freq - theoretical_freq) / theoretical_freq
            if theoretical_freq > 0
            else 0
        )
        numbers_stats[num] = {
            "hits": hits,
            "percentage": f"{observed_freq*100:.2f}",
            "gap": calculate_gap(historico, [num]),
            "color": get_color(num),
            "deviation": deviation,
        }
    return numbers_stats


def _analyze_groups(historico: List[int], total_spins: int) -> Dict:
    group_stats = defaultdict(dict)
    for group_name, sub_groups in GROUPS.items():
        for name, data in sub_groups.items():
            numbers, prob = data["numbers"], data["prob"]
            hits = sum(1 for num in historico if num in numbers)
            observed_freq = hits / total_spins if total_spins > 0 else 0
            deviation = (observed_freq - prob) / prob if prob > 0 else 0
            group_stats[group_name][name] = {
                "hits": hits,
                "percentage": f"{observed_freq*100:.1f}",
                "gap": calculate_gap(historico, numbers),
                "deviation": deviation,
                "prob": prob,
                "numbers": numbers,
            }
    return dict(group_stats)


def _analyze_sequences(historico: List[int]) -> Dict:
    return {}


def _calculate_coverage_cycle(historico: List[int]) -> Dict:
    return {}


def _analyze_mirror_numbers(historico: List[int], total_spins: int) -> Dict:
    return {}


def _generate_alerts(group_stats: Dict) -> List[str]:
    alerts = []
    return alerts


# --- FUNÇÃO DE ANÁLISE PRINCIPAL ---

def analyze_historical_slice(historico: List[int], ia_strategists: Dict, active_strategy_weights: Dict = None) -> Dict:
    """
    Analisa uma fatia de histórico do zero. Usado para backtesting, treinamento de ML
    e simulações históricas onde um estado incremental não pode ser usado.
    """
    total_spins = len(historico)
    if total_spins < 1:
        return {"total_spins": 0, "status": "Aguardando dados...", "historico_recente": []}

    # Esta é a lógica de recálculo original, necessária para análises históricas
    frequencies = Counter(historico)
    numbers_stats = _analyze_individual_numbers(historico, total_spins, frequencies)
    group_stats = _analyze_groups(historico, total_spins)
    
    # Chama todas as outras funções de análise de padrões
    streak_analysis = _analyze_current_streaks(historico)
    follower_analysis = _analyze_followers(historico)
    pressure_analysis = _analyze_oppositional_pressure(historico)
    dynamic_neighbor_analysis = _analyze_dynamic_neighbors(historico, numbers_stats)
    post_zero_analysis = _analyze_post_zero_behavior(historico)
    sequences = _analyze_sequences(historico)
    coverage_cycle = _calculate_coverage_cycle(historico)
    mirror_analysis = _analyze_mirror_numbers(historico, total_spins)
    terrain_analysis = _analyze_terrain(historico, numbers_stats)
    final_analysis = _analyze_finals(historico)
    twin_analysis = _analyze_twins(historico)
    alternating_patterns_analysis = _analyze_alternating_patterns(historico)
    wheel_jump_analysis = _analyze_wheel_jumps(historico)
    
    sorted_by_deviation = sorted(numbers_stats.items(), key=lambda item: item[1]["deviation"])
    hot_numbers = [{"num": item[0], "dev": item[1]["deviation"]} for item in sorted_by_deviation[-5:]]
    cold_numbers = [{"num": item[0], "dev": item[1]["deviation"]} for item in sorted_by_deviation[:5]]
    game_rhythm = _analyze_game_rhythm(historico)

    opportunity_scores = _calculate_opportunity_scores(
        group_stats=group_stats, numbers_stats=numbers_stats, hot_numbers=hot_numbers,
        cold_numbers=cold_numbers, streak_analysis=streak_analysis, follower_analysis=follower_analysis,
        terrain_analysis=terrain_analysis, pressure_analysis=pressure_analysis, twin_analysis=twin_analysis,
        alternating_patterns=alternating_patterns_analysis, wheel_jump_analysis=wheel_jump_analysis,
        final_analysis=final_analysis, mirror_analysis=mirror_analysis, historico=historico,
        total_spins=total_spins, weights=(active_strategy_weights or {})
    )

    analyst_bulletin = _generate_analyst_bulletin(game_rhythm, opportunity_scores, weights=(active_strategy_weights or {}))
    
    # Retorna a mesma estrutura de dados que a função principal
    return {
        "total_spins": total_spins,
        "last_number": historico[-1] if historico else None,
        "historico_recente": historico[-25:],
        "group_stats": group_stats,
        "numbers_stats": numbers_stats,
        "intelligence": {
            "opportunity_scores": opportunity_scores,
            "analyst_bulletin": analyst_bulletin,
        },
    }

def analyze_statistics(
    state: Any,  # Recebe o objeto AnalysisState
    ia_strategists: Dict,
    active_strategy_weights: Dict = None,
    ml_prediction: Dict = None,
    ml_numbers_prediction: Dict = None,
    session_analysis: Dict = None,
) -> Dict[str, Any]:
    
    # 1. << CORRIGIDO >> Pega os dados direto do objeto de estado. Sem recálculos!
    total_spins = state.total_spins
    historico = state.historico  # É uma deque
    numbers_stats = state.numbers_stats
    group_stats = state.group_stats

    if total_spins < 1:
        return {"total_spins": 0, "status": "Aguardando dados...", "historico_recente": []}

    weights_to_use = active_strategy_weights or ia_strategists.get("O Equilibrista (Híbrido)", {}).get("weights", {})

    # 2. << MANTIDO >> As análises de padrões rodam, pois são leves e focam no histórico recente
    streak_analysis = _analyze_current_streaks(historico)
    follower_analysis = _analyze_followers(historico)
    pressure_analysis = _analyze_oppositional_pressure(historico)
    dynamic_neighbor_analysis = _analyze_dynamic_neighbors(historico, numbers_stats)
    post_zero_analysis = _analyze_post_zero_behavior(historico)
    sequences = _analyze_sequences(historico)
    coverage_cycle = _calculate_coverage_cycle(historico)
    mirror_analysis = _analyze_mirror_numbers(historico, total_spins)
    terrain_analysis = _analyze_terrain(historico, numbers_stats)
    final_analysis = _analyze_finals(historico)
    twin_analysis = _analyze_twins(historico)
    alternating_patterns_analysis = _analyze_alternating_patterns(historico)
    wheel_jump_analysis = _analyze_wheel_jumps(historico)

    # 3. << MANTIDO >> A lógica de alto nível continua a mesma
    sorted_by_deviation = sorted(numbers_stats.items(), key=lambda item: item[1]["deviation"])
    hot_numbers = [{"num": item[0], "dev": item[1]["deviation"]} for item in sorted_by_deviation[-5:]]
    cold_numbers = [{"num": item[0], "dev": item[1]["deviation"]} for item in sorted_by_deviation[:5]]

    game_rhythm = _analyze_game_rhythm(historico)

    opportunity_scores = _calculate_opportunity_scores(
        group_stats=group_stats,
        numbers_stats=numbers_stats,
        hot_numbers=hot_numbers,
        cold_numbers=cold_numbers,
        streak_analysis=streak_analysis,
        follower_analysis=follower_analysis,
        terrain_analysis=terrain_analysis,
        pressure_analysis=pressure_analysis,
        twin_analysis=twin_analysis,
        alternating_patterns=alternating_patterns_analysis,
        wheel_jump_analysis=wheel_jump_analysis,
        final_analysis=final_analysis,
        mirror_analysis=mirror_analysis,
        historico=historico,
        total_spins=total_spins,
        weights=weights_to_use,
        ml_prediction=ml_prediction,
        ml_numbers_prediction=ml_numbers_prediction,
        session_analysis=session_analysis,
    )

    analyst_bulletin = _generate_analyst_bulletin(
        game_rhythm,
        opportunity_scores,
        weights=weights_to_use
    )

    # 4. << MANTIDO >> Retorna a estrutura de dados completa para o frontend
    return {
        "total_spins": total_spins,
        "last_number": historico[-1] if historico else None,
        "historico_recente": list(historico)[-25:],
        "group_stats": group_stats,
        "wheel_order": ROULETTE_WHEEL_ORDER,
        "colors": ROULETTE_COLORS,
        "numbers_stats": numbers_stats,
        "hot_numbers": hot_numbers,
        "cold_numbers": cold_numbers,
        "intelligence": {
            "game_rhythm": game_rhythm,
            "opportunity_scores": opportunity_scores,
            "analyst_bulletin": analyst_bulletin,
            "follower_analysis": follower_analysis,
            "streak_analysis": streak_analysis,
            "pressure_analysis": pressure_analysis,
            "dynamic_neighbor_analysis": dynamic_neighbor_analysis,
            "post_zero_analysis": post_zero_analysis,
            "sequences": sequences,
            "coverage_cycle": coverage_cycle,
            "mirror_analysis": mirror_analysis,
            "final_analysis": final_analysis,
            "twin_analysis": twin_analysis,
            "alternating_patterns": alternating_patterns_analysis,
            "wheel_jump_analysis": wheel_jump_analysis,
            "ml_group_prediction": ml_prediction, # Adicionado para consistência
            "ml_numbers_prediction": ml_numbers_prediction, # Adicionado para consistência
        },
        "alerts": _generate_alerts(group_stats),
    }