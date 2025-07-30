# analysis_updater.py
import logging
from collections import deque, Counter
from typing import List, Dict, Any
from analisador import ROULETTE_COLORS, ALL_NUMBERS, GROUPS, get_color
import config

class AnalysisState:
    """
    Mantém o estado de todas as estatísticas para evitar recálculos completos.
    Esta classe é o núcleo da otimização de performance.
    """
    def __init__(self):
        self.total_spins = 0
        self.historico = deque(maxlen=config.HISTORICO_MAX_LEN) # Mantém um histórico recente de tamanho razoável
        self.numbers_stats: Dict[int, Dict[str, Any]] = {}
        self.group_stats: Dict[str, Dict[str, Any]] = {}
        self._initialize_structures()
        logging.info("Gerenciador de Estado de Análise inicializado.")

    def _initialize_structures(self):
        """Prepara as estruturas de dados com valores zerados."""
        for num in ALL_NUMBERS:
            self.numbers_stats[num] = {
                'hits': 0, 'gap': 0, 'deviation': 0.0, 'color': get_color(num)
            }
        
        for group_type, subgroups in GROUPS.items():
            self.group_stats[group_type] = {}
            for name, data in subgroups.items():
                self.group_stats[group_type][name] = {
                    'hits': 0, 'gap': 0, 'deviation': -1.0, # Começa negativo para indicar frio
                    'prob': data['prob'], 'numbers': data['numbers']
                }

    def initialize_from_history(self, historico_completo: List[int]):
        """
        Constrói o estado inicial a partir de um histórico completo.
        Esta é a operação "pesada" que roda apenas uma vez no início.
        """
        logging.info(f"Construindo estado inicial a partir de {len(historico_completo)} giros...")
        self._initialize_structures() # Reseta tudo antes de construir
        for numero in historico_completo:
            self.update_with_new_spin(numero, is_initial_build=True)
        logging.info("Estado inicial construído com sucesso.")

    def update_with_new_spin(self, new_number: int, is_initial_build: bool = False):
        """
        Atualiza todas as estatísticas de forma incremental com um novo giro.
        Esta é a operação "leve" que roda a cada novo número.
        """
        self.total_spins += 1
        self.historico.append(new_number)

        # 1. Atualiza estatísticas dos NÚMEROS INDIVIDUAIS
        for num, stats in self.numbers_stats.items():
            if num == new_number:
                stats['gap'] = 0
                stats['hits'] += 1
            else:
                stats['gap'] += 1
            
            # Recalcula o desvio (operação leve)
            observed_freq = stats['hits'] / self.total_spins if self.total_spins > 0 else 0
            theoretical_freq = 1 / 37
            stats['deviation'] = (observed_freq - theoretical_freq) / theoretical_freq if theoretical_freq > 0 else 0

        # 2. Atualiza estatísticas dos GRUPOS
        for group_type, subgroups in self.group_stats.items():
            for name, stats in subgroups.items():
                if new_number in stats['numbers']:
                    stats['gap'] = 0
                    stats['hits'] += 1
                else:
                    stats['gap'] += 1

                # Recalcula o desvio (operação leve)
                observed_freq = stats['hits'] / self.total_spins if self.total_spins > 0 else 0
                prob = stats['prob']
                stats['deviation'] = (observed_freq - prob) / prob if prob > 0 else 0
                
        if not is_initial_build:
            logging.info(f"Estado atualizado incrementalmente com o número {new_number}.")
            
    def revert_last_spin(self):
        """
        Reverte o estado removendo o último giro do histórico e recalculando
        as estatísticas com base no histórico restante.
        """
        if not self.historico:
            logging.warning("Tentativa de reverter, mas o histórico está vazio.")
            return

        last_number = self.historico.pop()
        logging.info(f"Revertendo o último giro ({last_number}) do estado em memória.")
        
        # A forma mais robusta de recalcular o estado é usar a função de inicialização
        # sobre o histórico agora menor. É rápido pois opera em memória.
        self.initialize_from_history(list(self.historico))
        
        logging.info("Estado revertido e recalculado com sucesso.")
