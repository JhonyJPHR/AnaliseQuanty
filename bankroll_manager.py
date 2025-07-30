import logging

class BankrollManager:
    """
    Gerencia o estado de uma sessão de apostas, aplicando regras de stop-loss e stop-win.
    """
    def __init__(self):
        self.is_active = False
        self.initial_bankroll = 100.0
        self.current_bankroll = 100.0
        self.stop_loss_threshold = 0.0 # Valor em que a sessão para por perda
        self.stop_win_threshold = 0.0  # Valor em que a sessão para por ganho
        self.status_message = "Sessão não iniciada."
        self.session_profit = 0.0
        self.stop_loss_percent = 0.20 # 20%
        self.stop_win_percent = 0.40 # 40%

    def start_session(self, initial_amount: float, stop_loss: float, stop_win: float):
        """Inicia uma nova sessão de gerenciamento."""
        self.initial_bankroll = float(initial_amount)
        self.current_bankroll = float(initial_amount)
        self.stop_loss_percent = float(stop_loss)
        self.stop_win_percent = float(stop_win)
        
        # Calcula os valores absolutos para parada
        self.stop_loss_threshold = self.initial_bankroll * (1 - self.stop_loss_percent)
        self.stop_win_threshold = self.initial_bankroll * (1 + self.stop_win_percent)
        
        self.is_active = True
        self.status_message = "Sessão em andamento."
        self.session_profit = 0.0
        logging.info(f"Sessão de Banca iniciada. Inicial: ${self.initial_bankroll:.2f}, Stop-Loss em: ${self.stop_loss_threshold:.2f}, Stop-Win em: ${self.stop_win_threshold:.2f}")

    def stop_session(self):
        """Para a sessão manualmente."""
        self.is_active = False
        self.status_message = f"Sessão parada manualmente. Balanço final: ${self.current_bankroll:.2f}"
        logging.info("Sessão de Banca parada manualmente.")

    def update_balance(self, profit_or_loss: float):
        """Atualiza o balanço e verifica os limites."""
        if not self.is_active:
            return

        self.current_bankroll += profit_or_loss
        self.session_profit = self.current_bankroll - self.initial_bankroll
        self._check_limits()

    def _check_limits(self):
        """Verifica se os limites de stop-loss ou stop-win foram atingidos."""
        if self.current_bankroll <= self.stop_loss_threshold:
            self.is_active = False
            self.status_message = f"❌ LIMITE DE PERDA ATINGIDO. Balanço: ${self.current_bankroll:.2f}. Pausa recomendada."
            logging.warning(self.status_message)
        elif self.current_bankroll >= self.stop_win_threshold:
            self.is_active = False
            self.status_message = f"✅ META DE GANHOS ATINGIDA! Balanço: ${self.current_bankroll:.2f}. Sessão concluída."
            logging.warning(self.status_message)

    def get_status(self) -> dict:
        """Retorna o estado atual do gerenciador em um dicionário."""
        return {
            "is_active": self.is_active,
            "initial_bankroll": self.initial_bankroll,
            "current_bankroll": self.current_bankroll,
            "session_profit": self.session_profit,
            "status_message": self.status_message,
            "stop_loss_percent": self.stop_loss_percent,
            "stop_win_percent": self.stop_win_percent
        }