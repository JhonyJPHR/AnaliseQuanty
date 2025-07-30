import datetime
from sqlalchemy import create_engine, Column, Integer, DateTime, String, Index, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
import config 

Base = declarative_base()
class Giro(Base):
    """
    Modelo SQLAlchemy que representa um único giro da roleta.
    """
    __tablename__ = 'giros'
    
    id = Column(Integer, primary_key=True)
    numero = Column(Integer, nullable=False)
    cor = Column(String(10)) # 'red', 'black', ou 'green'
    timestamp = Column(DateTime, default=datetime.datetime.now, index=True)
    session_id = Column(String(50), index=True, nullable=True) 

    __table_args__ = (Index('ix_giros_timestamp_desc', timestamp.desc()),)

    def __repr__(self):
        return f"<Giro(id={self.id}, numero={self.numero}, cor='{self.cor}', session='{self.session_id}')>" 

class AnaliseVeredito(Base):
    """
    Armazena a análise pós-morte de cada predição de alta confiança da IA.
    """
    __tablename__ = 'analises_veredito'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    predicted_numbers = Column(String(100)) 
    actual_number = Column(Integer)
    was_hit = Column(Boolean)
    winning_number_profile = Column(String(500)) 
    ai_conclusion = Column(String(1000))
    prediction_reasons = Column(String(500), nullable=True) #
    outcome_quality = Column(String(50), nullable=True, default='MISS', index=True)

    def __repr__(self):
        return f"<AnaliseVeredito(id={self.id}, hit={self.was_hit}, quality='{self.outcome_quality}')>"


# Configuração do banco de dados SQLite
DATABASE_URL = config.DATABASE_URL
engine = create_engine(DATABASE_URL) 
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    Cria todas as tabelas no banco de dados se elas ainda não existirem.
    """
    Base.metadata.create_all(bind=engine)