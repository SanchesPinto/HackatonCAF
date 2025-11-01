import sqlite3
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any

# --- CONFIGURAÇÃO DO BANCO DE DADOS (SQLite) ---
DB_FILE = "fabrica_poc.db"
app = FastAPI(title="API de Cotação de Peças - PoC (SQLite)")

def get_db_connection():
    """Cria e retorna uma conexão com o banco de dados SQLite."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Isso permite acessar colunas por nome
    return conn

# --- FUNÇÕES DE REQUISIÇÃO (DO CÓDIGO ANTERIOR) ---

def get_data(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Função genérica para executar consultas e retornar resultados como lista de dicionários."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    
    # Converte os objetos Row para dicionários
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results

@app.get("/categorias", summary="Lista todas as categorias.")
def listar_categorias():
    """Retorna a lista de todas as Categorias."""
    query = "SELECT id, nome FROM Categorias"
    # O ID no SQLite é um inteiro, que é facilmente serializável em JSON
    return get_data(query)

@app.get("/maquinas/{categoria_id}", summary="Lista máquinas por categoria.")
def listar_maquinas(categoria_id: int):
    """Retorna Máquinas de uma Categoria específica."""
    query = "SELECT id, nome FROM Maquinas WHERE categoria_id = ?"
    return get_data(query, (categoria_id,))

@app.get("/partes/{maquina_id}", summary="Lista partes por máquina.")
def listar_partes(maquina_id: int):
    """Retorna Partes de uma Máquina específica."""
    query = "SELECT id, nome FROM Partes WHERE maquina_id = ?"
    return get_data(query, (maquina_id,))

@app.get("/pecas/{parte_id}", summary="Lista peças por parte.")
def listar_pecas(parte_id: int):
    """Retorna Peças de uma Parte específica (Códigos e Descrições)."""
    query = "SELECT codigo, descricao FROM Pecas WHERE parte_id = ?"
    return get_data(query, (parte_id,))

