import sqlite3
from typing import Dict, List, Any

DB_FILE = "fabrica_poc.db"

def setup_database():
    """Cria o banco de dados e as tabelas e retorna a conexão."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Define as tabelas usando chaves estrangeiras (links)
    cursor.execute("""DROP TABLE IF EXISTS Pecas""")
    cursor.execute("""DROP TABLE IF EXISTS Partes""")
    cursor.execute("""DROP TABLE IF EXISTS Maquinas""")
    cursor.execute("""DROP TABLE IF EXISTS Categorias""")
    
    # Nível 1
    cursor.execute("""
        CREATE TABLE Categorias (
            id INTEGER PRIMARY KEY,
            nome TEXT NOT NULL
        )
    """)
    
    # Nível 2
    cursor.execute("""
        CREATE TABLE Maquinas (
            id INTEGER PRIMARY KEY,
            categoria_id INTEGER,
            nome TEXT NOT NULL,
            n_serie TEXT UNIQUE,
            FOREIGN KEY(categoria_id) REFERENCES Categorias(id)
        )
    """)
    
    # Nível 3
    cursor.execute("""
        CREATE TABLE Partes (
            id INTEGER PRIMARY KEY,
            maquina_id INTEGER,
            nome TEXT NOT NULL,
            FOREIGN KEY(maquina_id) REFERENCES Maquinas(id)
        )
    """)
    
    # Nível 4
    cursor.execute("""
        CREATE TABLE Pecas (
            id INTEGER PRIMARY KEY,
            parte_id INTEGER,
            codigo TEXT UNIQUE NOT NULL,
            descricao TEXT,
            FOREIGN KEY(parte_id) REFERENCES Partes(id)
        )
    """)
    
    return conn, cursor

def populate_database(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    """Insere os dados da PoC na estrutura do SQLite."""

    # 1. Categoria
    cursor.execute("INSERT INTO Categorias (nome) VALUES (?)", ("Moedores",))
    cat_id = cursor.lastrowid

    # 2. Máquina
    cursor.execute("INSERT INTO Maquinas (categoria_id, nome, n_serie) VALUES (?, ?, ?)", 
                   (cat_id, "Moedor de Carne CAF 8 Inox", "378988"))
    maq_id = cursor.lastrowid

    # 3. Partes
    nomes_das_partes = [
        "MOTOR, CÁRTER E ENGRENAGENS", 
        "CAPA", 
        "ELÉTRICA", 
        "CONJUNTO DE MOAGEM"
    ]
    partes_ids: Dict[str, int] = {}
    
    for nome in nomes_das_partes:
        cursor.execute("INSERT INTO Partes (maquina_id, nome) VALUES (?, ?)", (maq_id, nome))
        partes_ids[nome] = cursor.lastrowid
    
    parte_conjunto_moagem_id = partes_ids["CONJUNTO DE MOAGEM"]

    # 4. Peças
    pecas_data = [
        ("91010", "VOLANTE 8 / 9"),
        ("82150", "DISCO INOX CAF 8 FURO 5,0 MM (V.24)"),
        ("86110", "CRUZETA INOX CAF 8 (V.24)"),
        ("81200", "CARACOL MONTADO 8"),
        ("81100", "BOCAL CONJUNTO 8"),
    ]
    for codigo, descricao in pecas_data:
        cursor.execute("INSERT INTO Pecas (parte_id, codigo, descricao) VALUES (?, ?, ?)", 
                       (parte_conjunto_moagem_id, codigo, descricao))

    conn.commit()
    print(f"✅ População SQLite concluída no arquivo: {DB_FILE}")
    print(f"   ID da Máquina para teste: {maq_id}")

if __name__ == "__main__":
    conn, cursor = setup_database()
    populate_database(conn, cursor)
    conn.close()