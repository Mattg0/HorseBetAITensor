import sqlite3

def create_connection(db_file):
    """Create a database connection to the SQLite database."""
    conn = sqlite3.connect(db_file)
    return conn

def initialize_db(conn):
    """Create the participants table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute('''
              CREATE TABLE IF NOT EXISTS Course (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    jour DATE,
    comp INTEGER,
    hippodrome TEXT,
    meteo TEXT,
    dist INTEGER,
    corde TEXT,
    natpis TEXT,
    pistegp TEXT,
    typec TEXT,
    temperature REAL,
    forceVent REAL,
    directionVent TEXT,
    nebulosite TEXT,
    participants TEXT
);
        ''')

    cursor.execute('''
          CREATE TABLE IF NOT EXISTS Courses_Test (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comp INTEGER,
        arriv TEXT,
        hippodrome TEXT,
        meteo TEXT,
        dist INTEGER,
        corde TEXT,
        natpis TEXT,
        pistegp TEXT,
        typec TEXT,
        temperature REAL,
        forceVent REAL,
        directionVent TEXT,
        nebulosite TEXT,
        participants TEXT
    );
            ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Cheval (
            idche INTEGER PRIMARY KEY,
            nom TEXT NOT NULL,
            sexe TEXT NOT NULL,
            age INTEGER NOT NULL,
            last_musique  TEXT
        )
        ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Jockey (
            idJockey INTEGER PRIMARY KEY,
            nom TEXT NOT NULL,
            last_musique  TEXT
            )
        ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Entraineur (
            idEntraineur INTEGER PRIMARY KEY,
            nom TEXT NOT NULL,
            last_musique  TEXT
            );
        ''')

    conn.commit()
    conn.close()