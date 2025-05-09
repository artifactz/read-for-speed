import sqlite3
from pathlib import Path


DB_PATH = "ml/font/pdf_data.db"

__db_file_exists = Path(DB_PATH).exists()
_connection = sqlite3.connect(DB_PATH)
_cursor = _connection.cursor()


def create_table():
    _cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdf_files (
            path TEXT PRIMARY KEY,
            primary_font TEXT,
            primary_font_raw TEXT,
            qa_status TEXT
        )
    """)
    _connection.commit()

if not __db_file_exists:
    create_table()


def add_pdf_file(path: str, primary_font: str=None, primary_font_raw: str=None, qa_status: str=None):
    _cursor.execute("""
        INSERT INTO pdf_files (path, primary_font, primary_font_raw, qa_status)
        VALUES (?, ?, ?, ?)
    """, (path, primary_font, primary_font_raw, qa_status))
    _connection.commit()
    return _cursor.lastrowid


def get_pdf_file(path: str):
    _cursor.execute("""
        SELECT * FROM pdf_files WHERE path = ?
    """, (path,))
    values = _cursor.fetchone()
    keys = [desc[0] for desc in _cursor.description]
    if values is None:
        return None
    return dict(zip(keys, values))


def get_pdf_files():
    _cursor.execute("SELECT * FROM pdf_files")
    pdf_files = _cursor.fetchall()
    keys = [desc[0] for desc in _cursor.description]
    return [dict(zip(keys, values)) for values in pdf_files]


def update_pdf_file(path: str, **kw_args):
    set_clause = ", ".join([f"{k} = ?" for k in kw_args.keys()])
    values = list(kw_args.values())
    values.append(path)
    _cursor.execute(f"""
        UPDATE pdf_files SET {set_clause} WHERE path = ?
    """, values)
    _connection.commit()
    return _cursor.rowcount > 0


if __name__ == "__main__":
    print(get_pdf_files()[-1])
