import math
import os
import sqlite3
import psycopg2
from datetime import datetime

# Connection config
DATABASE_URL = os.getenv("DATABASE_URL")  # Provided by K8s YAML
current_dir = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB_PATH = os.path.join(current_dir, "rag_metrics.db")


def get_connection():
    """Returns a connection and the correct placeholder character."""
    if DATABASE_URL:
        # Postgres mode (Kubernetes)
        conn = psycopg2.connect(DATABASE_URL)
        return conn, "%s"
    else:
        # SQLite mode (Local)
        conn = sqlite3.connect(SQLITE_DB_PATH)
        return conn, "?"


def init_db() -> None:
    conn, q = get_connection()
    cursor = conn.cursor()

    id_type = "SERIAL PRIMARY KEY" if DATABASE_URL else "INTEGER PRIMARY KEY AUTOINCREMENT"

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS evaluation_logs (
            id {id_type},
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            faithfulness REAL,
            answer_relevancy REAL,
            context_utilization REAL,
            is_corrected INTEGER DEFAULT 0,
            bleu REAL,
            rouge_l REAL,
            factcc_consistency REAL,
            ner_coverage REAL,
            ner_hallucination REAL,
            ner_density REAL,
            harm_score REAL,
            harm_category TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_to_db(question, answer, scores) -> None:
    def clean_score(val):
        try:
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                return 0.0
            return f_val
        except:
            return 0.0

    conn, q = get_connection()
    cursor = conn.cursor()

    query = f"""
        INSERT INTO evaluation_logs (
            timestamp, question, answer, faithfulness, answer_relevancy, 
            context_utilization, is_corrected, bleu, rouge_l,
            factcc_consistency, ner_coverage, ner_hallucination, 
            ner_density, harm_score, harm_category
        )
        VALUES ({q}, {q}, {q}, {q}, {q}, {q}, {q}, {q}, {q}, {q}, {q}, {q}, {q}, {q}, {q})
    """

    values = (
        datetime.now().isoformat(),
        question,
        answer,
        clean_score(scores.get("faithfulness")),
        clean_score(scores.get("answer_relevancy")),
        clean_score(scores.get("context_utilization")),
        int(scores.get("is_corrected", 0)),
        clean_score(scores.get("bleu")),
        clean_score(scores.get("rouge_l")),
        clean_score(scores.get("factcc_consistency")),
        clean_score(scores.get("ner_coverage")),
        clean_score(scores.get("ner_hallucination")),
        clean_score(scores.get("ner_density")),
        clean_score(scores.get("harm_score")),
        scores.get("harm_category", "none"),
    )

    cursor.execute(query, values)
    conn.commit()
    conn.close()