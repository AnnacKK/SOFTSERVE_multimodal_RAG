import sqlite3
from datetime import datetime
import math
import os
# This finds the directory this script is in (src/monitoring_db/)
current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(current_dir, "rag_metrics.db")



def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Create a table to store production logs and Ragas scores
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    ''')
    conn.commit()
    conn.close()

def log_to_db(question, answer, scores):
    def clean_score(val):
        try:
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                return 0.0
            return f_val
        except:
            return 0.0

    faithfulness_score = clean_score(scores.get('faithfulness'))
    answer_relevancy_score = clean_score(scores.get('answer_relevancy'))
    context_utilization_score = clean_score(scores.get('context_utilization'))
    is_corrected = int(scores.get('is_corrected', 0))
    bleu=clean_score(scores.get('bleu'))
    rouge_l=clean_score(scores.get('rouge_l'))
    factcc_consistency=clean_score(scores.get('factcc_consistency'))
    ner_coverage=clean_score(scores.get('ner_coverage'))
    ner_hallucination=clean_score(scores.get('ner_hallucination'))
    ner_density=clean_score(scores.get('ner_density'))
    harm_score=clean_score(scores.get('harm_score'))
    harm_category=scores.get('harm_category', 'none')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO evaluation_logs (timestamp, question, answer, faithfulness, answer_relevancy,context_utilization,
        is_corrected, bleu, rouge_l, 
            factcc_consistency, 
            ner_coverage, ner_hallucination, ner_density,
            harm_score, harm_category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        question,
        answer,
        faithfulness_score,answer_relevancy_score,context_utilization_score,
        is_corrected,bleu, rouge_l,
            factcc_consistency,
            ner_coverage, ner_hallucination, ner_density,
            harm_score, harm_category
    ))
    conn.commit()
    conn.close()