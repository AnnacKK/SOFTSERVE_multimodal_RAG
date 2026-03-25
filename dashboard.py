import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path

st.set_page_config(page_title="MM RAG Monitor 🛰️", layout="wide")

current_dir = Path(__file__).parent
DB_PATH = current_dir / "src" / "monitoring_DB" / "rag_metrics.db"


def load_data():
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(DB_PATH), check_same_thread=False) as conn:
            df = pd.read_sql_query("SELECT * FROM evaluation_logs", conn)
            return df
    except Exception as e:
        st.error(f"🔥 Database Error: {e}")
        return pd.DataFrame()


st.title("🛰️ The Batch: Oracle Performance Dashboard")

with st.sidebar:
    st.header("Controls")
    if st.button('🔄 Refresh Data'):
        st.cache_data.clear()
        st.rerun()
    st.info(f"Monitoring: `{DB_PATH.name}`")

df = load_data()

if not df.empty:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp', ascending=False)

    st.subheader("🎯 Core RAGAS Scores")
    c1, c2, c3, c4 = st.columns(4)


    def get_avg(col):
        return df[col].mean() if col in df.columns else 0.0


    c1.metric("Faithfulness", f"{get_avg('faithfulness'):.3f}")
    c2.metric("Relevancy", f"{get_avg('answer_relevancy'):.3f}")
    c3.metric("Utilization", f"{get_avg('context_utilization'):.3f}")

    total_corrected = df['is_corrected'].sum() if 'is_corrected' in df.columns else 0
    c4.metric("Corrections", int(total_corrected), delta_color="inverse")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🔍 NeuLab InterpretEval (NER Health)")
        # Calculate eRec and eOOV
        ner_cols = st.columns(2)
        ner_cols[0].metric("NER Coverage (eRec)", f"{get_avg('ner_coverage'):.3f}")
        # Hallucination is inverse (lower is better)
        hallu = get_avg('ner_hallucination')
        ner_cols[1].metric("Hallucination Rate (eOOV)", f"{hallu:.3f}", delta=f"{hallu:.2f}", delta_color="inverse")

        st.line_chart(df.set_index('timestamp')[['ner_coverage', 'ner_hallucination']])

    with col_right:
        st.subheader("🛡️ Safety & Consistency (Salesforce / MS)")
        saf_cols = st.columns(2)
        saf_cols[0].metric("FactCC Consistency", f"{get_avg('factcc_consistency'):.3f}")
        saf_cols[1].metric("Avg Harm Risk", f"{get_avg('harm_score'):.3f}", delta_color="inverse")

        st.area_chart(df.set_index('timestamp')[['factcc_consistency', 'harm_score']])

    st.divider()
    st.subheader("📝 Detailed Evaluation Logs")


    # Custom color coding for safety
    def highlight_hallucinations(s):
        return ['background-color: #450a0a' if v > 0.4 else '' for v in
                s] if s.name == 'ner_hallucination' else [''] * len(s)

    st.divider()
    st.subheader("⚖️ Metric Tradeoff Analysis")

    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.write("**Faithfulness vs. Relevancy (The RAG Frontier)**")
        st.scatter_chart(
            df,
            x='answer_relevancy',
            y='faithfulness',
            color='is_corrected',
            size='ner_density'
        )
        st.caption("Target: Top-Right Corner (High Relevancy AND High Faithfulness)")

    with t_col2:
        st.write("**Entity Recall (eRec) vs. Hallucination (eOOV)**")

        st.scatter_chart(
            df,
            x='answer_relevancy',
            y='faithfulness',
            color='harm_score',
            size='factcc_consistency'
        )
        st.caption("Target: Bottom-Right Corner (High Coverage AND Low Hallucination)")


    st.dataframe(
        df.style.apply(highlight_hallucinations, axis=0),
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Time", format="D MMM, HH:mm"),
            "ner_hallucination": st.column_config.ProgressColumn("Hallucination (eOOV)", min_value=0, max_value=1),
            "factcc_consistency": st.column_config.NumberColumn("FactCC", format="%.2f"),
            "harm_score": st.column_config.NumberColumn("Harm", format="%.2f"),
            "is_corrected": st.column_config.CheckboxColumn("Refined?"),
        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("⚠️ No data found. Ensure your `rag_metrics.db` is populated.")