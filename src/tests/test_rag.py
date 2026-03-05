import pytest
import giskard
import pandas as pd
import asyncio
from ollama import Client
from src.engine.rag_engine import MultimodalRAG

from qdrant_client import AsyncQdrantClient, models
import json
import os


qdrant_client=AsyncQdrantClient(url="http://localhost:6333", timeout=60)
async def get_context_from_qdrant(question: str, collection_name: str = "the_batch_mini"):
    """Fetch the top relevant text chunk for the judge to use"""

    search_result = await qdrant_client.query(
        collection_name=collection_name,
        query_text=question,
        limit=5
    )
    # Extract the text payload from the top point
    if search_result:
        # Extract and join all 5 text payloads with a separator
        merged_text = "\n---\n".join([
            res.payload.get("chunk_text", "") for res in search_result
        ])
        return merged_text
    return "No context found."

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def qwen_judge_relevance(question, answer, context):
    """The 'Oracle' judge using local Qwen"""
    prompt = f"""
    You are an impartial judge. Evaluate if the ANSWER directly addresses the user'S QUESTION using the provided CONTEXT.
    QUESTION: {question}
    CONTEXT: {context}
    ANSWER: {answer}
    Rules:
    - If the answer is off-topic or ignores context, say 'FAIL'.
    - If the answer is helpful and relevant, say 'PASS'.
    Result (PASS/FAIL only):"""

    ollama_client = Client(host='http://localhost:11434')

    try:
        response = ollama_client.generate(model="qwen2.5:1.5b", prompt=prompt)
        return "PASS" in response['response'].upper()
    except Exception:
        return False  # Fail safe if Ollama is down

@pytest.mark.asyncio
async def test_batch_rag_evaluation():
    # --- STEP 0: BOOTSTRAP QDRANT FROM JSON ---
    json_path = "src/tests/data/test_points.json"
    coll_name = "the_batch_mini"  # Ensure this matches what get_context_from_qdrant uses

    # 1. Load the distilled data
    with open(json_path, "r", encoding="utf-8") as f:
        points_data = json.load(f)

    # 2. Recreate the schema (Dense + Sparse)
    if await qdrant_client.collection_exists(coll_name):
        await qdrant_client.delete_collection(coll_name)

    await qdrant_client.create_collection(
        collection_name=coll_name,
        vectors_config={
            "text": models.VectorParams(size=384, distance=models.Distance.COSINE),
            "image": models.VectorParams(size=512, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "text-sparse": models.SparseVectorParams(index=models.SparseIndexParams())
        }
    )

    # 3. Upsert the JSON points into Qdrant
    points = [models.PointStruct(**p) for p in points_data]
    await qdrant_client.upsert(collection_name=coll_name, points=points)
    print(f"✅ Bootstrapped {len(points)} points from JSON into {coll_name}")

    # --- STEP 1: INITIALIZE ENGINE ---
    rag_engine = MultimodalRAG()
    assert rag_engine.groq_key is not None, "GR_TOKEN environment variable is missing!"

    # 1. Define Sample Dataset
    test_samples = [
        {"question": "How are AI monopolies affecting the market?", "category": "Business"},
        {"question": "What is the latest in transformer world models?", "category": "ML Research"}
    ]
    test_df = pd.DataFrame(test_samples)

    # 2. Define the Prediction Function (Bridging Async RAG to Sync Giskard)
    def model_predict(df: pd.DataFrame):
        answers = []
        for q in df["question"]:
            # Note: run_async is used here because Giskard calls this synchronously
            res = run_async(rag_engine.run_hybrid_rag(q))
            answers.append(res.get("answer", str(res)) if isinstance(res, dict) else str(res))
        return answers

    # 3. Wrap for Giskard
    giskard_model = giskard.Model(
        model=model_predict,
        model_type="text_generation",
        name="RAG_Batch_Evaluator",
        feature_names=["question", "category"]
    )

    giskard_dataset = giskard.Dataset(df=test_df, name="The_Batch_Multimodal_Sample")

    # 4. Run Automated Scan (Using to_thread for sync Giskard)
    scan_results = await asyncio.to_thread(giskard.scan, giskard_model, giskard_dataset)
    scan_results.to_html("giskard_report.html")

    # 5. Judicial Check (Using native await for async calls)
    for i, row in test_df.iterrows():
        # Get actual answer from RAG
        res = await rag_engine.run_hybrid_rag(row['question'])
        actual_answer = res.get("answer", str(res))

        # Fetch context from our newly created JSON collection
        context_for_judge = await get_context_from_qdrant(row['question'], collection_name=coll_name)

        is_relevant = qwen_judge_relevance(row['question'], actual_answer, context_for_judge)
        assert is_relevant, f"❌ Judge failed relevance for: {row['question']}"

    assert not scan_results.has_issues(severity="high"), "❌ Giskard found high-severity vulnerabilities"
