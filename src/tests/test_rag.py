import pytest
import giskard
import pandas as pd
import asyncio
from ollama import Client
from src.engine.rag_engine import MultimodalRAG
import os

from qdrant_client import AsyncQdrantClient, models
import json

from giskard.llm.client.litellm import LiteLLMClient


qdrant_client=AsyncQdrantClient(url="http://localhost:6333", timeout=60)
async def get_context_from_qdrant(question: str, collection_name_child: str):
    """Fetch the top relevant text chunk for the judge to use"""

    search_result = await qdrant_client.query(
        collection_name=collection_name_child,
        query_text=question,
        limit=5
    )

    if search_result:

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
        return False


async def recreate_qdrant(client: AsyncQdrantClient, child_coll: str, parent_coll: str, child_json: str, parent_json: str):
    """Reconstructs the Child and Parent test databases"""

    with open(child_json, "r", encoding="utf-8") as f:
        child_data = json.load(f)
    with open(parent_json, "r", encoding="utf-8") as f:
        parent_data = json.load(f)

    # --- RECREATE PARENT COLLECTION (Small-to-Big Target) ---
    if await client.collection_exists(parent_coll):
        await client.delete_collection(parent_coll)

    # Parents are retrieved by ID, so we use an empty vectors_config
    await client.create_collection(
        collection_name=parent_coll,
        vectors_config={}
    )

    # Populate Parents
    parent_points = [models.PointStruct(**p) for p in parent_data]
    await client.upsert(collection_name=parent_coll, points=parent_points)

    # --- RECREATE CHILD COLLECTION (Vector Search Target) ---
    if await client.collection_exists(child_coll):
        await client.delete_collection(child_coll)

    await client.create_collection(
        collection_name=child_coll,
        vectors_config={
            "text": models.VectorParams(size=384, distance=models.Distance.COSINE),
            "image": models.VectorParams(size=512, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "text-sparse": models.SparseVectorParams(index=models.SparseIndexParams())
        }
    )

    # Populate Children
    child_points = [models.PointStruct(**p) for p in child_data]
    await client.upsert(collection_name=child_coll, points=child_points)

    print(f"✅ Success: Recreated {parent_coll} ({len(parent_points)} pts) and {child_coll} ({len(child_points)} pts)")


@pytest.mark.asyncio
async def test_batch_rag_evaluation():
    child_json = "src/tests/data/test_child.json"
    parent_json = "src/tests/data/test_parents.json"

    child_coll = "the_batch_mini"
    parent_coll = "the_batch_parents_mini"

    await recreate_qdrant(qdrant_client, child_coll, parent_coll,child_json,parent_json)






    rag_engine = MultimodalRAG()
    rag_engine.CHILD_COLL = child_coll
    rag_engine.PARENT_COLL = parent_coll
    assert rag_engine.groq_key is not None, "GR_TOKEN environment variable is missing!"


    test_samples = [
        {"question": "How are AI monopolies affecting the market?", "category": "Business"},
        {"question": "What is the latest in transformer world models?", "category": "ML Research"}
    ]
    test_df = pd.DataFrame(test_samples)

    def model_predict(df: pd.DataFrame):
        answers = []
        for q in df["question"]:

            res = run_async(rag_engine.run_hybrid_rag(q))
            if res is None:
                answers.append("No answer generated")
            else:
                answers.append(res.get("answer", str(res)) if isinstance(res, dict) else str(res))
        return answers

    os.environ["GROQ_API_KEY"] = os.getenv("GR_TOKEN", "")
    ollama_llm = LiteLLMClient(
    model="groq/openai/gpt-oss-safeguard-20b"
)
    giskard_model = giskard.Model(
        model=model_predict,
        model_type="text_generation",
        name="RAG_Batch_Evaluator",
        description="A RAG engine that retrieves context from Qdrant and generates answers about AI research, business, culture news.",
        feature_names=["question", "category"],
        llm_client=ollama_llm
    )

    giskard_dataset = giskard.Dataset(df=test_df, name="The_Batch_Multimodal_Sample")


    scan_results = await asyncio.to_thread(giskard.scan, giskard_model, giskard_dataset)
    scan_results.to_html("giskard_report.html")


    for i, row in test_df.iterrows():
        # Get actual answer from RAG
        res = await rag_engine.run_hybrid_rag(row['question'])
        if res is None:
            actual_answer = "Error: Engine returned None"
        actual_answer = res.get("answer", str(res))

        context_for_judge = await get_context_from_qdrant(row['question'], collection_name_child=child_coll)

        is_relevant = qwen_judge_relevance(row['question'], actual_answer, context_for_judge)
        assert is_relevant, f"❌ Judge failed relevance for: {row['question']}"

    assert not scan_results.has_issues(severity="high"), "❌ Giskard found high-severity vulnerabilities"
