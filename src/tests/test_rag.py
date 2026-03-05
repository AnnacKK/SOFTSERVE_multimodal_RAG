import pytest
import giskard
import pandas as pd
import asyncio
from ollama import Client
from src.engine.rag_engine import MultimodalRAG

from qdrant_client import AsyncQdrantClient



qdrant_client=AsyncQdrantClient(url="http://localhost:6333", timeout=60)
async def get_context_from_qdrant(question: str, collection_name: str = "the_batch_children"):
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
    # Initialize engine
    rag_engine = MultimodalRAG()

    # 1. Define Sample Dataset
    test_samples = [
        {"question": "How are AI monopolies affecting the market?", "category": "Business"},
        {"question": "What is the latest in transformer world models?", "category": "ML Research"},
        {"question": "Summarize the latest productivity impact from GenAI.", "category": "Weekly Issues"},
        {"question": "How do H100 GPUs compare to older hardware?", "category": "Hardware"}
    ]
    test_df = pd.DataFrame(test_samples)

    # 2. Define the Prediction Function (Bridging Async RAG to Sync Giskard)
    def model_predict(df: pd.DataFrame):
        answers = []
        for q in df["question"]:
            # Wrap the async query call
            res = run_async(rag_engine.run_hybrid_rag(q))
            # Ensure we return a string for Giskard's text_generation type
            answers.append(res.get("answer", str(res)) if isinstance(res, dict) else str(res))
        return answers

    # 3. Wrap for Giskard
    giskard_model = giskard.Model(
        model=model_predict,
        model_type="text_generation",
        name="RAG_Batch_Evaluator",
        feature_names=["question", "category"],
        description="RAG engine retrieving from Qdrant and generating with Qwen"
    )

    giskard_dataset = giskard.Dataset(
        df=test_df,
        name="The_Batch_Multimodal_Sample",
        target=None,
        column_types={"question": "text", "category": "category"}
    )

    # 4. Run Automated Scan (Vulnerability Guard)
    scan_results = giskard.scan(giskard_model, giskard_dataset)
    scan_results.to_html("giskard_report.html")

    # 5. Custom Judicial Check (Relevance Guard)
    # We iterate through the scan results or the dataset to apply our Qwen Judge
    # For CI/CD, we'll manually check the first few for the 'Judicial' pass

    assert rag_engine.groq_key is not None, "GR_TOKEN environment variable is missing!"

    for i, row in test_df.iterrows():
        # 1. Get the actual answer from your RAG engine
        res = run_async(rag_engine.run_hybrid_rag(row['question']))
        actual_answer = res.get("answer", str(res))

        # 2. ?? NEW: Fetch the context specifically for the judge
        context_for_judge =await get_context_from_qdrant(row['question'])

        # 3. ?? NEW: Pass it to the judge
        is_relevant = qwen_judge_relevance(row['question'], actual_answer, context_for_judge)

        assert is_relevant, f"? Judge failed relevance for: {row['question']}"
    # Final CI/CD Gate
    assert not scan_results.has_issues(severity="high"), "? Giskard found high-severity vulnerabilities"
