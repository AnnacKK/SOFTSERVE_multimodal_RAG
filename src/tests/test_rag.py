import pytest
import giskard
import pandas as pd
import asyncio
from ollama import Client
from src.engine.rag_engine import MultimodalRAG
import os
import datetime
from qdrant_client import AsyncQdrantClient, models
import json
from flashrank import Ranker, RerankRequest


ranker = Ranker()


qdrant_client=AsyncQdrantClient(url="http://localhost:6333", timeout=120)


async def get_context_from_qdrant(rag_engine: MultimodalRAG, question: str, collection_name_child: str):
    """Fetch the top relevant text chunk using the same Hybrid logic as the RAG engine"""

    # 1. Prepare all 3 vectors (Dense, Vision, Sparse)
    t_vec = await asyncio.to_thread(rag_engine.text_model.encode, question, normalize_embeddings=True)
    i_vec = await asyncio.to_thread(rag_engine.vision_model.encode, question, normalize_embeddings=True)

    sparse_res_list = await asyncio.to_thread(list, rag_engine.sparse_model.embed([question]))
    sparse_res = sparse_res_list[0]
    sparse_vec = models.SparseVector(
        indices=sparse_res.indices.tolist(),
        values=sparse_res.values.tolist(),
    )


    search_result = await qdrant_client.query_points(
        collection_name=collection_name_child,
        prefetch=[
            models.Prefetch(query=t_vec.tolist(), using="text", limit=15),
            models.Prefetch(query=i_vec.tolist(), using="image", limit=15),
            models.Prefetch(query=sparse_vec, using="text-sparse", limit=15),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=15
    )

    if search_result.points:
        passages = []
        for i, res in enumerate(search_result.points):
            txt = res.payload.get("chunk_text") or res.payload.get("full_text")
            if txt:
                passages.append({"id": i, "text": txt})

        if not passages:
            return "No text content found in retrieval results."

        # 3. Rerank to find the absolute best context for the Judge
        rerank_request = RerankRequest(query=question, passages=passages)
        results = ranker.rerank(rerank_request)

        return "\n---\n".join([r['text'] for r in results[:5]])

    return "No context found."

# def run_async(coro):
#     try:
#         loop = asyncio.get_event_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#     return loop.run_until_complete(coro)


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


    await client.create_collection(
        collection_name=parent_coll,
        vectors_config={"none": models.VectorParams(size=1, distance=models.Distance.COSINE)}
    )

    parent_points = []
    for p in parent_data:

        payload_data = p.get("payload", p)

        point = models.PointStruct(
            id=p["id"],
            vector={"none": [0.0]},
            payload={
                "full_text": payload_data.get("full_text", ""),
                "image_b64": payload_data.get("image_b64", ""),
                "headline": payload_data.get("headline", "No Headline"),
                "url": payload_data.get("url", ""),
                "type": payload_data.get("type", "article")
            }
        )
        parent_points.append(point)
    await client.upsert(collection_name=parent_coll, points=parent_points)


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

    child_points = []
    for p in child_data:
        point = models.PointStruct(
            id=p["id"],
            vector=p["vector"],
            payload=p["payload"]
        )
        child_points.append(point)
    await client.upsert(collection_name=child_coll, points=child_points)
    await asyncio.sleep(5)

    print(f"✅ Success: Recreated {parent_coll} ({len(parent_points)} pts) and {child_coll} ({len(child_points)} pts)")


@pytest.mark.asyncio
async def test_batch_rag_evaluation():
    test_loop = asyncio.get_running_loop()
    test_sem = asyncio.Semaphore(3)

    child_json = "src/tests/data/test_child.json"
    parent_json = "src/tests/data/test_parents.json"

    child_coll = "the_batch_mini"
    parent_coll = "the_batch_parents_mini"

    await recreate_qdrant(qdrant_client, child_coll, parent_coll,child_json,parent_json)

    rag_engine = MultimodalRAG()
    rag_engine.RERANK_LIMIT = 0.0
    rag_engine.THRESHOLD = 0.0
    rag_engine.client=qdrant_client
    rag_engine.CHILD_COLL = child_coll
    rag_engine.PARENT_COLL = parent_coll
    if not rag_engine.groq_key:
        rag_engine.groq_key = os.getenv("GR_TOKEN")

    assert rag_engine.groq_key is not None, "GR_TOKEN environment variable is missing!"


    test_samples = [
        {"question": "How are AI monopolies affecting the market?", "category": "Business"},
        {"question": "What is the latest in transformer world models?", "category": "ML Research"}
    ]
    test_df = pd.DataFrame(test_samples)
    test_df["ground_truth"] = "N/A"


    def model_predict(df: pd.DataFrame):
        async def wrapped_predict(q):
            async with test_sem:
                try:
                    rag_engine.chat_history.clear()

                    res = await asyncio.wait_for(rag_engine.run_hybrid_rag(q), timeout=500.0)

                    if res is None:
                        return "Error: Engine returned None"

                    answer = res.get("answer", "")
                    if "I couldn't find any relevant snippets" in answer:
                        return "I am sorry, but I do not have enough information to answer that question."
                    return answer

                except Exception as e:
                    return f"Error: {str(e)}"

        async def run_batch():
            results = []
            for q in df["question"]:
                results.append(await wrapped_predict(q))
                await asyncio.sleep(0.2)
            return results

        future = asyncio.run_coroutine_threadsafe(run_batch(), test_loop)
        return future.result()

    gemini_token = os.getenv("GEMINI_API_KEY")
    if gemini_token:
        os.environ["GEMINI_API_KEY"] = gemini_token
        os.environ["GOOGLE_API_KEY"] = gemini_token

    api_base = "http://localhost:11434"

    # llm = LiteLLMClient(model="ollama/qwen2.5:1.5b")
    giskard.llm.set_llm_model("ollama/qwen2.5:1.5b", disable_structured_output=True, api_base=api_base)

    giskard_model=giskard.Model(
        model=model_predict,
        model_type="text_generation",
        name="RAG_Batch_Evaluator",
        description="A RAG engine that retrieves context from Qdrant and generates answers about AI research, business, culture news.",
        feature_names=["question", "category"]
    )

    giskard_dataset=giskard.Dataset(df=test_df, name="The_Batch_Multimodal_Sample")


    scan_results = await asyncio.to_thread(giskard.scan, giskard_model, giskard_dataset,only=["hallucination","faithfulness"],params={
        "hallucination": {"samples_limit": 1},
        "sycophancy": {"samples_limit": 1},
        "faithfulness": {"samples_limit": 1} #only for weak system, increase number for pull request
    })
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"giskard_report_{timestamp}.html"
    try:
        scan_results.to_html(report_name)
        print(f"✅ Giskard report successfully saved to {report_name}")
    except Exception as e:
        print(f"❌ Failed to save Giskard report: {e}")



    print("-----------STARTING SELF JUDGE---------------")

    # async def run_judge(row):
    #     async with test_sem:
    #         res = await rag_engine.run_hybrid_rag(row['question'])
    #         if res is None:
    #             actual_answer = "Error: Engine returned None"
    #         else:
    #             actual_answer = res.get("answer", str(res))
    #
    #         context_for_judge = await get_context_from_qdrant(rag_engine, row['question'],
    #                                                           child_coll)
    #
    #         is_relevant = qwen_judge_relevance(row['question'], actual_answer, context_for_judge)
    #         return {
    #             "timestamp": datetime.datetime.now().isoformat(),
    #             "question": row['question'],
    #             "context_len": len(context_for_judge),
    #             "context_snippet": context_for_judge[:200] + "...",  # Truncate for readability
    #             "answer": actual_answer,
    #             "judge_decision": "PASS" if is_relevant else "FAIL"
    #         }
    #
    #
    # judge_tasks = [run_judge(row) for _, row in test_df.iterrows()]
    # judge_results = await asyncio.gather(*judge_tasks)

    # judge_report_name = f"judge_results_{timestamp}.json"
    # with open(judge_report_name, "w", encoding="utf-8") as f:
    #     json.dump(judge_results, f, indent=4)

    # print(f"✅ Judge results saved to {judge_report_name}")

    # if os.getenv("GITHUB_STEP_SUMMARY"):
    #     with open(os.getenv("GITHUB_STEP_SUMMARY"), "a") as summary:
    #         summary.write("### ⚖️ LLM Judge Results\n")
    #         summary.write("| Question | Decision | Answer Preview |\n")
    #         summary.write("| :--- | :--- | :--- |\n")
    #         for r in judge_results:
    #             icon = "✅" if r["judge_decision"] == "PASS" else "❌"
    #             summary.write(f"| {r['question']} | {icon} {r['judge_decision']} | {r['answer'][:50]}... |\n")
    #
    #         summary.write("\n### 🛡️ Giskard Scan\n")
    #         summary.write(f"- Issues Found: {len(scan_results.issues)}\n")
    #         summary.write(
    #             f"- [View Full HTML Report]({os.getenv('GITHUB_SERVER_URL')}/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')})\n")
    #
    #
    # judge_fails = [r for r in judge_results if r["judge_decision"] == "FAIL"]
    # assert not judge_fails, f"❌ Judge failed {len(judge_fails)} cases. Check {judge_report_name}"
    major_issues = [issue for issue in scan_results.issues if issue.level == "major"]
    assert not major_issues, f"❌ Giskard found {len(major_issues)} MAJOR issues. Check report."
    await qdrant_client.close()




