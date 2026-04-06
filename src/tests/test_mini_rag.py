import pytest
import asyncio
import json
import os
from qdrant_client import AsyncQdrantClient, models
from src.engine.rag_engine import MultimodalRAG


@pytest.mark.asyncio
async def test_recreate_and_retrieve():
    # 1. Setup Client
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = AsyncQdrantClient(url=url,timeout=120)

    child_coll = "test_diag_children"
    parent_coll = "test_diag_parents"

    # 2. Load Data from your JSON files
    with open("data/test_child.json", "r") as f:
        child_data = json.load(f)
    with open("data/test_parents.json", "r") as f:
        parent_data = json.load(f)

    # 3. Clean and Recreate Collections
    for coll in [child_coll, parent_coll]:
        if await client.collection_exists(coll):
            await client.delete_collection(coll)

    # Create Parent (Simple Metadata Store)
    await client.create_collection(
        collection_name=parent_coll,
        vectors_config={"none": models.VectorParams(size=1, distance=models.Distance.COSINE)}
    )

    # Create Child (The Vector Store)
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

    # 4. Populate with Explicit ID Handling
    # We use str(id) to ensure UUIDs from JSON don't get mangled
    await client.upsert(
        collection_name=parent_coll,
        points=[models.PointStruct(id=p["id"], vector={"none": [0.0]}, payload=p["payload"]) for p in parent_data]
    )

    await client.upsert(
        collection_name=child_coll,
        points=[models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in child_data]
    )

    # 5. Test the RAG Engine
    rag = MultimodalRAG()
    rag.CHILD_COLL = child_coll
    rag.PARENT_COLL = parent_coll

    # CRITICAL: Force the gates open so we can see ANY match
    rag.RERANK_LIMIT = 0.3
    rag.THRESHOLD = 0.3

    question = "What is the latest in transformer world models?"
    print(f"\n--- Testing Query: {question} ---")

    response = await rag.run_hybrid_rag(question)

    # 6. Assertions & Logging
    print(f"RESULT HEADLINE: {response.get('headline')}")
    print(f"RESULT ANSWER: {response.get('answer')[:100]}...")

    assert response["headline"] != "No Results", "❌ ERROR: RAG still returned No Results!"
    assert len(response["sources"]) > 0, "❌ ERROR: No sources found (Parent link broken)!"

    print("✅ SUCCESS: Data is flowing correctly from JSON to RAG!")
    await client.close()