import asyncio
import json
import os
from qdrant_client import AsyncQdrantClient


def serialize_qdrant_data(data):
    """Recursively converts Qdrant objects to JSON-serializable types."""
    if isinstance(data, dict):
        return {k: serialize_qdrant_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_qdrant_data(i) for i in data]
    if hasattr(data, "as_py"):
        return data.as_py()
    if hasattr(data, "model_dump"):
        return data.model_dump()
    return data


async def distill_to_json():
    client = AsyncQdrantClient(url="http://127.0.0.1:6333", timeout=120)

    child_source = "the_batch_children"
    parent_source = "the_batch_parents"
    child_out = "src/tests/data/test_child.json"
    parent_out = "src/tests/data/test_parents.json"

    try:
        print(f"📡 Connecting to Qdrant...")

        # 1. Distill Children
        print(f"🔍 Extracting children from {child_source}...")
        c_res, _ = await client.scroll(collection_name=child_source, limit=300, with_payload=True, with_vectors=True)

        child_points = []
        parent_ids_to_fetch = set()

        for r in c_res:
            child_points.append({"id": r.id, "payload": r.payload, "vector": serialize_qdrant_data(r.vector)})
            p_id = r.payload.get("parent_id")
            if p_id:
                parent_ids_to_fetch.add(p_id)

        # 2. Distill Referenced Parents
        print(f"🔍 Fetching {len(parent_ids_to_fetch)} referenced parents from {parent_source}...")
        parent_points = []
        if parent_ids_to_fetch:
            p_res = await client.retrieve(
                collection_name=parent_source,
                ids=list(parent_ids_to_fetch),
                with_payload=True,
                with_vectors=False  # Parents usually don't need vectors for retrieval-only
            )
            for r in p_res:
                parent_points.append({"id": r.id, "payload": r.payload, "vector": {}})

        # 3. Save Files
        os.makedirs("src/tests/data", exist_ok=True)

        with open(child_out, "w", encoding="utf-8") as f:
            json.dump(child_points, f, indent=2)

        with open(parent_out, "w", encoding="utf-8") as f:
            json.dump(parent_points, f, indent=2)

        print(f"🚀 Success!")
        print(f"   📦 Children: {len(child_points)} points -> {child_out}")
        print(f"   📦 Parents: {len(parent_points)} points -> {parent_out}")

    except Exception as e:
        print(f"🚨 Critical Failure: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(distill_to_json())