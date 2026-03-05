import asyncio
import json
import os
from qdrant_client import AsyncQdrantClient


def serialize_qdrant_data(data):
    """Recursively converts Qdrant objects (Vectors, SparseVectors) to JSON-serializable types."""
    if isinstance(data, dict):
        return {k: serialize_qdrant_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_qdrant_data(i) for i in data]
    # Check for Qdrant's SparseVector or Vector objects
    if hasattr(data, "as_py"):
        return data.as_py()
    # Check for Pydantic models (common in newer qdrant-client)
    if hasattr(data, "model_dump"):
        return data.model_dump()
    return data


async def distill_to_json():
    client = AsyncQdrantClient(url="http://127.0.0.1:6333", timeout=120)
    source_coll = "the_batch_children"
    out_path = "src/tests/data/test_points.json"

    try:
        print(f"📡 Connecting to Qdrant at 127.0.0.1...")
        res, _ = await client.scroll(
            collection_name=source_coll,
            limit=200,
            with_payload=True,
            with_vectors=True
        )

        if not res:
            print("❌ Source collection is empty.")
            return

        print(f"✅ Extracted {len(res)} points. Serializing for JSON...")

        # 🟢 FIX: Transform Records into serializable dictionaries
        points_data = []
        for r in res:
            points_data.append({
                "id": r.id,
                "payload": r.payload,
                "vector": serialize_qdrant_data(r.vector)  # Deep clean the vector objects
            })

        # 4. Save to file
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(points_data, f, indent=2)

        print(f"🚀 Success! Created {out_path} ({os.path.getsize(out_path) / 1024:.2f} KB)")

    except Exception as e:
        print(f"🚨 Critical Failure: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(distill_to_json())