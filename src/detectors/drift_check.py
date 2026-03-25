import time
import numpy as np
from qdrant_client import QdrantClient
from kubernetes import client, config
import sys


DRIFT_THRESHOLD = 0.7  # Adjust based on testing
BASELINE_SIZE = 1000  # Historical baseline
WINDOW_SIZE = 100  # Recent queries to check


def get_drift_score():
    q_client = QdrantClient(url="http://qdrant-db:6333")

    # 1. Get Baseline (Historical queries)
    baseline = q_client.scroll(collection_name="llm_cache", limit=BASELINE_SIZE, with_vectors=True)[0]
    # 2. Get Recent (Newest queries)
    recent = q_client.scroll(collection_name="llm_cache", limit=WINDOW_SIZE, with_vectors=True)[0]

    if len(recent) < WINDOW_SIZE: return 0.0

    b_vecs = np.array([hit.vector for hit in baseline])
    r_vecs = np.array([hit.vector for hit in recent])

    # Simple Drift: Distance between centroids
    b_centroid = np.mean(b_vecs, axis=0)
    r_centroid = np.mean(r_vecs, axis=0)

    return np.linalg.norm(b_centroid - r_centroid)


def trigger_job():
    config.load_incluster_config()
    batch_v1 = client.BatchV1Api()

    active_jobs = batch_v1.list_namespaced_job(namespace="default", label_selector="app=data-ingestion")
    if len(active_jobs.items) > 0:
        print("⏳ Ingestion already in progress. Skipping.")
        return

    job = client.V1Job(
        metadata=client.V1ObjectMeta(generate_name="ingest-drift-"),
        spec=client.V1JobSpec(
            ttl_seconds_after_finished=3600,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"app": "data-ingestion"}),
                spec=client.V1PodSpec(
                    containers=[client.V1Container(
                        name="processor",
                        image="multimodalrag-data_processing:latest",
                        imagePullPolicy="Never",
                        command=["python", "data_processing/mapping.py"],
                        env=[
                            {"name": "QDRANT_HOST", "value": "qdrant-db"},
                            {"name": "OLLAMA_BASE_URL", "value": "http://host.docker.internal:11434"}
                        ]
                    )],
                    restart_policy="OnFailure"
                )
            )
        )
    )
    batch_v1.create_namespaced_job(namespace="default", body=job)


def run_once():
    try:
        score = get_drift_score()
        print(f"📊 Daily Drift Score: {score:.4f}")

        if score > 0.7:  # Your threshold
            print("🚨 Threshold exceeded. Triggering ingestion...")
            trigger_job()
        else:
            print("✅ Data is stable. No action needed.")

    except Exception as e:
        print(f"❌ Drift check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_once()