import pytest
import asyncio
from src.engine.rag_engine import MultimodalRAG


@pytest.mark.asyncio
async def test_rag_functional_smoke_test():
    # 1. Initialize the engine
    rag = MultimodalRAG()

    # 2. Run a simple query
    question = "How are AI monopolies affecting the market?"
    response = await rag.run_hybrid_rag(question)

    # 3. Validations
    print(f"\n--- Sanity Check Results ---")
    print(f"Question: {question}")

    # Check that we didn't get None (Fixing your previous error)
    assert response is not None, "❌ Engine returned None! Check retrieval thresholds."

    # Check for the expected structure
    assert isinstance(response, dict), "❌ Response should be a dictionary."
    assert "answer" in response, "❌ Response missing 'answer' key."

    # Check that the answer actually contains text
    answer = response["answer"]
    print(f"Answer: {answer[:100]}...")

    assert len(answer) > 20, "❌ Answer is too short or empty."
    assert "Error" not in answer, f"❌ Engine returned an error string: {answer}"

    # 4. Cleanup
    await rag.close()
