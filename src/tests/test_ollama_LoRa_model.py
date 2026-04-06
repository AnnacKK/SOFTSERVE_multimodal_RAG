import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import time
from src.config import config

OLLAMA_BASE_URL = "http://localhost:11434"



async def test_fine_tuned_model():
    print("--- Testing Model: ---")

    try:
        llm =  ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=config.VARIATIONS_LLAMA_MODEL_NAME,
            temperature=0.6)

        # 1. Basic Connection Test
        start_time = time.time()
        question = "latest news about AWS model"
        print(f"Sending Question: {question}")

        # Using a simple HumanMessage to test LangChain compatibility
        response = await llm.ainvoke([HumanMessage(content=question)])

        end_time = time.time()

        print("\n--- Model Response ---")
        print(response.content)
        print("----------------------")
        print(f"Response Time: {end_time - start_time:.2f} seconds")

        # 2. Check for Fine-Tuning "Style"
        if len(response.content) > 5:
            print("\n✅ SUCCESS: Model is responding.")
        else:
            print("\n⚠️ WARNING: Model returned an empty or very short response.")

    except Exception as e:
        print("\n❌ FAILED: Could not connect to Ollama or model not found.")
        print(f"Error: {e}")
        print("\nDEBUG TIPS:")
        print("1. Run 'ollama list' in terminal to see if '' exists.")
        print("2. Ensure Ollama is running in the background.")


if __name__ == "__main__":
    try:
        asyncio.run(test_fine_tuned_model())
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise
    time.sleep(0.1)