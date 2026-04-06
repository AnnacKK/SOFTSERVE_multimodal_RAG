import asyncio
import math
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.engine.rag_engine import MultimodalRAG
from src.monitoring_DB.metrics_DB import init_db, log_to_db
from src.metrics.metrics import Evaluator
from src.config import config
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from contextlib import asynccontextmanager


print("START DB")
init_db()
engine = MultimodalRAG()
templates = Jinja2Templates(directory="templates")

gpu_semaphore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gpu_semaphore

    gpu_semaphore = asyncio.Semaphore(1)
    print("🚦 GPU Semaphore initialized on the active event loop.")

    # # Start heartbeat
    # heartbeat_task = asyncio.create_task(keep_warm_heartbeat())

    yield

    # heartbeat_task.cancel()
    await engine.close()
app = FastAPI(title="Oracle RAG WebSocket API", lifespan=lifespan)

# async def keep_warm_heartbeat():
#     """
#     Sends a tiny periodic request to Google Colab to keep the
#     11B Vision model weights loaded in the VRAM.
#     """
#     print("💓 Heartbeat Service Started")
#     while True:
#         try:
#             await engine.llm.ainvoke("keep-warm-ping")
#             print("💓 Heartbeat: Remote Model is Warm.")
#         except Exception as e:
#             print(f"💔 Heartbeat Failed: {e}. Colab might be offline.")
#
#         # Ping every 3 minutes (Ollama unloads models after 5 mins of silence)
#         await asyncio.sleep(180)
# --- 🛰️ HELPER: CLEAN RAGAS SCORES ---
def clean_ragas_scores(raw_scores: dict) -> dict:
    defaults = {
        'faithfulness': 0.0, 'answer_relevancy': 0.0, 'context_utilization': 0.0, 'bleu': 0.0, 'rouge_l': 0.0, 'factcc_consistency': 0.0,
        'ner_coverage': 0.0, 'ner_hallucination': 0.0, 'ner_density': 0.0,
        'harm_score': 0.0, 'harm_category': 'none', 'is_corrected': 0
    }

    cleaned = defaults.copy()
    for key, value in raw_scores.items():
        if key in cleaned:
            val = value.item() if hasattr(value, 'item') else value
            try:
                val = float(val) if key != 'harm_category' else str(val)
                cleaned[key] = 0.0 if isinstance(val, float) and (math.isnan(val) or math.isinf(val)) else val
            except:
                pass
    return cleaned

rllm = ChatOllama(
    base_url=config.OLLAMA_BASE_URL,
    model=config.VARIATIONS_LLAMA_MODEL_NAME,
    #headers={"ngrok-skip-browser-warning": "true"},
    temperature=0,
    format="json",
    num_ctx=4096,
    timeout=400,
keep_alive=0
)
embeddings_standard = HuggingFaceEmbeddings(model_name=config.TEXT_MODEL_NAME)

async def audit_and_push_correction(question, initial_result, websocket):
    async with gpu_semaphore:
        """Audits the answer and pushes a correction if faith_score is low."""
        print("model:",rllm )
        print("⚖️ Background Audit: Starting deep evaluation...")
        try:

            evaluator = Evaluator(embeddings=embeddings_standard, llm=rllm)
            if isinstance(initial_result.get("context_text"), str):
                clean_contexts = [initial_result.get("context_text")]
            elif isinstance(initial_result.get("context_text"), list):
                clean_contexts = [str(c) for c in initial_result.get("context_text")]
            else:
                clean_contexts = ["No context available"]

            # Inside your audit_and_push_correction
            final_contexts = [
                    str(c).strip().lstrip('.').strip()
                    for c in initial_result.get("context_text") if len(str(c)) > 20
                ]

            print("----------CONTEXT--------:",final_contexts)

            # 2. Run check
            eval_report = await evaluator.check_response(
                question,
                initial_result["answer"],
                final_contexts
            )


            if isinstance(eval_report, dict):
                scores = eval_report
            else:
                scores = eval_report.to_pandas().to_dict('records')[0] if hasattr(eval_report, 'to_pandas') else {}
            clean_scores = clean_ragas_scores(scores)

            faith_score = float(clean_scores.get('faithfulness', 1.0))

            #  HALLUCINATION CHECK
            if faith_score < 0.7:
                print(f"⚠️ Low Faithfulness ({faith_score}). Refining...")

                # Make sure this method exists in your MultimodalRAG class
                refined_answer = await engine.refine_answer(
                    question,
                    initial_result["answer"],
                    final_contexts,
                    initial_result["gallery"]

                )


                # PUSH the correction back to the client
                await websocket.send_json({
                    "type": "correction",
                    "text": refined_answer,
                    "sources": initial_result.get("sources"),
                    "note": "✨ Note: I've refined this answer for better accuracy."
                })

                clean_scores['is_corrected'] = 1
                log_to_db(question, refined_answer, clean_scores)
                print(f"🏁 Background Audit Finished after refining. Scores: {clean_scores}")

            else:
                clean_scores['is_corrected'] = 0
                log_to_db(question, initial_result["answer"], clean_scores)
                print(f"🏁 Background Audit Finished. Scores: {clean_scores}")

                # Update frontend status
                await websocket.send_json({
                    "type": "status",
                    "status": "verified",
                    "scores": clean_scores
                })


        except Exception as e:

            print(f"🚨 Audit Failed: {e}")
            await websocket.send_json({

                "type": "error",

                "message": "⚠️ Sorry, an error occurred during the quality audit. Showing raw response."

            })

@app.get("/", response_class=HTMLResponse)
async def serve_site(request: Request):
    return templates.TemplateResponse("UI_template.html", {"request": request})



@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            user_q = data.get("query", "").strip()
            mode = data.get("mode", "Hybrid")

            if user_q.lower() in ["hi", "hello", "hey"]:
                await websocket.send_json({"type": "answer", "text": "Hello! Ask me anything."})
                continue

            if not user_q: continue

            async with gpu_semaphore:
                result = await engine.run_hybrid_rag(user_q, mode=mode)
            if result is None:
                await websocket.send_json({"type": "error", "text": "No relevant context found."})
                return

            if not result or result.get("answer") in ["no result", "no_context",
                                                       "⚠️ System overload. Could not generate text.", "⚠️"] or "⚠️" in result.get("answer"):
                 print("⚠️ Generation failed or empty. Skipping Audit to save VRAM/SSL.")
                 await websocket.send_json({
                     "type": "answer",
                     "text": result.get("answer") or "System is currently overloaded. Please try a shorter question.",
                     "status": "error"
                 })
                 continue

            # Check if the score is too low to even bother generating
            if  result.get("confidence_score", 0) < 0.2:
                print(f"🛑 Gatekeeper blocked: {user_q[:30]}...")
                await websocket.send_json({
                    "type": "answer",
                    "answer": "I cannot answer this. It seems outside my current knowledge base.",
                    "status": "blocked"
                })
                continue

            await websocket.send_json({
                "type": "metadata",
                "headline": result.get("headline") or "Search Result",
                "confidence": result.get("confidence_score", 0),
                #"image_description": result.get("image_description"),
                "sources": result.get("sources")
            })

            await websocket.send_json({
                "type": "answer",
                "answer": result["answer"],
                "headline": result.get("headline") or "Search Result",
                "confidence": result.get("confidence_score", 0),
                "gallery": result.get("gallery"),
                #"image_description": result.get("image_description"),
                "sources": result.get("sources")
            })



            await asyncio.sleep(0.1)
            asyncio.create_task(
                audit_and_push_correction(user_q, result, websocket)
            )

    except WebSocketDisconnect:
        print("🔌 Client disconnected")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)