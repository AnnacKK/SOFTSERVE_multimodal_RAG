import asyncio
import hashlib
import os

import httpx
import nest_asyncio
import numpy as np
from fastembed import SparseTextEmbedding
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import PointStruct
from sentence_transformers import CrossEncoder, SentenceTransformer
from torch import nn

from src.config import config
from src.prompts.prompt_template import prompts

nest_asyncio.apply()
import re

from groq import AsyncGroq
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import torch

# stabilize tunnel
stable_client = httpx.AsyncClient(
    verify=False,
    http2=False,
    timeout=httpx.Timeout(300.0, connect=60.0),
    limits=httpx.Limits(max_keepalive_connections=0, max_connections=10),
    headers={"Connection": "close", "ngrok-skip-browser-warning": "true"},
)

qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = 6333


class MultimodalRAG:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.qdrant_url = os.getenv("QDRANT_URL", getattr(config, "QDRANT_URL", "http://localhost:6333"))

        self.client = AsyncQdrantClient(url=self.qdrant_url, timeout=60)
        #self.client = AsyncQdrantClient(url=config.QDRANT_URL, timeout=60)
        self.text_model = SentenceTransformer(config.TEXT_MODEL_NAME, device=self.device)
        self.vision_model = SentenceTransformer(
            config.IMAGE_MODEL_NAME,
            device=self.device,
            model_kwargs={}
        )
        self.reranker = CrossEncoder(
            config.RERANKER_NAME,
            device=self.device,
            activation_fn=nn.Sigmoid(),
        )
        self.lock = asyncio.Semaphore(1)
        self.llm_lock = asyncio.Semaphore(2)
        self.CHILD_COLL = config.CHILD_COLL
        self.PARENT_COLL = config.PARENT_COLL
        self.THRESHOLD = 0.3
        self.RERANK_LIMIT = 0.6
        self.QDRANT_LIMIT=15
        self.CANDIDATES_LIMIT=4
        self.groq_key = getattr(config, 'GR_TOKEN', None) or os.getenv("GR_TOKEN")

        if not self.groq_key:
            raise ValueError("Missing GR_TOKEN. Set it in config.py or as an Environment Variable.")
        self.groq_client = AsyncGroq(api_key=self.groq_key)

        self.model_id = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.base_model_kwargs = {
            "temperature": 0.2,
            "max_tokens": 4098,
            "top_p": 0.9,
            "stream": True,
            "frequency_penalty": 0.5,
        }
        self.judge_model_kwargs = {
            "temperature": 0.0,
            "max_tokens": 4098,
            "top_p": 0.9,
            "stream": True,
            "frequency_penalty": 0.5,
        }

        # self.llm = ChatOllama(
        #     base_url=config.OlLAMA_URL_COLLAB,
        #     model="llama3.2-vision:11b",
        #     temperature=0.3,
        #     num_ctx=2048,
        #     repeat_penalty=1.3,
        #     top_p=0.9,num_thread=4,
        #     timeout=300,
        #             )
        self.groq_llm = ChatGroq(api_key=self.groq_key, model_name=self.model_id)
        self.sparse_model = SparseTextEmbedding(model_name=config.SPARSE_MODEL_NAME)
        self.chat_history = ChatMessageHistory()

        self.trimer = trim_messages(
            max_tokens=600,
            strategy="last",
            token_counter=self.groq_llm,
            start_on="human",
            include_system=True,
        )

        self.variations_llama = ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.VARIATIONS_LLAMA_MODEL_NAME,
            temperature=0.6,
        )
        self.base_prompt = prompts.base_prompt
        self.critique_chain = prompts.critique_chain
        self.query_expansion = (
            prompts.query_expansion | self.variations_llama | StrOutputParser()
        )

        self.semantic_cache = {}
        self.cache_client = self.client
        self.cache_collection = "llm_cache"
        # Check if cache collection exists, if not, create it
        self._cache_initialized = False

    async def close(self) -> None:
        """Properly shuts down async connections to avoid 'loop closed' errors."""
        """Properly shuts down all async connections to avoid 'loop closed' errors."""
        try:
            await self.client.close()
            await self.cache_client.close()

        except Exception:
            pass

    def reranker_scores_report(self, scores):
        if not scores:
            return {"best_score": 0.0, "median_scores": 0.0, "iqr": 0.0}
        max_scores = np.max(scores)
        median_scores = np.median(scores)
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        return {
            "max_scores": float(max_scores),
            "median_scores": float(median_scores),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
        }

    def create_message(self, prompt, images):
        if isinstance(images, str):
            images = [images]

            # Filter out empty or too-short strings
        valid_images = [img for img in images if img and len(img) > 100]

        content = [{"type": "text", "text": prompt}]

        # Append all valid images to the content list
        for img in valid_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                },
            )
        return [{"role": "user", "content": content}]

    def clean_newsletter_text(self, text: str) -> str:
        """Removes scraping noise and header/footer artifacts."""
        if "Dear friends," in text:
            text = text.rsplit("Dear friends,", maxsplit=1)[-1]
        if "Keep learning!" in text:
            text = text.split("Keep learning!")[0]

        noise_patterns = [
            r"Published\s+\w+\s+\d+,\s+\d+",
            r"Reading time\s+\d+\s+min read",
            r"Share\s+Loading.*Player\.\.\.",
            r"AudioNative Player",
            r"Elevenlabs Text to Speech",
            r"\(https?://\S+\)",
        ]
        cleaned = text
        for pattern in noise_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip()

    async def describe_image(self, image_b64: str, related_titles: list) -> str:
        """Generates a technical description using the Vision LLM."""
        if not image_b64 or len(image_b64) < 100:
            return "No visual data found in payload."

        titles_str = ", ".join(related_titles)

        prompt = f"""This image is from an AI newsletter.
        The current news topics are: {titles_str}.
        TASK: Provide a one-sentence technical description.
        Describe it in one technical sentence for a screen reader.
        Focus on diagrams, charts, or specific AI symbols present.
        RULES:
        - If it's a diagram/chart: Describe the data or flow (e.g., 'A bar chart showing...').
        - If it's a person/photo: Identify the subject and context (e.g., 'Andrew Ng speaking at the World Economic Forum').
        - If it's a logo: Say 'Logo of [Name]'.
        - dont say "No diagram, chart or specific AI symbol present.", instead say "image for [article name]"
        - MAXIMUM 15 words. Be literal and technical. No fluff like 'This is an image
        - Generate exactly ONE technical sentence. DO NOT YAP, max is 10 words
        - dont describe image as "The image from 'The Batch' AI newsletter "
        - dont describe visual features like color of image
        - describe only what image about, if image is logo, say "logo of and name"
        - sentence must be descriptive
        -dont generate sentence like "man with coffee sitting"
        - sentence must be descriptive and connected to title
        - Focus ONLY on the unique graphic representing the specific news items mentioned above.
        - Describe the graphic's layout (e.g., 'A bar chart showing...', 'A chemical diagram of...', 'A screenshot of a code interface...')."""

        messages = self.create_message(prompt, image_b64)

        try:
            out = await self.groq_client.chat.completions.create(
                messages=messages,
                model=self.model_id,
                **{**self.base_model_kwargs, "stream": False},
            )
            return out.choices[0].message.content.strip()
        except Exception:
            return "Illustration related to the AI news content."

    async def _ensure_cache_exists(self) -> None:
        """Forcefully checks and recreates the cache collection if missing."""
        try:
            response = await self.cache_client.get_collections()
            existing_names = [c.name for c in response.collections]

            if self.cache_collection not in existing_names:
                await self.cache_client.create_collection(
                    collection_name=self.cache_collection,
                    vectors_config=models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception:
            self._cache_initialized = False

    async def get_semantic_cache(self, question: str):
        await self._ensure_cache_exists()
        try:
            clean_question = question.strip().lower()

            vector = await asyncio.to_thread(self.text_model.encode, clean_question)
            vector_list = vector.tolist()

            response = await self.cache_client.query_points(
                collection_name=self.cache_collection,
                query=vector_list,
                limit=1,
                with_payload=True,
            )

            if response.points:
                score = response.points[0].score

                if score >= config.CACHE_THRESHOLD:
                    return response.points[0].payload["answer_data"]
            else:
                pass

        except Exception:
            pass
        return None

    async def set_semantic_cache(self, question: str, answer_data: dict) -> None:
        await self._ensure_cache_exists()
        try:
            clean_q = question.strip().lower()
            vector = await asyncio.to_thread(self.text_model.encode, clean_q)

            await self.cache_client.upsert(
                collection_name=self.cache_collection,
                points=[
                    PointStruct(
                        id=hashlib.md5(clean_q.encode()).hexdigest(),
                        vector=vector.tolist(),
                        payload={"question": clean_q, "answer_data": answer_data},
                    ),
                ],
            )
        except Exception:
            pass

    async def generate_variations(self, original_question: str):

        try:
            response = await self.query_expansion.ainvoke(
                {"question": original_question},
            )

            variations = [q.strip() for q in response.split("\n") if q.strip()]
            return variations[:3]  # Keep top 3
        except Exception:
            return [original_question]

    async def get_all_vecs(self,q):
        t = asyncio.to_thread(self.text_model.encode, q, normalize_embeddings=True)
        i = asyncio.to_thread(self.vision_model.encode, q, normalize_embeddings=True)
        s = asyncio.to_thread(list, self.sparse_model.embed([q]))
        return await asyncio.gather(t, i, s)

    async def run_hybrid_rag(
        self,
        query_str,
        bypass_cache: bool = False,
        category=None,
        mode="Hybrid",use_history=True
    ):
        if not bypass_cache:
            cached = await self.get_semantic_cache(query_str)
            if cached:
                return cached

        if category:
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value=category),
                    ),
                ],
            )

        if not use_history: #clear history only in testing
            self.chat_history.clear()

        async with self.lock:
            variations = await self.generate_variations(query_str)
            queries = [query_str, *variations]
            vector_results = await asyncio.gather(*[self.get_all_vecs(q) for q in queries])

            all_hits = []
            seen_ids = set()
            for idx, (t_vec, i_vec, s_res) in enumerate(vector_results):
                sparse_vec = models.SparseVector(indices=s_res[0].indices.tolist(), values=s_res[0].values.tolist())

                results = await self.client.query_points(
                    collection_name=self.CHILD_COLL,
                    prefetch=[
                        models.Prefetch(query=t_vec.tolist(), using="text", limit=self.QDRANT_LIMIT),
                        models.Prefetch(query=i_vec.tolist(), using="image", limit=self.QDRANT_LIMIT),
                        models.Prefetch(query=sparse_vec, using="text-sparse", limit=self.QDRANT_LIMIT),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    #query_points=query_filter,
                    limit=self.QDRANT_LIMIT,
                )
                for hit in results.points:
                    if hit.id not in seen_ids:
                        all_hits.append(hit)
                        seen_ids.add(hit.id)

            if not all_hits:
                return {
                    "headline": "No Results",
                    "answer": "I couldn't find any relevant hints in the database for this query.",
                    "confidence_score": 0,
                    "sources": []
                }

            try:
                pairs = [
                    [query_str, hit.payload.get("chunk_text", "")] for hit in all_hits
                ]
                scores = await asyncio.to_thread(self.reranker.predict, pairs)
                for i, hit in enumerate(all_hits):
                    hit.score = float(scores[i])
            except Exception:
                pass

        sorted_hits = sorted(all_hits, key=lambda x: x.score, reverse=True)
        best_score = sorted_hits[0].score

        fetch_tasks = []
        unique_p_ids = []
        seen_headlines = set()
        scores = []

        for hit in sorted_hits:
            p_id = hit.payload.get("parent_id")
            if hit.score < self.RERANK_LIMIT: #not add bad rerank chunks
                continue
            if p_id and p_id not in unique_p_ids:
                scores.append(hit.score)
                unique_p_ids.append(p_id)
                fetch_tasks.append(
                    self.client.retrieve(
                        collection_name=self.PARENT_COLL,
                        ids=[p_id],
                    ),
                )
            if len(unique_p_ids) >= self.CANDIDATES_LIMIT:
                break

        parent_results = await asyncio.gather(*fetch_tasks)

        # ---  SMALL-TO-BIG EXPANSION ---
        seen_ids = set()
        seen_parents_text_content = []
        seen_parents_payloads = []
        sources = []
        images_and_descriptions = []
        temp_imgs=[]
        image_tasks=[]

        for doc_list in parent_results:
            if doc_list:
                payload = doc_list[0].payload
                headline = payload.get("headline", "Untitled")
                if headline not in seen_headlines:
                    seen_headlines.add(headline)
                    seen_parents_payloads.append(payload)
                    idx = len(seen_parents_payloads)
                    sources.append({"title": headline, "url": payload.get("url")})
                    parent_doc = Document(
                        page_content=payload.get("full_text"),
                        metadata={
                            "index": idx,
                            "title": headline,
                        },
                    )
                    raw_img = payload.get("image_b64", "")
                    if len(raw_img) > 100:
                        pure_string = (
                            raw_img.split(",")[-1] if "," in raw_img else raw_img
                        )
                        pure_string = "".join(pure_string.split())
                        task = self.describe_image(pure_string, [headline])
                        image_tasks.append(task)
                        temp_imgs.append(pure_string)

                        # Only add to gallery if it's NOT a logo
                        # if "logo" not in desc.lower():
                        #     images_and_descriptions.append(
                        #         {
                        #             "image": pure_string,
                        #             "description": desc,
                        #         },
                        #     )

                    seen_parents_text_content.append(parent_doc)

        #async logo picking
        all_descs = await asyncio.gather(*image_tasks) if image_tasks else []
        images_and_descriptions = [
            {"image": i, "description": d}
            for i, d in zip(temp_imgs, all_descs)
            # check if 'd' is a string before checking for "logo"
            if isinstance(d, str) and "logo" not in d.lower()
        ]
        print(f"DEBUG: sorted_hits scores: {[h.score for h in sorted_hits]}")

        if not seen_parents_payloads:
            return {
                "headline": "No Relevant Context",
                "answer": "I couldn't find any relevant snippets in the database for this query.",
                "confidence_score": 0,
                "sources": []
            }


        combined_context = "\n\n---\n\n".join(
            [doc.page_content for doc in seen_parents_text_content],
        )

        top_parent = seen_parents_payloads[0]
        trimmed_history = self.trimer.invoke(self.chat_history.messages)

        if not seen_parents_payloads or not seen_parents_payloads[0].get(
            "full_text",
        ):
            return {
                "headline": "Error",
                "answer": "No context found to analyze.",
                "confidence_score": 0,
            }

        if (
            best_score < self.THRESHOLD
            and self.reranker_scores_report(scores).get("median_scores") < 0.1
        ):
            return {
                "headline": "No Direct Match Found",
                "answer": "I found some articles, but nothing specifically answering your question.",
                "confidence_score": best_score,
                "image_b64": "",
            }



        base_text = ""


        async with self.llm_lock:
            try:
                prompt_text = self.base_prompt.format(
                    context=combined_context,
                    question=query_str,
                    history=trimmed_history,
                )
                chat_completion = await self.groq_client.chat.completions.create(
                    messages=self.create_message(prompt_text, images_and_descriptions),
                    model=self.model_id,
                    **self.base_model_kwargs,
                )

                async for chunk in chat_completion:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        base_text += token

                judge_prompt_text = self.critique_chain.format(
                    context=combined_context,
                    question=query_str,
                    answer=base_text,
                )

                judge_completion = await self.groq_client.chat.completions.create(
                    messages=self.create_message(
                        judge_prompt_text,
                        images_and_descriptions,
                    ),
                    model=self.model_id,
                    **self.judge_model_kwargs,
                )
                final_answer = ""
                async for chunk in judge_completion:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        final_answer += token

            except (asyncio.TimeoutError, Exception):
                self.chat_history.clear()
                return {
                    "headline": "Service Timeout",
                    "answer": "⚠️ System is unresponsive. Please try again.",
                    "sources": []
                }

        if "⚠️" not in final_answer:
            self.chat_history.add_user_message(query_str)
            self.chat_history.add_ai_message(final_answer)
        final_answer = final_answer.strip()
        if final_answer.lower().startswith("here is"):
            final_answer = "\n".join(final_answer.split("\n")[1:]).strip()
            final_answer = final_answer.replace("*", "")

        self.reranker_scores_report(scores) or {}

        response_data = {
            "headline": top_parent.get("headline", "Untitled"),
            "answer": final_answer,
            "image_description": images_and_descriptions[0]["description"]
            if images_and_descriptions
            else "No diagrams found.",
            "gallery": images_and_descriptions,
            "url": top_parent.get("url", "#"),
            "type": top_parent.get("type"),
            "full_text": top_parent["full_text"],
            "confidence_score": self.reranker_scores_report(scores).get(
                "median_scores",
            ),
            "sources": sources,
            "context_text": [d.page_content for d in seen_parents_text_content],
            "generation_variations": [str(q) for q in queries],
        }
        if "⚠️" not in final_answer:
            await self.set_semantic_cache(query_str, response_data)
        return response_data

    async def refine_answer(
        self,
        question: str,
        original_answer: str,
        contexts: list,
        img,
    ):
        """Refines a hallucinated answer using a strict critique prompt."""
        context_block = "\n---\n".join([str(c) for c in contexts])
        refine_prompt = f"""
           You are a strict Machine Learning Tutor.
           The original answer below was flagged for containing information NOT found in the notes.

           QUESTION: {question}
           ORIGINAL ANSWER: {original_answer}

           STRICT CONTEXT FROM NOTES:
           {context_block}

           TASK:
           Rewrite the answer. Use ONLY the facts provided in the context above.
           If the context does not contain the answer, say 'I cannot verify this in my notes.'
           Keep the tone helpful but extremely literal.
           """

        refined_prompt = self.create_message(refine_prompt, img)
        judge_completion = await self.groq_client.chat.completions.create(
            messages=refined_prompt,
            model=self.model_id,
            **self.judge_model_kwargs,
        )
        response = ""
        async for chunk in judge_completion:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                response += token
        return response
