import asyncio
import httpx
import json
import logging
import random
import os
import re
import sys
from typing import List, Dict, Optional, Set
from bs4 import BeautifulSoup
from groq import AsyncGroq
from src.config import config

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CATEGORIES = {
    "Business": "https://www.deeplearning.ai/the-batch/tag/business/",
    "AI Careers": "https://www.deeplearning.ai/the-batch/tag/ai-careers/",
    "Data Points": "https://www.deeplearning.ai/the-batch/tag/data-points/",
    "ML Research": "https://www.deeplearning.ai/the-batch/tag/research/",
    "Weekly Issues": "https://www.deeplearning.ai/the-batch/",
    "Andrew's Letters": "https://www.deeplearning.ai/the-batch/tag/letters/",
    "Science": "https://www.deeplearning.ai/the-batch/tag/science/",
    "Hardware": "https://www.deeplearning.ai/the-batch/tag/hardware/",
    "Culture": "https://www.deeplearning.ai/the-batch/tag/culture/"
}


class LoRABatchScraper:
    def __init__(self, target_per_category: int = 25, checkpoint_path: str = "data_processing/lora_train_data.jsonl"):
        self.target_per_category = target_per_category
        self.checkpoint_path = checkpoint_path
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MLOps-Scraper/2.5"}
        self.dataset: List[Dict] = []
        self.processed_urls: Set[str] = self._load_checkpoint_urls()
        self.request_cooldown = 12.0

        # Rate limiting control: prevent overwhelming Groq
        self.semaphore = asyncio.Semaphore(2)

        self.groq_key = getattr(config, 'GR_TOKEN', None) or os.getenv("GR_TOKEN")
        if not self.groq_key:
            raise ValueError("Missing GR_TOKEN environment variable.")
        self.groq_client = AsyncGroq(api_key=self.groq_key)

    def _load_checkpoint_urls(self) -> Set[str]:
        """Loads already processed URLs to avoid redundant API calls."""
        urls = set()
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'source' in data:
                            urls.add(data['source'])
                    except json.JSONDecodeError:
                        continue
        logging.info(f"Loaded {len(urls)} existing entries from checkpoint.")
        return urls

    async def close(self) -> None:
        await self.groq_client.close()
        logging.info("Cleaned up Groq client sessions.")

    async def fetch_page(self, client: httpx.AsyncClient, url: str) -> Optional[str]:
        try:
            response = await client.get(url, follow_redirects=True, timeout=15.0)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.text
        except Exception as e:
            logging.error(f"Network Error [{url}]: {e}")
            return None

    def get_content(self, soup: BeautifulSoup) -> str:
        selectors = ['.gh-content', '.gh-canvas', '.post-content', 'article', '.issue-content']
        for selector in selectors:
            tag = soup.select_one(selector)
            if tag:
                for noise in tag(['script', 'style', 'nav', 'footer', 'header']):
                    noise.decompose()
                text = tag.get_text(separator=" ", strip=True)
                text = re.sub(r"Published\s+.*?min read", "", text, flags=re.IGNORECASE | re.DOTALL)
                text = text.replace("Share Loading... Player...", "").strip()
                if len(text) > 500: return text
        return ""

    async def get_all_links_paginated(self, client: httpx.AsyncClient, base_tag_url: str) -> List[str]:
        """Iterates through pages to find enough links."""
        all_links = []
        page_num = 1

        while len(all_links) < self.target_per_category:
            url = base_tag_url if page_num == 1 else f"{base_tag_url.rstrip('/')}/page/{page_num}/"
            html = await self.fetch_page(client, url)
            if not html: break

            soup = BeautifulSoup(html, 'html.parser')
            page_links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                # Link validation for DeepLearning.ai structure
                if '/the-batch/' in href and any(char.isdigit() for char in href):
                    full_link = href if href.startswith('http') else f"https://www.deeplearning.ai{href}"
                    page_links.append(full_link)

            unique_page_links = list(set(page_links))
            if not unique_page_links: break

            all_links.extend(unique_page_links)
            all_links = list(set(all_links))
            page_num += 1
            await asyncio.sleep(0.5)

        return all_links[:self.target_per_category]

    async def generate_summary(self, text: str, category: str, max_retries: int = 3) -> Optional[str]:
        async with self.semaphore:
            # Enforce a mandatory wait to stay under RPM limits
            await asyncio.sleep(self.request_cooldown)

            # Use 8b-instant for higher throughput if 70b is constantly 429ing
            # model_name = "llama-3.1-8b-instant"
            model_name = "moonshotai/kimi-k2-instruct"

            for i in range(max_retries):
                try:
                    chat_completion = await self.groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Summarize news into one dense, professional paragraph."},
                            {"role": "user", "content": f"Category: {category}\nText: {text[:3000]}"}
                        ],
                        model=model_name,
                        temperature=0.1,
                    )
                    return chat_completion.choices[0].message.content
                except Exception as e:
                    if "429" in str(e):
                        # Adaptive backoff: Increase cooldown if we hit a 429 despite the semaphore
                        self.request_cooldown += 5.0
                        wait = (2 ** i) + random.uniform(5, 10)
                        logging.warning(f"Rate Limit Hit. New Cooldown: {self.request_cooldown}s. Sleeping {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        logging.error(f"LLM Error: {e}")
                        return None
            return None

    def _append_to_checkpoint(self, entry: Dict):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        with open(self.checkpoint_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    async def run(self):
        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            for category, tag_url in CATEGORIES.items():
                logging.info(f"🚀 Processing Category: {category}")
                article_links = await self.get_all_links_paginated(client, tag_url)

                count = 0
                for link in article_links:
                    # Skip if already processed in previous runs
                    if link in self.processed_urls:
                        logging.info(f"⏭️ Skipping (already processed): {link}")
                        continue

                    html = await self.fetch_page(client, link)
                    if not html: continue

                    text = self.get_content(BeautifulSoup(html, 'html.parser'))
                    if text:
                        summary = await self.generate_summary(text, category)
                        if summary:
                            entry = {
                                "instruction": f"Summarize the following {category} news.",
                                "context": text[:3500],
                                "response": summary,
                                "source": link
                            }
                            self.dataset.append(entry)
                            self._append_to_checkpoint(entry)
                            self.processed_urls.add(link)
                            count += 1
                            logging.info(f"✅ [{category}] {count}/{self.target_per_category}")

                    await asyncio.sleep(random.uniform(1.0, 2.0))

    def save_final_json(self, filename: str = "data_processing/lora_train_data.json"):
        # Re-read everything from checkpoint to ensure full dataset
        final_list = []
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                for line in f:
                    final_list.append(json.loads(line))

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(final_list, f, ensure_ascii=False, indent=4)
        logging.info(f"💾 Final dataset saved: {len(final_list)} samples.")


async def main():
    scraper = LoRABatchScraper(target_per_category=40)
    try:
        await scraper.run()
        scraper.save_final_json()
    except Exception as e:
        logging.critical(f"Main Loop Failure: {e}")
    finally:
        await scraper.close()


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting...")