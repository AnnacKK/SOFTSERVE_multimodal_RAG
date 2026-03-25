import httpx, uuid, io, base64, time
from bs4 import BeautifulSoup
from PIL import Image
from qdrant_client import QdrantClient,models
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer, CrossEncoder
from urllib.parse import urljoin
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import config
from fastembed import SparseTextEmbedding
import re
import os


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = 6333

client = QdrantClient(host=qdrant_host, port=qdrant_port,timeout=60)


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

class QdrantScrapping:
    def __init__(self):
        self.client = client
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vision_model = SentenceTransformer(config.IMAGE_MODEL_NAME,device='cpu')
        self.text_model = SentenceTransformer(config.TEXT_MODEL_NAME, device='cpu')

        self.PARENT_COLL = config.PARENT_COLL
        self.CHILD_COLL = config.CHILD_COLL
        self.TARGET_COUNT = 200
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        self.qnt_config=models.BinaryQuantization(
    binary=models.BinaryQuantizationConfig(
        always_ram=True
    )
)
        # models.ScalarQuantization(
        #     scalar=models.ScalarQuantizationConfig(
        #         type=models.ScalarInt8,
        #         always_ram=True,
        #     ),
        # )

    def init_db(self):
        for coll in [self.PARENT_COLL, self.CHILD_COLL]:
            if client.collection_exists(coll): client.delete_collection(coll)
        client.create_collection(self.PARENT_COLL, vectors_config={"none": VectorParams(size=1, distance=Distance.COSINE)})
        client.create_collection(self.CHILD_COLL, vectors_config={"text": VectorParams(size=384, distance=Distance.COSINE,
                                                                                       hnsw_config=models.HnswConfigDiff(
                                                                                           m=16,  # Max links per node
                                                                                           ef_construct=100,
                                                                                           # Accuracy during indexing
                                                                                           full_scan_threshold=10000
                                                                                       ),

                                                                                       quantization_config=self.qnt_config),
                                                             "image": VectorParams(size=512, distance=Distance.COSINE,quantization_config=self.qnt_config)},
                                 sparse_vectors_config={
                                     "text-sparse": models.SparseVectorParams(index=models.SparseIndexParams(full_scan_threshold=1000))
                                 }
                                 )
        print("🚀 Database Initialized with HNSW Graph Index")

    def _process_and_resize_image(self, img_data, max_size=(128, 128)):
        """Resizes image and converts to JPEG to minimize base64 string size."""
        img = Image.open(io.BytesIO(img_data)).convert("L")

        v_img_input = img.resize((224, 224), Image.Resampling.LANCZOS)
        v_image = self.vision_model.encode(v_img_input, normalize_embeddings=True).tolist()

        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=70, optimize=True)
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        if len(img_b64) > 15000:
            print(f"⚠️ Warning: Large Payload ({len(img_b64)} chars). Consider lowering quality.")

        return img_b64, v_image


    def get_content(self,soup):
        """Aggressive content extraction with fallbacks."""
        for selector in ['.gh-content', '.gh-canvas', '.post-content', 'article', '.issue-content']:
            tag = soup.select_one(selector)
            if tag:

                for noise in tag(['script', 'style', 'nav', 'footer']): noise.decompose()
                text = tag.get_text(separator=" ", strip=True)
                # Strip the newsletter meta-data that confuses the LLM
                text = re.sub(r"Published\s+.*?min read", "", text, flags=re.IGNORECASE | re.DOTALL)
                text = text.replace("Share Loading... Player...", "").strip()
                if len(text) > 500: return text


        best_div = None
        max_ps = 0
        for div in soup.find_all('div'):
            p_count = len(div.find_all('p'))
            if p_count > max_ps:
                max_ps = p_count
                best_div = div

        if best_div:
            return best_div.get_text(separator=" ", strip=True)
        return ""



    def get_article_links(self,category_url, limit=200):
        """Navigates through pages to collect enough unique links."""
        links = set()
        page_num = 1

        while len(links) < limit:
            page_url = f"{category_url}page/{page_num}/" if page_num > 1 else category_url
            try:
                resp = httpx.get(page_url, headers=HEADERS, timeout=15)
                if resp.status_code != 200: break  # Stop if page doesn't exist

                soup = BeautifulSoup(resp.text, 'html.parser')
                page_links = {urljoin("https://www.deeplearning.ai", a['href'])
                              for a in soup.find_all('a', href=True)
                              if "/the-batch/" in a['href'] and not any(
                        x in a['href'] for x in ["/tag/", "/author/", "/page/"])}

                if not page_links: break
                links.update(page_links)
                print(f"   📄 Page {page_num}: Found {len(page_links)} links (Total: {len(links)})")
                page_num += 1
                time.sleep(1)
            except:
                break

        return list(links)[:limit]

    def scrape_and_ingest(self):
        print("STARTING")
        self.init_db()
        print("COLLECTION CREATED")
        for name, url in CATEGORIES.items():
            print(f"📂 Scanning category: {name}")
            articles_urls = self.get_article_links(url, limit=self.TARGET_COUNT)

            for article_url in articles_urls:
                try:
                    print(f"🔍 Processing: {article_url}")
                    page_resp = httpx.get(article_url, headers=HEADERS, timeout=15)
                    psoup = BeautifulSoup(page_resp.text, 'html.parser')
                    full_text = self.get_content(psoup)

                    if len(full_text) < 300: continue
                    headline = "Untitled"
                    og_title = psoup.find("meta", property="og:title")
                    if og_title:
                        # "News | The Batch" -> "News"
                        headline = og_title["content"].split('|')[0].strip()
                    else: headline = psoup.find('h2').text.strip() if psoup.find('h2') else "Untitled"


                    img_tag = psoup.find("meta", property="og:image")
                    if not img_tag: continue
                    img_data = httpx.get(img_tag["content"]).content

                    img_b64, v_image = self._process_and_resize_image(img_data)

                    parent_docs = self.parent_splitter.split_text(full_text)

                    all_child_texts = []
                    child_metadata_map = []

                    for p_text in parent_docs:
                        parent_id = str(uuid.uuid4())
                        client.upsert(
                            self.PARENT_COLL,
                            points=[PointStruct(
                                id=parent_id,
                                vector={"none": [0.0]},
                                payload={
                                    "full_text": p_text,
                                    "image_b64": img_b64,
                                    "headline": headline,
                                    "url": article_url,
                                    "type": name
                                }
                            )]
                        )


                        current_children = self.child_splitter.split_text(p_text)
                        for c_text in current_children:
                            all_child_texts.append(c_text)
                            child_metadata_map.append({"parent_id": parent_id, "text": c_text})


                    if all_child_texts:
                        print(f"  ⚡ Batch encoding {len(all_child_texts)} child chunks...")
                        all_embeddings = self.text_model.encode(all_child_texts, batch_size=config.CHUNK_BATCH_SIZE,
                                                                normalize_embeddings=True, show_progress_bar=False)

                        all_sparse = list(self.sparse_model.embed(all_child_texts))

                        child_points = []
                        for i, (dense_vec, sparse_obj) in enumerate(zip(all_embeddings,all_sparse)):
                            child_points.append(PointStruct(
                                id=str(uuid.uuid4()),
                                vector={
                                    "text": dense_vec.tolist(),
                                    "image": v_image,
                                    "text-sparse": models.SparseVector(
                                        indices=sparse_obj.indices.tolist(),
                                        values=sparse_obj.values.tolist())
                                },
                                payload={
                                    "parent_id": child_metadata_map[i]["parent_id"],
                                    "chunk_text": child_metadata_map[i]["text"]
                                }
                            ))

                        client.upsert(self.CHILD_COLL, points=child_points)

                    print(f"  ✅ DONE: {headline[:30]}")

                except Exception as e:
                    print(f"  ❌ Error processing {article_url}: {e}")




if __name__ == "__main__":
    processor = QdrantScrapping()
    processor.scrape_and_ingest()