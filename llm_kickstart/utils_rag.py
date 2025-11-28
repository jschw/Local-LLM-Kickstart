import hnswlib
from light_embed import TextEmbedding
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urljoin, urlparse

def crawl_website(url: str, timeout: int, max_depth: int = 1) -> str:
    """
    Recursively crawl a website starting from `url` up to `max_depth` link depth.
    Returns concatenated markdown content from all visited pages.
    """

    DEFAULT_TARGET_CONTENT = ['article', 'div', 'main', 'p']
    strip_elements = ['a']
    visited = set()

    def _crawl(current_url, depth):
        if depth > max_depth or current_url in visited:
            return ""
        visited.add(current_url)
        try:
            print(f"--> Crawling: {current_url} (depth {depth})")
            response = requests.get(current_url, timeout=timeout)
        except requests.exceptions.RequestException as e:
            print(f"-->  Request error for {current_url}: {e}")
            return ""
        
        content_type = response.headers.get('Content-Type', '')

        if 'text/html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(['script', 'style']):
                script.decompose()

            max_text_length = 0
            main_content = ""
            for tag in soup.find_all(DEFAULT_TARGET_CONTENT):
                text_length = len(tag.get_text())
                if text_length > max_text_length:
                    max_text_length = text_length
                    main_content = tag

            content = str(main_content)
            if len(content) == 0:
                return ""

            output = md(
                content,
                keep_inline_images_in=['td', 'th', 'a', 'figure'],
                strip=strip_elements
            )

            # Recursively crawl links if depth < max_depth
            if depth < max_depth:
                links = set()
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    # Only follow http(s) links, resolve relative URLs
                    joined = urljoin(current_url, href)
                    parsed = urlparse(joined)
                    if parsed.scheme in ['http', 'https']:
                        links.add(joined)
                
                for link in links:
                    print(f"Parsing sublink: {link}")
                    output += "\n\n" + _crawl(link, depth + 1)

            output = output.replace('\n','')
            output = output.replace('\t','')
            output = output.strip()
            return output

        elif 'text/plain' in content_type:
            if len(response.text) == 0:
                return ""
            output = md(
                response.text,
                keep_inline_images_in=['td', 'th', 'a', 'figure'],
                strip=strip_elements
            )
            output = output.replace('\n','')
            output = output.replace('\t','')
            output = output.strip()
            return output

        elif 'application/pdf' in content_type:
            # TODO: PDF crawling not implemented
            return ""
        else:
            print(f"--> Unknown content type for {current_url}.")
            return ""

    return _crawl(url, 1).strip()

class KickstartVectorsearch:
    def __init__(self):

        # Load and initialize embedding model
        self.embedding_model  = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')

    def index_vectorstore(self, input):
        try:
            # Split
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " ", ""]
            )

            self.chunks = text_splitter.split_text(input)
            print(f"Number of chunks: {len(self.chunks)}")

            # Create embeddings
            print("-> Creating embeddings...")
            embeddings = self.embedding_model.encode(self.chunks, normalize_embeddings=True)
            
            print(f"-> Created embeddings for {len(self.chunks)} chunks.")

            # Create index
            print("-> Creating vectorstore index...")
            self.vectorstore = hnswlib.Index(space='cosine', dim=384)
            self.vectorstore.init_index(max_elements=8000, ef_construction=200, M=48)
            self.vectorstore.add_items(embeddings)

            # Controlling the recall by setting ef
            self.vectorstore.set_ef(50)

            return True
        
        except:
            return False

    def init_vectorstore_pdf(self, pdf_path):
        # Read and split PDF file
        print(f"-> Reading PDF file {pdf_path}...")

        try:
            reader = PdfReader(pdf_path)

            raw_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n"

            print("-> Length of extracted PDF text:", len(raw_text))

            if self.index_vectorstore(raw_text):
                print("-> Vectorstore ready.")
                return True
            else:
                return False
        
        except:
            return False

    def init_vectorstore_web(self, url, ref_depth=2):
        # Init vectorstore with website content
        website_content = crawl_website(url, 5, max_depth=ref_depth)

        if website_content is not None:
            if self.index_vectorstore(website_content):
                print("-> Vectorstore ready.")
                return True
            else:
                return False
        else:
            return False
    
    def search_knn(self, prompt, num_chunks=4) -> list:
        new_embedding = self.embedding_model.encode([prompt], normalize_embeddings=True)

        # Fetch k neighbors
        chunk_ind, distances = self.vectorstore.knn_query(new_embedding, k=num_chunks)

        # De-Reference chunks
        result_chunks = []
        for ind in chunk_ind[0]:
            result_chunks.append(self.chunks[ind])

        return result_chunks
    