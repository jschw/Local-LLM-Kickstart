import hnswlib
from light_embed import TextEmbedding
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urljoin, urlparse

def crawl_website(url: str, timeout: int, max_depth: int = 1):
    """
    Recursively crawl a website starting from `url` up to `max_depth` link depth.
    Returns a list of (markdown_content, url) tuples for all visited pages.
    """

    DEFAULT_TARGET_CONTENT = ['article', 'div', 'main', 'p']
    strip_elements = ['a']
    visited = set()
    results = []

    def _crawl(current_url, depth):
        if depth > max_depth or current_url in visited:
            return
        visited.add(current_url)
        try:
            print(f"--> Crawling: {current_url} (depth {depth})")
            response = requests.get(current_url, timeout=timeout)
        except requests.exceptions.RequestException as e:
            print(f"-->  Request error for {current_url}: {e}")
            return

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
                return

            output = md(
                content,
                keep_inline_images_in=['td', 'th', 'a', 'figure'],
                strip=strip_elements
            )

            output = output.replace('\n','')
            output = output.replace('\t','')
            output = output.strip()
            if output:
                results.append((output, current_url))

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
                    _crawl(link, depth + 1)

        elif 'text/plain' in content_type:
            if len(response.text) == 0:
                return
            output = md(
                response.text,
                keep_inline_images_in=['td', 'th', 'a', 'figure'],
                strip=strip_elements
            )
            output = output.replace('\n','')
            output = output.replace('\t','')
            output = output.strip()
            if output:
                results.append((output, current_url))

        elif 'application/pdf' in content_type:
            # TODO: PDF crawling not implemented
            return
        else:
            print(f"--> Unknown content type for {current_url}.")
            return

    _crawl(url, 1)
    return results

class KickstartVectorsearch:
    def __init__(self):
 
        # Load and initialize embedding model
        self.embedding_model  = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')
        self.chunks = []
        self.chunk_metadata = []

        self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " ", ""]
            )

    def index_vectorstore(self, input, chunk_metadata=None):
        try:
            # Split
            self.chunks = self.text_splitter.split_text(input)
            print(f"-> Number of chunks: {len(self.chunks)}")

            # If metadata is provided, use it; else, fill with empty dicts
            if chunk_metadata is not None:
                self.chunk_metadata = chunk_metadata
            else:
                self.chunk_metadata = [{} for _ in self.chunks]

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
        
        except Exception as e:
            print(f"Error in index_vectorstore: {e}")
            return False

    def init_vectorstore_pdf(self, pdf_paths:list):
        print("-> Creating vectorstore index...")
        self.vectorstore = hnswlib.Index(space='cosine', dim=384)
        self.vectorstore.init_index(max_elements=10000, ef_construction=200, M=48)

        self.chunks = []
        self.chunk_metadata = []
        
        # Loop through path list
        for doc_path in pdf_paths:
            # Read and split PDF file
            print(f"-> Reading PDF file {doc_path}...")

            try:
                reader = PdfReader(doc_path)

                all_chunks = []
                all_metadata = []
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        # Split each page into chunks
                        page_chunks = self.text_splitter.split_text(text)
                        all_chunks.extend(page_chunks)
                        all_metadata.extend([
                            {"source_info": os.path.basename(doc_path), "source_position": page_num}
                            for _ in page_chunks
                        ])

                print("-> Number of chunks:", len(all_chunks))

                if len(all_chunks) == 0:
                    print("-> No text extracted from PDF.")
                    return False

                # Store chunks and metadata, and index
                self.chunks += all_chunks
                self.chunk_metadata += all_metadata

                # Create embeddings and index
                print(f"-> Creating embeddings for document {doc_path} ...")
                embeddings = self.embedding_model.encode(all_chunks, normalize_embeddings=True)
                print(f"-> Created embeddings for {len(all_chunks)} chunks.")

                self.vectorstore.add_items(embeddings)

            except Exception as e:
                print(f"--> Error while reading PDF: {e}")
                continue

        self.vectorstore.set_ef(50)

        print("-> Vectorstore ready.")
        return True

    def init_vectorstore_web(self, urls:list, deep=False):
        # Init vectorstore
        print("-> Creating vectorstore index...")
        self.vectorstore = hnswlib.Index(space='cosine', dim=384)
        self.vectorstore.init_index(max_elements=10000, ef_construction=200, M=48)

        self.chunks = []
        self.chunk_metadata = []

        for url in urls:

            if deep:
                ref_depth=2
                print(f"-> Deep crawling {url}.")
            else:
                ref_depth=1
                print(f"-> Crawling {url}.")

            # Init vectorstore with website content
            page_contents = crawl_website(url, 5, max_depth=ref_depth)

            if page_contents is not None and len(page_contents) > 0:
                all_chunks = []
                all_metadata = []

                for page_text, page_url in page_contents:
                    chunks = self.text_splitter.split_text(page_text)
                    all_chunks.extend(chunks)
                    all_metadata.extend([
                        {"source_info": page_url, "source_position": 0}
                        for _ in chunks
                    ])

                self.chunks += all_chunks
                self.chunk_metadata += all_metadata

                # Create embeddings and index
                print(f"-> Creating embeddings for {url}...")
                embeddings = self.embedding_model.encode(all_chunks, normalize_embeddings=True)
                print(f"-> Created embeddings for {len(all_chunks)} chunks.")

                self.vectorstore.add_items(embeddings)

            else:
                print(f"-> Page {url} contains no data, skipped.")
                continue

        self.vectorstore.set_ef(50)
        
        print("-> Vectorstore ready.")
        return True
    
    def search_knn(self, prompt, num_chunks=4) -> list:
        new_embedding = self.embedding_model.encode([prompt], normalize_embeddings=True)

        # Fetch k neighbors
        chunk_ind, distances = self.vectorstore.knn_query(new_embedding, k=num_chunks)

        # De-Reference chunks and metadata
        results = []

        for i, ind in enumerate(chunk_ind[0]):
            chunk = self.chunks[ind]
            meta = self.chunk_metadata[ind] if hasattr(self, "chunk_metadata") and len(self.chunk_metadata) > ind else {}
            similarity = 1 - distances[0][i]
            results.append({
                "chunk": chunk,
                "source_info": meta.get("source_info", None),
                "source_position": meta.get("source_position", None),
                "similarity": similarity
            })

        return results
    