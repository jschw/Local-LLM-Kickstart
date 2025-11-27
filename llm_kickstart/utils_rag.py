import hnswlib
from light_embed import TextEmbedding
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

def crawl_website(url: str, timeout: int) -> str :
    print("Crawling function")
    DEFAULT_TARGET_CONTENT = ['article', 'div', 'main', 'p']
    strip_elements = ['a']

    try:
        print(f"- Crawling: {url}")
        response = requests.get(url, timeout=timeout)
    except requests.exceptions.RequestException as e:
        print(f"-->  Request error for {url}: {e}")
        return None
    
    content_type = response.headers.get('Content-Type', '')
    print(f"Content type: {content_type}")

    if 'text/html' in content_type:
        # Create BS4 instance for parsing
        soup = BeautifulSoup(response.text, 'html.parser')

        # Strip unwanted tags
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

        # Return if text > 0
        if len(content) == 0:
            return None
        
        # Parse markdown
        output = md(
            content,
            keep_inline_images_in=['td', 'th', 'a', 'figure'],
            strip=strip_elements
        )
    
        print(f'--> Success')

        return output

    elif 'text/plain' in content_type:
        # Return if text > 0
        if len(response.text) == 0:
            return None
        
        # Parse markdown
        output = md(
            response.text,
            keep_inline_images_in=['td', 'th', 'a', 'figure'],
            strip=strip_elements
        )
    
        print(f'--> Success')

        return output
    
    elif 'application/pdf'in content_type:
        # TODO
        pass
    
    elif 'text/html' not in content_type:
        print(f"--> Content not text/html for {url}")
        return None

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
            self.vectorstore.init_index(max_elements=1000, ef_construction=200, M=48)
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

    def init_vectorstore_str(self, input, ref_depth=2):
        # Init vectorstore with website content
        if self.index_vectorstore(input):
            print("-> Vectorstore ready.")
            return True
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
    