import hnswlib
from light_embed import TextEmbedding
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class KickstartVectorsearch:
    def __init__(self):

        # Load and initialize embedding model
        self.embedding_model  = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')

        self.fallback_document_path = os.path.expanduser("~/LLM_Kickstart_Documents")

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

            # Split
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", " ", ""]
            )

            self.chunks = text_splitter.split_text(raw_text)
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

            print("-> Vectorstore ready.")

            return True
        
        except:
            return False

    
    def init_vectorstore_web(self, url, ref_depth=2):
        # ToDo
        pass
    
    def search_knn(self, prompt, num_chunks=4) -> list:
        new_embedding = self.embedding_model.encode([prompt], normalize_embeddings=True)

        # Fetch k neighbors
        chunk_ind, distances = self.vectorstore.knn_query(new_embedding, k=num_chunks)

        # De-Reference chunks
        result_chunks = []
        for ind in chunk_ind[0]:
            result_chunks.append(self.chunks[ind])

        return result_chunks
    