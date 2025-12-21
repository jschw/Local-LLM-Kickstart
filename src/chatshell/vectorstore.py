import hnswlib
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import networkx as nx
import numpy as np
from light_embed import TextEmbedding
from .utils_rag import crawl_website

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ChatshellVectorsearch:
    def __init__(self):

        # Load and initialize embedding model
        self.chunks = []
        self.chunk_metadata = []
        self.context_list = []

        self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " ", ""]
            )
        
        self.embedding_model = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')

    def reset_context(self):
        self.context_list = []

    def get_context(self):
        context = ""
        for i in self.context_list:
            context += f"{{\n{i}\n}},\n"

        return context
    
    def add_context(self, input):
        self.context_list.append(input)
    
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
    
    def init_vectorstore_str(self, clipboard_string):
        print("-> Creating vectorstore index...")
        self.vectorstore = hnswlib.Index(space='cosine', dim=384)
        self.vectorstore.init_index(max_elements=10000, ef_construction=200, M=48)

        self.chunks = []
        self.chunk_metadata = []
        
        try:
            # Chunking clipboard string
            if clipboard_string:
                # Split each page into chunks
                clipboard_chunks = self.text_splitter.split_text(clipboard_string)
                self.chunks += clipboard_chunks

            print("-> Number of chunks:", len(self.chunks))

            if len(self.chunks) == 0:
                print("-> No text extracted from clipboard.")
                return False

            # Create embeddings and index
            print(f"-> Creating embeddings for clipboard context ...")
            embeddings = self.embedding_model.encode(self.chunks, normalize_embeddings=True)
            print(f"-> Created embeddings for {len(self.chunks)} chunks.")

            self.vectorstore.add_items(embeddings)

        except Exception as e:
            print(f"--> Error while creating vectorstore from clipboard: {e}")
            return False

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
    
    def generate_text_summary(self, text: list):

        # Split into sentences
        sentences = []
        for s in text:
            sentences.extend(sent_tokenize(s))

        # Remove empty sentences or whitespace-only sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) == 0:
            return ""

        # --- Encode sentences using transformer model ---

        sentence_vectors = self.embedding_model.encode(
            sentences,
            normalize_embeddings=True  # Unit-length vectors -> use of fast dot-product instead of cosine-similarity
        )

        # Build similarity matrix (FAST)
        sim_mat = np.dot(sentence_vectors, sentence_vectors.T)

        # Remove self-similarity
        np.fill_diagonal(sim_mat, 0)

        # Build graph and compute PageRank
        nx_graph = nx.from_numpy_array(sim_mat)
        print("-> Running pagerank algorithm")
        scores = nx.pagerank(nx_graph)

        # Rank sentences
        ranked_sentences = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)),
            reverse=True
        )

        # Number of sentences in summary
        sn = min(10, len(ranked_sentences))
        
        summary_context = ""
        for i in range(sn):
            summary_context += f"{{\n{ranked_sentences[i][1]}\n}},\n"

        return summary_context
    
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
    