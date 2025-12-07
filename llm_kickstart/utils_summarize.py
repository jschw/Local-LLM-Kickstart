import os
import urllib.request
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import zipfile
import numpy as np
from pathlib import Path
import appdirs

def load_glove_embeddings(dim=100):
    """
    Downloads GloVe 6B embeddings, extracts them, and loads the selected dimension.
    Returns: dict mapping word -> vector (np.array).
    """

    target_dir  = Path(appdirs.user_config_dir(appname='LLM_Kickstart'))
    target_dir.mkdir(parents=True, exist_ok=True)
    
    expected_file = os.path.join(target_dir, f"glove.6B.{dim}d.txt")
    
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "glove.6B.zip")
    
    # 1. Download if missing
    if not os.path.exists(expected_file):
        print("--> Downloading GloVe embeddings...")
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        urllib.request.urlretrieve(url, zip_path)
    else:
        print("--> GloVe already exists. Skipping download.")
    
    # 2. Extract if files missing
    if not os.path.exists(expected_file):
        print("--> Extracting GloVe files...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)

        # Delete zip and all unneccessary files
        os.remove(zip_path)
        os.remove(os.path.join(os.path.dirname(zip_path), "glove.6B.100d.txt"))
        os.remove(os.path.join(os.path.dirname(zip_path), "glove.6B.200d.txt"))
        os.remove(os.path.join(os.path.dirname(zip_path), "glove.6B.300d.txt"))
    
    # 3. Load embeddings into dict
    print(f"--> Loading {dim}D GloVe vectors...")
    embeddings = {}
    with open(expected_file, 'r', encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    
    print(f"--> Loaded {len(embeddings)} word vectors.")
    return embeddings

def generate_text_summary(text:list, embedding_model=None):
    if embedding_model == None:
        print("--> Please provide an embedding model.")
        return None

    # Adjust according to input embedding model
    vector_len = 50

    # split the the text in the articles into sentences
    sentences = []
    for s in text:
        sentences.append(sent_tokenize(s))  

    # flatten the list
    sentences = [y for x in sentences for y in x]

    # remove punctuations, numbers and special characters
    clean_sentences = [re.sub("[^a-zA-Z]", " ", s) for s in sentences]

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    stop_words = stopwords.words('english')

    # function to remove stopwords
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    # Formation of sentence vectors
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([embedding_model.get(w, np.zeros((vector_len,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((vector_len,))
        sentence_vectors.append(v)

    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,vector_len), sentence_vectors[j].reshape(1,vector_len))[0,0]
        

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    # Sorting and printing Summary
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    # Specify number of sentences to form the summary
    sn = 10
    summary_context = ""
    for i in range(sn):
        summary_context += f"{{\n{ranked_sentences[i][1]}\n}},\n"

    return summary_context
