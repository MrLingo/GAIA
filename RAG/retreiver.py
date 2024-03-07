import chromadb
from operator import itemgetter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


EMBEDDINGS_MODEL = 'all-MiniLM-L6-v2'
VECTOR_DB_PATH = r'RAG\vector_db'

# Init client.
persistent_client = chromadb.PersistentClient()
embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDINGS_MODEL)

# Directly query, no need to chunk every time.
vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function)        
           
def retrieve(query: str):    
    vectorstore = vector_db.similarity_search_with_score(query)
    
    # Retrieve the one with the highest score.
    highest_score_embedding = max(vectorstore, key=itemgetter(1))

    for document in vectorstore:
        if highest_score_embedding[1] == document[1]:            
            return document[0].page_content.replace('    ', '').replace('\t', ' ').replace('\n', '')                     