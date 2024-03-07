from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

THE_VITAL_QUESTION_PDF_PATH = r'.\documents\the_vital_question.pdf'
EMBEDDINGS_MODEL = 'all-MiniLM-L6-v2'
VECTOR_DB_PATH = r'.\vector_db'

loader = PyPDFLoader(THE_VITAL_QUESTION_PDF_PATH)
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    # 250 chars documents - 170 left for user input.    
    chunk_size = 250,     
    chunk_overlap = 0,
    length_function = len,
)

docs = text_splitter.split_documents(pages)

embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDINGS_MODEL)

# Load it into Chroma and store locally.
Chroma.from_documents(docs, embedding_function, persist_directory=VECTOR_DB_PATH)