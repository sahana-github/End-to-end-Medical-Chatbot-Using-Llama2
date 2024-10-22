from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


# Initialize Pinecone
pc = Pinecone()
index_name = "medical-chatbot"

# Initialize LangchainPinecone
docsearch = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)

