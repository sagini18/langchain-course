import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = UnstructuredLoader(
        file_path="/home/navaratnam-sagini/dev/space/langchain-course/src/langchain_course/medium_blog.txt",
        chuncking_strategy="basic",
        max_characters=100000,
    )
    documents = loader.load()

    print("splitting")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("Ingestion complete") 
    # https://app.pinecone.io/organizations/-P6IT42jAgoYLL78Qn-Z/projects/13e634f8-8555-4710-8d6e-a2739f976561/indexes/medium-blogs-embeddings-index/browser
