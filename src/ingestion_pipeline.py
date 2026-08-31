import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def main():
    docs = load_documents("docs")

    chunks = split_documents(docs, 800, 0)
    embeddingModel = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = create_vector_store(chunks, embeddingModel)





def load_documents(docs_path):
    if not os.path.exists(docs_path):
        raise FileNotFoundError("Path not found")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls = TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError("No files found")

    return documents


def split_documents(documents, chunk_size,chunk_overlap=0):
    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

def create_vector_store(chunks, embeddingModel, persist_directory = "db/chroma_db"):

    vectorStore = Chroma.from_documents(
        documents = chunks,
        embedding = embeddingModel,
        persist_directory = persist_directory,
        collection_metadata ={"hnsw:space":"cosine"}
    )

    return vectorStore




if __name__ == "__main__":
    main()