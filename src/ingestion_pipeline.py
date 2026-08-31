import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def main():
    docs = load_documents("docs")




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





if __name__ == "__main__":
    main()