
import logging
from config.logging_config import setup_logging
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_corpus():
    return ["Document 1", "Document 2", "Document 3"]

def load_documents():
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    logger.info(f"Total characters: {len(docs[0].page_content)}")
    logger.info(docs[0].page_content[:500])
    return docs

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    split_documents = text_splitter.split_documents(documents)
    logger.info(f"Split blog post into {len(split_documents)} sub-documents.")
    return split_documents

def embed_and_store_documents(split_documents, vector_store):
    document_ids = vector_store.add_documents(documents=split_documents)
    logger.info(document_ids[:100])

def setup_rag():
    corpus = load_corpus()
    model = init_chat_model("gpt-4.1")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore.from_texts(corpus, embeddings)
    documents = load_documents()
    split_docs = split_documents(documents)
    embed_and_store_documents(split_docs, vector_store)

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Set up logger")
    load_dotenv()
    setup_rag()